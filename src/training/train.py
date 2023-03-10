import contextlib
import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_cast_dtype, CLIP, CustomTextCLIP
from .distributed import is_master
from .zero_shot import zero_shot_eval
from .precision import get_autocast


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }

class Batch(dict):
    """Batch wrapper for a dictionary of tensors."""

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

    def size(self):
        if self.keys():
            k = list(self.keys())[0]
            return len(self[k])
        return 0

    def to(self, *, device, non_blocking=False, dtypes={}):
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                dtype = dtypes.get(k, v.dtype)
                self[k] = v.to(device=device, non_blocking=non_blocking, dtype=dtype)
            elif isinstance(v, dict):
                self[k] = Batch(v).to(device=device, non_blocking=non_blocking, dtypes=dtypes)
        return self


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    is_flava = args.model.startswith('flava')


    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    # FLAVA unimodal dataloaders
    if is_flava and args.flava_unimodal_mae:
        data['flava-mae'].set_epoch(epoch)
        mae_dataloader = iter(data['flava-mae'].dataloader)
    if is_flava and args.flava_unimodal_mlm:
        data['flava-mlm'].set_epoch(epoch)
        mlm_dataloader = iter(data['flava-mlm'].dataloader)

    if args.accum_freq > 1:
        assert not args.model.startswith('flava'), 'FLAVA does not support gradient accumulation'
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        dtypes = {'image': cast_dtype}
        batch = Batch(**batch).to(device=device, non_blocking=True, dtypes=dtypes)

        # FLAVA unimodal batches
        if is_flava and args.flava_unimodal_mae:
            try:
                mae_batch = next(mae_dataloader)
            except:
                mae_dataloader = iter(data['flava-mae'].dataloader)
                mae_batch = next(mae_dataloader)
            mae_batch = Batch(**mae_batch).to(device=device, non_blocking=True, dtypes=dtypes)

        if is_flava and args.flava_unimodal_mlm:
            try:
                mlm_batch = next(mlm_dataloader)
            except:
                mlm_dataloader = iter(data['flava-mlm'].dataloader)
                mlm_batch = next(mlm_dataloader)
            mlm_batch = Batch(**mlm_batch).to(device=device, non_blocking=True, dtypes=dtypes)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            no_grad_sync = model.no_sync if args.distributed else contextlib.nullcontext
            losses_dict = {}

            # FLAVA unimodal forward passes
            if is_flava and args.flava_unimodal_mlm:
                with no_grad_sync():
                    with autocast():
                        mlm_out = model(**mlm_batch, unimodal_mlm=True)
                        mlm_losses = loss.forward_mlm(**mlm_out)
                        total_mlm_loss = sum(mlm_losses.values())
                        losses_dict["unimodal_mlm_loss"] = total_mlm_loss
                    backward(total_mlm_loss, scaler)

            if is_flava and args.flava_unimodal_mae:
                with no_grad_sync():
                    with autocast():
                        mae_out = model(**mae_batch, unimodal_mae=True)
                        mae_losses = loss.forward_mae(**mae_out)
                        total_mae_loss = sum(mae_losses.values())
                        losses_dict["unimodal_mae_loss"] = total_mae_loss
                    backward(total_mae_loss, scaler)

            # Multimodal forward pass
            with autocast():
                model_out = model(**batch, output_dict=True)
                assert isinstance(model_out, dict)
                if args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(images, texts)
                    model_out.update({f'dist_{k}' : v for k, v in dist_model_out.items()})
                losses = loss(**model_out, output_dict=True)
                if isinstance(losses, dict):
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss
                else:
                    total_loss = losses
                    losses = {"loss": losses}
                losses_dict.update(losses)
            backward(total_loss, scaler)

        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    output = model(**batch, output_dict=True)
                    assert isinstance(output, dict)
                    chunk_image_features = output["image_features"]
                    chunk_text_features = output["text_features"]
                accum_image_features.append(chunk_image_features)
                accum_text_features.append(chunk_text_features)

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(image=images, text=texts, output_dict=True)
                    logit_scale = model_out.pop("logit_scale")
                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] +  [model_out[key]] + accumulated[j + 1:])
                    losses_dict = loss(**inputs, logit_scale=logit_scale, output_dict=True)
                    del inputs
                    total_loss = sum(losses_dict.values())
                    losses_dict["loss"] = total_loss
                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and ((i_accum % args.log_every_n_steps) == 0 or batch_count == num_batches_per_epoch):
            batch_size = batch.size()
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses_dict.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale = model_out['logit_scale']
            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name:val.val for name,val in losses_m.items()})

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def evaluate(model, data, epoch, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                batch = Batch(**batch).to(device=device, non_blocking=True, dtypes={'image': cast_dtype})

                with autocast():
                    output = model(**batch, output_dict=True)
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    image_features, text_features, logit_scale = output['image_features'], \
                                                                 output['text_features'], \
                                                                 output['logit_scale']
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = batch.size()
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(output)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch})

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
