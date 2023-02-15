import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import open_clip
from open_clip.factory import get_tokenizer
from training.data import get_imagenet
from training.scheduler import cosine_lr
from training.train import AverageMeter
from training.zero_shot import zero_shot_eval

try:
    import evaluate
except ImportError:
    raise ImportError("Please install HF evaluate: pip install evaluate")


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hateful-memes",
        type=str,
        default=None,
        help="Path to Hateful Memes dataset directory.",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size per GPU."
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RN50",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--pretrained",
        default='',
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )

    args = parser.parse_args(args)
    return args


class HatefulMemesDataset(Dataset):
    """Hateful Memes dataset."""

    def __init__(self, path, split, transforms, tokenizer=None):
        super().__init__()

        if split == "train":
            jsonl = os.path.join(path, "train.jsonl")
        elif split == "validation":
            jsonl = os.path.join(path, "dev_unseen.jsonl")
        elif split == "test":
            jsonl = os.path.join(path, "test_unseen.jsonl")
        else:
            raise ValueError(f"Invalid split {split}")

        self.path = path
        self.df = pd.read_json(jsonl, lines=True)
        self.length = len(self.df)
        self.transforms = transforms
        self.tokenize = tokenizer

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        img_path = os.path.join(self.path, item["img"])
        image = self.transforms(Image.open(str(img_path)))
        text = item["text"]
        label = item["label"]
        return {
            'image': image,
            'text': self.tokenize([text])[0],
            'label': label
        }


def get_task_dataloaders(transforms, args):
    tokenizer = get_tokenizer(args.model)
    hm_path = os.path.expanduser(args.hateful_memes)

    dataloaders = {}
    for split_name in ["train", "validation", "test"]:
        dataset = HatefulMemesDataset(
            hm_path,
            split_name,
            transforms,
            tokenizer=tokenizer,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
        )
        dataloaders[split_name] = dataloader

    return dataloaders


def compute_metrics(model, dataloader, device, args):
    model.eval()
    metric = evaluate.load("roc_auc")
    val_loss = 0
    samples_seen = 0
    for batch in dataloader:
        with torch.no_grad():
            image = batch["image"].to(device)
            text = batch["text"].to(device)
            label = batch["label"].to(device)
            samples_seen += text.shape[0]
            logits = model(image, text)
            logits = logits.view(-1)
            label = label.view(-1).float()
            pred_scores = torch.sigmoid(logits)
            batch_val_loss = nn.functional.binary_cross_entropy_with_logits(logits, label, reduction='sum')
        val_loss += batch_val_loss.item()
        metric.add_batch(
            prediction_scores=pred_scores.cpu().numpy(),
            references=label.cpu().numpy(),
        )
    model.train()
    metrics = metric.compute()
    metrics["loss"] = val_loss / samples_seen
    return metrics


def main(args):
    args = parse_args(args)
    random_seed(args.seed, 0)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, preprocess_train, preprocess_val = open_clip.factory.create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        pretrained_hf=False,
    )
    model_cfg = open_clip.factory.get_model_config(args.model)
    embed_dim = model_cfg["embed_dim"]

    data = get_task_dataloaders(preprocess_val, args)

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv[1:])
