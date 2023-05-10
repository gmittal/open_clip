import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import (
    DataCollatorForLanguageModeling,
    BertTokenizer,
    RobertaTokenizer,
    RobertaTokenizerFast,
    XLMRobertaTokenizer,
)

from .tokenizer import HFTokenizer


def tolist(x):
    if isinstance(x, list):
        return x
    elif hasattr(x, "numpy"):  # Checks for TF tensors without needing the import
        x = x.numpy()
    return x.tolist()


def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import numpy as np
    import torch

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result


@dataclass
class DataCollatorForWholeWordMask(DataCollatorForLanguageModeling):
    """
    https://github.com/RikVN/transformers/blob/0141e8bc61bc4f63f1b2b3b75440467feeee88d7/src/transformers/data/data_collator.py#L849
    """

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        if isinstance(examples[0], Mapping):
            input_ids = [e["input_ids"] for e in examples]
        else:
            input_ids = examples
            examples = [{"input_ids": e} for e in examples]

        batch_input = _torch_collate_batch(input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)

        mask_labels = []
        for e in examples:
            ref_tokens = []
            for id in tolist(e["input_ids"]):
                token = self.tokenizer._convert_id_to_token(id)
                ref_tokens.append(token)

            # For Chinese tokens, we need extra inf to mark sub-word, e.g [喜,欢]-> [喜，##欢]
            if "chinese_ref" in e:
                ref_pos = tolist(e["chinese_ref"])
                len_seq = len(e["input_ids"])
                for i in range(len_seq):
                    if i in ref_pos:
                        ref_tokens[i] = "##" + ref_tokens[i]
            mask_labels.append(self._whole_word_mask(ref_tokens))
        batch_mask = _torch_collate_batch(mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        inputs, labels = self.torch_mask_tokens(batch_input, batch_mask)
        return {"input_ids": inputs, "labels": labels}

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """
        cand_indexes = []
        special_tokens = [val for key, val in self.tokenizer.special_tokens_map.items()
                          if key not in ['unk_token', 'mask_token']]
        is_bert_tokenizer = isinstance(self.tokenizer, BertTokenizer)
        is_roberta_tokenizer = isinstance(self.tokenizer, (RobertaTokenizer, RobertaTokenizerFast))
        is_xlm_roberta_tokenizer = isinstance(self.tokenizer, XLMRobertaTokenizer)
        for (i, token) in enumerate(input_tokens):
            if token in special_tokens:
                continue

            if is_bert_tokenizer:
                if len(cand_indexes) >= 1 and token.startswith("##"):
                    cand_indexes[-1].append(i)
                else:
                    cand_indexes.append([i])
            elif is_roberta_tokenizer:
                # If a token doesn't start with Ġ, it's part of the previous token
                if len(cand_indexes) >= 1 and not token.startswith("Ġ"):
                    cand_indexes[-1].append(i)
                else:
                    cand_indexes.append([i])
            elif is_xlm_roberta_tokenizer:
                # If a token doesn't start with ▁, it's part of the previous token
                if len(cand_indexes) >= 1 and not token.startswith("▁"):
                    cand_indexes[-1].append(i)
                else:
                    cand_indexes.append([i])
            else:
                raise ValueError("Whole-word masking only implemented for BERT/RoBERTa/XLM-Roberta so far")

        if len(cand_indexes[-1]) == 0:
            cand_indexes = cand_indexes[:-1]

        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        if len(covered_indexes) != len(masked_lms):
            raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels

    def torch_mask_tokens(self, inputs: Any, mask_labels: Any) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


def get_flava_collate(hf_tokenizer, mlm_prob=0.15, whole_word=False):
    assert isinstance(hf_tokenizer, HFTokenizer), 'tokenizer must be HFTokenizer'
    tokenizer = hf_tokenizer.tokenizer
    mlm_collate_cls = DataCollatorForWholeWordMask if whole_word else DataCollatorForLanguageModeling
    mlm_collator = mlm_collate_cls(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_prob,
    )

    def collate(example_list):
        image_list, text_list = [], []
        for example in example_list:
            image_list.append(example["image"])
            text_list.append(example["text"])
        image = torch.stack(image_list)
        text = torch.stack(text_list)
        batch = {
            "image": image,
            "text": text,
        }

        # TODO: add whole word masking
        mlm_input = mlm_collator(text_list)  # TODO: add special_tokens_mask (improves efficiency)
        batch.update({
            "text_masked": mlm_input["input_ids"],
            "mlm_labels": mlm_input["labels"],
        })

        return batch

    return collate


def get_mlm_collate(hf_tokenizer, mlm_prob=0.15, whole_word=False):
    assert isinstance(hf_tokenizer, HFTokenizer), 'tokenizer must be HFTokenizer'
    tokenizer = hf_tokenizer.tokenizer
    mlm_collate_cls = DataCollatorForWholeWordMask if whole_word else DataCollatorForLanguageModeling
    mlm_collator = mlm_collate_cls(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_prob,
    )

    def collate(example_list):
        mlm_input = mlm_collator([example["text"] for example in example_list])
        return {
            "unimodal_text_masked": mlm_input["input_ids"],
            "unimodal_mlm_labels": mlm_input["labels"],
        }

    return collate
