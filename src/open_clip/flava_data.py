import torch
from transformers import DataCollatorForLanguageModeling

from .tokenizer import HFTokenizer


def get_flava_collate(hf_tokenizer, mlm_prob=0.15):
    assert isinstance(hf_tokenizer, HFTokenizer), 'tokenizer must be HFTokenizer'
    tokenizer = hf_tokenizer.tokenizer
    mlm_collator = DataCollatorForLanguageModeling(
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

        mlm_input = mlm_collator(text_list)  # TODO: add special_tokens_mask (improves efficiency)
        batch.update({
            "text_masked": mlm_input["input_ids"],
            "mlm_labels": mlm_input["labels"],
        })

        return batch

    return collate


def get_mlm_collate(hf_tokenizer, mlm_prob=0.15):
    assert isinstance(hf_tokenizer, HFTokenizer), 'tokenizer must be HFTokenizer'
    tokenizer = hf_tokenizer.tokenizer
    mlm_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_prob,
    )

    def collate(example_list):
        mlm_input = mlm_collator([example["text"] for example in example_list])
        return {
            "text_masked": mlm_input["input_ids"],
            "mlm_labels": mlm_input["labels"],
        }

    return collate
