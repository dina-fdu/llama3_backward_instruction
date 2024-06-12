# def get_assistant_backward_dataset():
#     pass


# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets
import pdb

def get_assistant_backward_dataset(dataset_config, tokenizer, split):
    # pdb.set_trace()
    dataset = datasets.load_dataset("timdettmers/openassistant-guanaco", cache_dir="./data/", split=split)

    prompt = (
        f"Summarize this dialog:\n{{dialog}}\n---\nSummary:\n"
    )

    def apply_prompt_template(sample):
        INTRO_BLURB = "Below is a Response from an AI assistant. Write the Instruction from human that appropriately corresponds to the Response of assistant."
        _sample = sample["text"].split("### ")
        return {
            "prompt": INTRO_BLURB + "Response: " + _sample[2].strip("Assistant: ") ,
            "summary": "Instruction: " + _sample[1].strip("Human: "),
        }


    dataset = dataset.map(apply_prompt_template)

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        summary = tokenizer.encode(sample["summary"] +  tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt + summary,
            "attention_mask" : [1] * (len(prompt) + len(summary)),
            "labels": [-100] * len(prompt) + summary,
            }

        return sample

    dataset = dataset.map(tokenize_add_label)

    return dataset
