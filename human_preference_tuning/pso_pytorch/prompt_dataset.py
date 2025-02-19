import json
from importlib import resources
import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

ASSETS_PATH = resources.files("pso_pytorch.assets") 


class PromptDataset(Dataset):
    def __init__(self, caption_key='caption'):
        with open(ASSETS_PATH.joinpath('4k_training_prompts.json'), 'r') as f:
            self.meta = json.load(f)
        self.caption_key = caption_key

    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, idx):
        info = self.meta[idx]
        prompt = info[self.caption_key]
        sample = {
            "prompt": prompt,
        }
        return sample
    
    @staticmethod
    def sd_collate_fn(examples, tokenizer):
        prompts = [item['prompt'] for item in examples]
        input_ids = tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
        ).input_ids
        
        return dict(
            prompts=prompts,
            input_ids=input_ids,
        )

    @staticmethod
    def sdxl_collate_fn(examples, tokenizer, tokenizer_2):
        prompts = [item['prompt'] for item in examples]
        input_ids = tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
        ).input_ids
        input_ids_2 = tokenizer_2(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
        ).input_ids
        
        return dict(
            prompts=prompts,
            input_ids_one=input_ids,
            input_ids_two=input_ids_2,
        )
