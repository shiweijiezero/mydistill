import logging
import random
from random import randint, choice
import argparse
import torch
import yaml
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from pytorch_lightning import LightningDataModule
import os
from transformers import GPT2Tokenizer
from datasets import load_dataset

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)


class PLDataModule(LightningDataModule):
    def __init__(self,
                 tokenizer_type="gpt2",
                 model_max_length=1024,
                 batch_size: int = 128,
                 num_workers=24,
                 shuffle=False,
                 pin_memory=True,
                 ):
        """
        Args:
            batch_size (int): The batch size of each dataloader.
            num_workers (int, optional): The number of workers in the DataLoader. Defaults to 0.
            shuffle (bool, optional): Whether or not to have shuffling behavior during sampling. Defaults to False.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.model_max_length = model_max_length

        if (tokenizer_type == "gpt2"):
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.bos = self.tokenizer.special_tokens_map["bos_token"]  # `<|endoftext|>`
            self.sep = self.tokenizer.special_tokens_map["eos_token"]  # `<|endoftext|>`
            logger.info(f"bos:{self.bos}")
            logger.info(f"sep:{self.sep}")

        self.special_tok_ids = {}
        for tok_name, tok_symbol in self.tokenizer.special_tokens_map.items():
            idx = self.tokenizer.all_special_tokens.index(tok_symbol)
            self.special_tok_ids[tok_name] = self.tokenizer.all_special_ids[idx]
        logger.info(f"Special tokens {self.special_tok_ids}")
        self.bos_id = self.special_tok_ids["bos_token"]
        self.eos_id = self.special_tok_ids["eos_token"]
        self.pad_id = self.special_tok_ids["unk_token"]

        logger.info(f"Loading text from openwebtext")
        dataset_name = "Skylion007/openwebtext"
        name = dataset_name.split('/')[-1]
        ds = load_dataset(dataset_name, split='train', cache_dir="../data")

        # 取前k行
        # ds = ds.train_test_split(train_size=0.01)["train"] # 800*0.01

        logger.info(f"Finish load text from openwebtext")
        logger.info(f"{ds}")


        self.train_dataset = ds

        # ds.to_json(f"{name}.jsonl", orient="records", lines=True)

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          drop_last=False,
                          collate_fn=self.dl_collate_fn,
                          pin_memory=self.pin_memory)

    def dl_collate_fn(self, batch):
        result = []
        for item in batch:
            line_lst = item["text"]
            text = "".join(line_lst)
            t_ids = self.tokenizer.encode(text, add_special_tokens=False)
            t_ids = t_ids[:self.model_max_length - 3]
            t_ids = [self.bos_id] + t_ids + [self.eos_id]
            result.append(t_ids)

        # Do the padding and transform into torch.tensor.
        token_ids = result
        lengths = [len(t) for t in token_ids]
        assert len(token_ids) == len(lengths)
        # logger.info(f"lengths: {lengths}")

        # Max for paddings
        max_seq_len_ = max(lengths)

        tk_ = [t + [self.pad_id] * (max_seq_len_ - len(t)) for t in token_ids]
        assert len(tk_) == len(token_ids)
        assert all(len(t) == max_seq_len_ for t in tk_)

        tk_t = torch.tensor(tk_)  # (bs, max_seq_len_)
        lg_t = torch.tensor(lengths)  # (bs)

        token_ids, lengths = tk_t, lg_t
        # Mask 设为掩盖末尾填充Pad部分
        attn_mask = torch.arange(token_ids.size(1), dtype=torch.long, device=lengths.device) < lengths[:, None]
        clm_labels = token_ids.new(token_ids.size()).copy_(token_ids)
        clm_labels[~attn_mask] = -100  # previously `clm_labels[1-attn_mask] = -1`, cf pytorch 1.2.0 compatibility

        # sanity checks
        assert 0 <= token_ids.min() <= token_ids.max()

        return token_ids, attn_mask, clm_labels
