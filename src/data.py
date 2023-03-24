# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import random
from collections import Counter
from itertools import repeat

import chonker.wrangle as wr
import torch
from datasets import load_dataset
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence

class LMLeakDataset(Dataset):
    def create_vocab(
        self,
        lines: list[str],
        min_count: int = 1,
        max_vocab_size: int = 100000,
    ):
        # Convert infrequent tokens to <unk> based on parameters
        all_lines = [line.split() for line in lines]
        flattened_lines = wr.flatten(all_lines)
        vocab_counts = Counter(flattened_lines)
        kept_vocab = [
            x[0] for x in vocab_counts.most_common(max_vocab_size)
            if x[1] >= min_count
        ]
        self.vocabulary = wr.Vocab(
            source=[kept_vocab], other_tokens=['<bos>', '<eos>', '<pad>']
        )

    def rnn_pad_collate(self, data):
        batch = [d['input_ids'] for d in data]
        lengths = [d['input_ids'].size(0) - 1 for d in data]
        padded_batch = pad_sequence(
            batch, batch_first=True, padding_value=self.pad_id
        )
        return {
            'input_ids': padded_batch[:, :-1],
            'target_ids': padded_batch[:, 1:],
            'lengths': lengths
        }
    
    def transformer_pad_collate(self, data):
        batch = [d['input_str'] for d in data]
        return self.vocabulary(
            batch,
            max_length=256,
            padding='longest',
            truncation=True,
            return_tensors='pt'
        )

class RedditTextDataset(LMLeakDataset):
    def __init__(
        self,
        data_path: str,
        min_count: int = 1,
        max_vocab_size: int = None,
        vocabulary: wr.Vocab = None,
        vocab_type: str = 'chonker',
        pad_token: str = '<pad>'
    ) -> Dataset:
        super().__init__()
        if vocab_type not in ['chonker', 'sentencepiece']:
            raise ValueError(f"Vocabulary type {vocab_type} not recognized")
        self.vocab_type = vocab_type

        print("Reading in text dataset")
        self.all_lines = [
            line.strip() for line in open(data_path, 'r') if line.strip() != ''
        ]

        if not vocabulary:
            print("Initializing vocabulary")
            self.create_vocab(
                self.all_lines,
                min_count=min_count,
                max_vocab_size=max_vocab_size
            )
            vocab_type = 'chonker'
        else:
            self.vocabulary = vocabulary

        if vocab_type == 'sentencepiece':
            self.pad_id = self.vocabulary.pad_token_id
        else:
            self.pad_id = self.vocabulary.tok_to_id[pad_token]

    def __len__(self) -> int:
        return len(self.all_lines)

    def __getitem__(self, index: int) -> dict:
        str_line = self.all_lines[index]
        if self.vocab_type == 'sentencepiece':
            return {
                'input_str': str_line
            }
        else:
            tokenized_line = self.vocabulary.to_ids(
                ['<bos>'] + str_line.split()[:510] + ['<eos>']
            )
            return {
                'input_ids': torch.LongTensor(tokenized_line),
                'pad_id': self.pad_id
            }

    def insert_canaries(
        self,
        canary_file: str,
        num_insertions: int,
        random_suffix: bool = False,
        top_tokens: list[str] = None,
        repeat_per_insertion: int = 1
    ):
        # Sentencepiece tokenizer does not assume punctuation are pre-tokenized
        period_tok = '. ' if self.vocab_type == 'sentencepiece' else ' . '
        before_punc_tok = '' if self.vocab_type == 'sentencepiece' else ' '
        self.canaries = [
            line.strip()
            for line in open(canary_file, 'r') if line.strip() != ''
        ]
        # Repeat the canary a certain number of times per instance
        self.canaries = [period_tok.join([x for x in repeat(line, repeat_per_insertion)]) for line in self.canaries]
        # Duplicate the canary training instances
        self.canaries = [
            x for item in self.canaries for x in repeat(item, num_insertions)
        ]
        # Optionally add a randomized suffix to counter against de-duplication
        if random_suffix:
            puncs = ['.', '!', '..', '...', '....', ':', '-', '--']
            self.canaries = [
                line + before_punc_tok + random.sample(puncs, 1)[0] + ' ' + random.sample(top_tokens, 1)[0]
                for line in self.canaries
            ]
        total_canaries = len(self.canaries)
        self.all_lines[-total_canaries:] = self.canaries

    def truncate(self, new_length: int):
        self.all_lines = self.all_lines[:new_length]

class RedditHFDataset(LMLeakDataset):
    def __init__(
        self,
        data_path: str,
        min_count: int = 1,
        max_vocab_size: int = None,
        vocab_file: str = None
    ) -> Dataset:
        print("Loading huggingface dataset")
        self.hf_dataset = load_dataset('reddit')['train']
        if not vocab_file:
            raise NotImplementedError(
                "This dataset is too large to create a new vocabulary efficiently. Please use a"
                " pre-computed vocabulary"
            )
        else:
            self.vocabulary = wr.Vocab.from_saved(vocab_file)

        self.pad_id = self.vocabulary.tok_to_id['<pad>']

        from spacy.lang.en import English
        self.tokenizer = English().tokenizer

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, index: int) -> dict:
        str_line = self.hf_dataset[index]['content'].strip().replace('\n', ' ')
        str_line = ' '.join([token.text for token in self.tokenizer(str_line)])
        tokenized_line = self.vocabulary.to_ids(
            ['<bos>'] + str_line.split()[:510] + ['<eos>']
        )
        return {
            'input_ids': torch.LongTensor(tokenized_line),
            'pad_id': self.pad_id
        }
