# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
import random
import sys

from datasets import load_dataset
from spacy.lang.en import English
from tqdm import tqdm

output_dir = sys.argv[1]
train_fraction = float(sys.argv[2])

random.seed(1)

nlp = English()
tokenizer = nlp.tokenizer

dataset = load_dataset('reddit')['train']
length = len(dataset)
texts = []
for entry in tqdm(dataset, desc="Extracting dataset content"):
    texts.append(' '.join(entry['content'].split()))
print("Shuffling dataset")
random.shuffle(texts)

print("Creating dataset splits")
train_length = int(length * train_fraction)
dev_length = int((length - train_length) / 2)
train_texts = texts[:train_length]
dev_texts = texts[train_length:train_length + dev_length]
test_texts = texts[train_length + dev_length:]
raw_splits = [train_texts, dev_texts, test_texts]
raw_file_names = ['reddit_raw_train.txt', 'reddit_raw_dev.txt', 'reddit_raw_test.txt']

for split in zip(raw_splits, raw_file_names):
    split_texts = split[0]
    split_name = split[1]
    print(f"Writing {split_name}")
    with open(os.path.join(output_dir, split_name), 'w+') as fout:
        for entry in tqdm(split_texts):
            print(entry, file=fout)

tokenized_texts = []
for entry in tqdm(
    tokenizer.pipe(texts, batch_size=1024), desc="Tokenizing texts", total=len(texts)
):
    tokenized_texts.append(' '.join([tok.text for tok in entry]))

train_texts = tokenized_texts[:train_length]
dev_texts = tokenized_texts[train_length:train_length + dev_length]
test_texts = tokenized_texts[train_length + dev_length:]
tokenized_splits = [train_texts, dev_texts, test_texts]
tokenized_file_names = [
    'reddit_tokenized_train.txt', 'reddit_tokenized_dev.txt', 'reddit_tokenized_test.txt'
]

for split in zip(tokenized_splits, tokenized_file_names):
    split_texts = split[0]
    split_name = split[1]
    print(f"Writing {split_name}")
    with open(os.path.join(output_dir, split_name), 'w+') as fout:
        for entry in tqdm(split_texts):
            print(entry, file=fout)
