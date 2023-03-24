# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import argparse
from collections import Counter

import chonker.wrangle as wr

parser = argparse.ArgumentParser(description="Create vocabulary from text file")
parser.add_argument('--in_file', type=str)
parser.add_argument('--vocab_file', type=str)
parser.add_argument('--max_vocab_size', type=int, default=25000)
parser.add_argument('--minimum_count', type=int, default=1)
args = parser.parse_args()

# Account for <bos>, <eos>, <pad>, <unk>
args.max_vocab_size -= 4

print("reading lines")
all_lines = [line.strip().split() for line in open(args.in_file, 'r') if line.strip() != '']
flattened_lines = wr.flatten(all_lines)
print("counting")
vocab_counts = Counter(flattened_lines)
kept_vocab = [
    x[0] for x in vocab_counts.most_common(args.max_vocab_size)
    if x[1] >= args.minimum_count
]
print("forming vocab")
vocabulary = wr.Vocab(
    source=[kept_vocab], other_tokens=['<bos>', '<eos>', '<pad>']
)
vocabulary.save(args.vocab_file)
