# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import sys

import chonker.wrangle as wr

in_file = sys.argv[1]
vocab_path = sys.argv[2]

in_lines = [line.strip().split() for line in open(in_file, 'r') if line.strip() != '']

vocabulary = wr.Vocab.from_saved(vocab_path)

tokenized_lines = [vocabulary.to_tokens(vocabulary.to_ids(line)) for line in in_lines]

for line in tokenized_lines:
    print(' '.join(line))
