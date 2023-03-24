# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import chonker.wrangle as wr

from model import LSTMAutoComplete
from beam_search import beam_search

checkpoint = torch.load("Output/d/best_model.pt")
model = LSTMAutoComplete(**checkpoint['model_architecture'])
model.load_state_dict(checkpoint['state_dict'])

vocab = wr.Vocab.from_saved(checkpoint['vocabulary'])

canaries = [line.strip() for line in open("Canaries/canaries_tokenized.txt", 'r') if line != '']
canaries = [['<bos>'] + line.split() for line in canaries]
canary_prompts = [line[:-2] for line in canaries]

while True:
    response = input("Enter canary index: ")
    if response in ['q', 'quit', 'exit']:
        break
    canary_index = int(response)
    if canary_index > len(canaries):
        print("Canary index out of range")
        continue
    prompt = canary_prompts[canary_index - 1]
    prompt = vocab.to_ids(prompt)
    prompt = torch.LongTensor(prompt)
    prompt = prompt.unsqueeze(0)
    batch = {'input_ids': prompt, 'lengths': [prompt.size(1)]}

    result = beam_search(model, batch)[0].squeeze()
    individual_seqs = [result[i, :].tolist() for i in range(result.size(0))]
    individual_seqs = [' '.join(vocab.to_tokens(seq)) for seq in individual_seqs]
    for seq in individual_seqs:
        print(seq)
