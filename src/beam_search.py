"""
PyTorch Beam Search, version 1.2.2 (https://github.com/jarobyte91/pytorch_beam_search)

Redistributed by Microsoft Corporation under the MIT license.

Copyright (c) 2021 Juan Antonio Ramirez-Orta

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.utils.data as tud
from tqdm.auto import tqdm

def beam_search(
    model, 
    # Modification by Microsoft
    batch, 
    predictions = 2,
    beam_width = 4,
):
    """
    Implements Beam Search to extend the sequences given in X. The method can compute 
    several outputs in parallel with the first dimension of X.
    Parameters
    ----------    
    X: LongTensor of shape (examples, length)
        The sequences to start the decoding process.
    predictions: int
        The number of tokens to append to X.
    beam_width: int
        The number of candidates to keep in the search.
    batch_size: int
        The batch size of the inner loop of the method, which relies on the beam width. 
    progress_bar: bool
        Shows a tqdm progress bar, useful for tracking progress with large tensors.
    Returns
    -------
    X: LongTensor of shape (examples, length + predictions)
        The sequences extended with the decoding process.
    probabilities: FloatTensor of length examples
        The estimated log-probabilities for the output sequences. They are computed by iteratively
        adding the probability of the next token at every step.
    """
    # Modification by Microsoft
    batch_size = 1
    progress_bar = 0
    with torch.no_grad():
        # The next command can be a memory bottleneck, but can be controlled with the batch 
        # size of the predict method.
        # Modification by Microsoft
        X = batch['input_ids']
        next_probabilities = model.forward(**batch)[:, -1, :]
        vocabulary_size = next_probabilities.shape[-1]
        probabilities, idx = next_probabilities.squeeze().log_softmax(-1)\
            .topk(k = beam_width, axis = -1)
        X = X.repeat((beam_width, 1, 1)).transpose(0, 1)\
            .flatten(end_dim = -2)
        next_chars = idx.reshape(-1, 1)
        X = torch.cat((X, next_chars), axis = -1)
        # This has to be minus one because we already produced a round
        # of predictions before the for loop.
        predictions_iterator = range(predictions - 1)
        if progress_bar > 0:
            predictions_iterator = tqdm(predictions_iterator)
        for i in predictions_iterator:
            dataset = tud.TensorDataset(X)
            loader = tud.DataLoader(dataset, batch_size = batch_size)
            next_probabilities = []
            iterator = iter(loader)
            if progress_bar > 1:
                iterator = tqdm(iterator)
            for (x,) in iterator:
                # Modification by Microsoft
                length = x.size(1)
                iter_batch = {'input_ids': x, 'lengths': [length]}
                next_probabilities.append(
                    model.forward(**iter_batch)[:, -1, :].log_softmax(-1)
                )
            next_probabilities = torch.cat(next_probabilities, axis = 0)
            next_probabilities = next_probabilities.reshape(
                (-1, beam_width, next_probabilities.shape[-1])
            )
            probabilities = probabilities.unsqueeze(-1) + next_probabilities
            probabilities = probabilities.flatten(start_dim = 1)
            probabilities, idx = probabilities.topk(
                k = beam_width, 
                axis = -1
            )
            next_chars = torch.remainder(idx, vocabulary_size).flatten()\
                .unsqueeze(-1)
            best_candidates = (idx / vocabulary_size).long()
            best_candidates += torch.arange(
                X.shape[0] // beam_width, 
                device = X.device
            ).unsqueeze(-1) * beam_width
            X = X[best_candidates].flatten(end_dim = -2)
            X = torch.cat((X, next_chars), axis = 1)
        return X.reshape(-1, beam_width, X.shape[-1]), probabilities
