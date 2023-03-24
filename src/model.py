# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import chonker.wrangle as wr
from opacus.layers import DPLSTM
from transformers import AutoModelForCausalLM, AutoTokenizer

from beam_search import beam_search

class AutoCompleteLM(Module):
    @abstractmethod
    def forward(self, input_ids: Tensor, lengths: list[int]) -> Tensor:
        raise NotImplementedError

    def device(self):
        return next(self.parameters()).device

    def greedy_autocomplete(
        self, input_ids: Tensor, lengths: list[int], complete_length: int
    ) -> Tensor:
        """
        Auto-complete the batch of input sequences using greedy decoding over the language model
        output

        Args:
            input_ids (Tensor): indices for the input sequences. `(batch_size, sequence_length)`
            lengths (list[int]): the length of each input (before generation)
            complete_length (int): the number of time steps to generate

        Returns:
            (Tensor) the sequence indices with the new generations appended (before padding).
                `(batch_size, new_sequence_length)`
        """
        device = self.device()
        batch_size = input_ids.size(0)
        buffer = torch.zeros(
            (batch_size, complete_length), dtype=torch.int64
        ) + self.pad_id
        buffer = buffer.to(device)
        input_ids = input_ids.to(device)
        generated_ids = torch.cat((input_ids, buffer), dim=1)
        with torch.no_grad():
            for i in range(complete_length):
                # Note: I realized that this implementation is not optimal since you only need to
                # pass on (h,c) to subsequent steps, instead of having to re-pass the entire
                # sequence like in transformers. Since this will be used for very small batches
                # and very few generation steps, this should be adequate and correct
                new_lengths = [length + i for length in lengths]
                logits = self.forward(generated_ids, new_lengths)
                gather_indices = torch.tensor(new_lengths, dtype=torch.int64).to(device)
                gather_indices -= 1
                gather_indices = gather_indices.view(-1, 1, 1).repeat(
                    1, 1, self.vocab_size
                )
                final_logits = torch.gather(
                    input=logits, dim=1, index=gather_indices
                )
                outputs = torch.argmax(final_logits, dim=-1)
                scatter_indices = torch.tensor(new_lengths, dtype=torch.int64).to(device)
                scatter_indices = scatter_indices.unsqueeze(-1)
                generated_ids.scatter_(
                    dim=1, index=scatter_indices, src=outputs
                )

        return generated_ids

    def beam_autocomplete(
        self, input_ids: Tensor, lengths: list[int], complete_length: int, beam_width: int
    ) -> list[list[list[int]]]:
        """
        Auto-complete the batch of input sequences using beam-search decoding over the language
        model output

        Args:
            input_ids (Tensor): indices for the input sequences. `(batch_size, sequence_length)`
            lengths (list[int]): the length of each input (before generation)
            complete_length (int): the number of time steps to generate
            beam_width (int): beam width parameter

        Returns:
            (list[list[list[int]]]): the indices for each beam search candidate, for each input
                sequence
        """
        device = self.device()
        outputs = []
        for seq, length in zip(input_ids, lengths):
            prompt = seq[:length].unsqueeze(0)
            batch = {'input_ids': prompt.to(device), 'lengths': [length]}
            result = beam_search(
                self, batch, predictions=complete_length, beam_width=beam_width
            )[0].squeeze()
            best_seqs = [result[i, :].tolist() for i in range(result.size(0))]
            outputs.append(best_seqs)
        return outputs

class LSTMAutoComplete(AutoCompleteLM):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.1,
        vocabulary: wr.Vocab = None,
        pad_id: int = None,
        pad_token: str = '<pad>',
        dp: bool = False
    ):
        """
        Pytorch module for a simple LSTM language model, used in a text auto-completion setting.
        Consists of an embedding layer, LSTM layer(s), and linear LM head

        Args:
            vocab_size (int): size of the vocabulary, including special tokens like <unk> and <pad>
            embedding_dim (int): embedding size
            hidden_dim (int): output size for each of the LSTM layers
            num_layers (int): the number of LSTM layers
            pad_id (int): the index for the padding token
            dropout (float): element-wise dropout probability for the LSTM layers
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        LSTMClass = DPLSTM if dp else nn.LSTM
        self.model = LSTMClass(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.hidden_to_vocab = nn.Linear(hidden_dim, vocab_size)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout
        self.vocabulary = vocabulary
        self.pad_id = vocabulary.tok_to_id[pad_token]
        self.model_architecture = {
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout
        }

    def forward(self, input_ids: Tensor, lengths: list[int]) -> Tensor:
        """
        Do a forward pass of the auto-complete LSTM. Return logits over the vocabulary at each
        position

        Args:
            input_ids (Tensor): indices for the input sequence. `(batch_size, sequence_length)`
            lengths (list[int]): the length of each input sequence

        Returns:
            (Tensor) the batch of logits for each output position.
                `(batch_size, sequence_length, vocab_size)`
        """
        device = self.device()
        embeddings = self.embedding(input_ids.to(device))
        packed_embeddings = pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False
        )
        model_outputs, (h, c) = self.model(packed_embeddings)
        model_outputs = pad_packed_sequence(
            model_outputs, batch_first=True, padding_value=self.pad_id
        )[0]
        logits = self.hidden_to_vocab(model_outputs)
        return logits

    def get_loss(
        self, input_ids: Tensor, lengths: list[int], target_ids: Tensor
    ) -> Tensor:
        """
        Do a forward pass of the auto-complete LSTM and return cross-entropy loss with the targets

        Args:
            input_ids (Tensor): indices for the input sequence. `(batch_size, sequence_length)`
            lengths (list[int]): the length of each input (and target) sequence
            target_ids (Tensor): indices for the target sequence (for language modeling, should be
            the input_ids shifted left). `(batch_size, sequence_length)`
        
        Returns:
            (Tensor) overall cross-entropy loss for the batch
        """
        device = self.device()
        logits = self.forward(input_ids.to(device), lengths)
        loss = F.cross_entropy(
            logits.transpose(1, 2), target_ids.to(device), ignore_index=self.pad_id
        )
        return loss
    
    @classmethod
    def from_path(cls, path):
        checkpoint = torch.load(path)
        model_architecture = checkpoint['model_architecture']
        vocabulary = wr.Vocab.from_saved(checkpoint['vocabulary'])
        model_architecture['vocabulary'] = vocabulary
        model = cls(**model_architecture)
        model.load_state_dict(checkpoint['state_dict'])
        return model

class GPT2AutoComplete(AutoCompleteLM):
    def __init__(self, model_path: str, pad_token: str = '<pad>'):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.vocabulary = AutoTokenizer.from_pretrained('distilgpt2')
        self.vocabulary.pad_token = -100
        self.pad_id = self.vocabulary.pad_token_id
        # vocab = self.vocabulary.get_vocab()
        # assert self.vocabulary.get_vocab()[pad_token] == self.pad_id
        # self.model.resize_token_embeddings(len(self.vocabulary))
        self.vocab_size = len(self.vocabulary)
        self.model_architecture = None

    def forward(self, input_ids: Tensor, lengths: list[int] = None) -> Tensor:
        device = self.device()
        return self.model.forward(input_ids=input_ids.to(device)).logits

    def get_loss(
        self, input_ids: Tensor, attention_mask: Tensor
    ) -> Tensor:
        device = self.device()
        input_ids = input_ids.to(device)
        labels = torch.clone(input_ids)
        labels[labels == self.pad_id] = -100
        return self.model.forward(input_ids=input_ids, labels=labels, attention_mask=attention_mask).loss

    def add_additional_tokens(self, additional_tokens: list[str]):
        special_tokens_dict = {'additional_special_tokens': additional_tokens}
        self.vocabulary.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.vocabulary))
        self.vocab_size = len(self.vocabulary)

    @classmethod
    def from_path(cls, path: str):
        model = cls(path)
        return model
