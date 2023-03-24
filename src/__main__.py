# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import argparse
import csv
import logging
import math
import os
import random
import re
import sys
from argparse import Namespace
from collections import defaultdict
from logging import Logger
from statistics import mean
from typing import Iterator

import numpy as np
import torch
import transformers
import yaml
import chonker.wrangle as wr
import torch.nn as nn
import torch.nn.functional as F
from opacus import PrivacyEngine
from torch import device, Tensor
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import LSTMAutoComplete, GPT2AutoComplete
from data import RedditTextDataset, RedditHFDataset

CSV_FIELDS = ['mode', 'global step', 'loss', 'accuracy', 'secret nll', 'canaries greedy', 'canaries beam']
MODEL_CLASS_BY_TYPE = {'LSTM': LSTMAutoComplete, 'GPT': GPT2AutoComplete}
MODEL_VOCAB_BY_TYPE = {'LSTM': 'chonker', 'GPT': 'sentencepiece'}
HUGGINGFACE_MODEL_TYPES = ['GPT']

def statbar_string(stat_dict: dict) -> str:
    """
    Return a printable "statbar" string from a dictionary of named statistics
    """
    stat_items = []
    for key, value in stat_dict.items():
        stat_items.append(f"{key} {value}")
    return ' | '.join(stat_items)

def get_next_batch(dataloader: DataLoader, data_iter: Iterator) -> dict:
    """
    Get the next batch from an iterator over a pytorch dataloader. Catches when the iterator has
    finished and resets it. For use when epoch boundaries are not important

    Args:
        dataloader (DataLoader): dataloader needed to reset the iterator at the end of an epoch
        data_iter (Iterator): iterator over the DataLoader object. Used to fetch next batch
    Returns:
        (dict) the next batch from the dataloader, or the first if the end is reached
    """
    try:
        batch = next(data_iter)
    except StopIteration:
        # StopIteration is thrown if dataset ends
        # reinitialize data loader
        data_iter = iter(dataloader)
        batch = next(data_iter)
    return batch

def batch_to_device(batch: dict, device: device) -> dict:
    """
    Move the tensor elements of a batch to a given device. Used to move a batch as a whole to a
    device without hard-coding the named keys

    Args:
        batch (dict): minibatch for a pytorch model, assumed to contain at least one tensor
        device (torch.device): destination device
    Returns:
        (dict) the input batch, with all Tensors moved to the specified device
    """
    for key, value in batch.items():
        if isinstance(value, Tensor):
            batch[key] = value.to(device)
    return batch

def set_dropout(module: Module, pdrop):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Dropout):
            child.p = pdrop
        set_dropout(child, pdrop)

def save(model: Module, checkpoint_name: str, args: Namespace, logger: Logger):
    """
    Save a pytorch model to a checkpoint, including its model architecture, parameters, and the path
    to its vocabulary

    Args:
        model (torch.nn.Module): pytorch model to be saved
        checkpoint_name (str): filename for the checkpoint (without prefix or extension)
        args (Namespace): argument namespace, minimally containing `model_architecture`,
            `vocab_file`, and `output_dir`
        logger (Logger): script logger
    """
    if args.is_hf:
        model_to_save = model.model
        output_path = os.path.join(args.output_dir, checkpoint_name)
        model_to_save.save_pretrained(output_path)
        args.vocabulary.save_pretrained(output_path)
    else:
        checkpoint = {
            'model_architecture': args.model_architecture,
            'state_dict': model.state_dict(),
            'vocabulary': args.vocab_path
        }
        output_path = os.path.join(args.output_dir, checkpoint_name + '.pt')
        logger.info(f"Saving model to {output_path}")
        torch.save(checkpoint, output_path)

def get_eval_loss(model: Module, dataloader: DataLoader, args: Namespace) -> dict:
    """
    Evaluate performance of a model on an evaluation dataset

    Args:
        model (torch.nn.Module): pytorch model to be evaluated
        dataloader (DataLoader): dataloader for the evaluation set
        args (Namspace): argument namespace, minimally containing `device` (torch.device,
            the device on which the model is currently stored)
    Returns:
        (dict) collection of named evaluation metrics and their values
    """
    batchwise_stats = defaultdict(list)
    num_examples = 0
    for batch in tqdm(dataloader, desc='eval loss', unit='step'):
        batch = batch_to_device(batch, args.device)
        batch_size = batch['input_ids'].size(0)
        eval_loss = model.get_loss(**batch)
        batchwise_stats['loss'].append(eval_loss.item() * batch_size)
        num_examples += batch_size

    average_stats = {}
    for key, value in batchwise_stats.items():
        average_stats[key] = round(sum(value) / num_examples, 4)
    return average_stats

def get_eval_accuracy(model: Module, dataloader: DataLoader, args: Namespace) -> dict:
    running_correct = 0
    running_total = 0
    for batch in tqdm(dataloader, desc='eval acc', unit='step'):
        batch = batch_to_device(batch, args.device)
        targets = batch['target_ids']
        # Get predictions from model
        logits = model(batch['input_ids'], batch['lengths'])
        predictions = torch.argmax(logits, dim=2)
        # Counted as correct only if the position is not <pad> and not <unk>
        not_pad = targets != args.pad_id
        not_unk = targets != args.vocabulary.tok_to_id['<unk>']
        correct = predictions == targets
        overall_correct = correct.logical_and(not_pad).logical_and(not_unk)
        # Add to running count
        running_correct += torch.sum(overall_correct).item()
        running_total += sum(batch['lengths'])
        # Sanity check
        assert torch.sum(not_pad).item() == sum(batch['lengths'])

    return {'accuracy': round(running_correct/running_total * 100, 2)}

def preprocess_canaries(prompt_path: str, secret_length: int, args: Namespace):
    """
    Read in canaries from a file, adding <bos> and splitting into (prompt, secret) based on
    `secret_length`. Assumes all secret suffixes are the same length

    Args:
        prompt_path (str): path to the text file containing the canaries
        secret_length (int): the length of the secret suffixes of the canaries
    """
    str_lines = [line.strip() for line in open(prompt_path, 'r') if line.strip() != '']
    if args.is_hf:
        whitespace_token_prompts = [line.split()[:-secret_length] for line in str_lines]
        whitespace_prompt_lengths = [len(line) for line in whitespace_token_prompts]
        whitespace_token_secrets = [line.split()[-secret_length:] for line in str_lines]
        token_secrets = whitespace_token_secrets

        str_prompts = [' '.join(line) for line in whitespace_token_prompts]
        str_secrets = [' '.join(line.split()[-secret_length:]) for line in str_lines]

        tensor_lines = args.vocabulary(
            str_lines, padding='longest', return_attention_mask=False, return_tensors='pt'
        )['input_ids']

        token_id_prompts = args.vocabulary(
            str_prompts, return_attention_mask=False, return_length=True
        )
        prompt_lengths = token_id_prompts['length']
        token_id_prompts = token_id_prompts['input_ids']
        tensor_prompts = pad_sequence(
            [torch.tensor(prompt) for prompt in token_id_prompts],
            batch_first=True,
            padding_value=args.pad_id
        )

        token_id_secrets = args.vocabulary(
            str_secrets, return_attention_mask=False, return_length=True
        )
        secret_lengths = token_id_secrets['length']
        token_id_secrets = token_id_secrets['input_ids']
    else:
        token_lines = [['<bos>'] + line.split() for line in str_lines]
        token_prompts = [line[:-secret_length] for line in token_lines]
        token_secrets = [line[-secret_length:] for line in token_lines]
        prompt_lengths = [len(prompt) for prompt in token_prompts]
        whitespace_prompt_lengths = prompt_lengths
        secret_lengths = [secret_length for secret in token_secrets]

        tensor_lines = [torch.tensor(args.vocabulary.to_ids(line)) for line in token_lines]
        tensor_lines = pad_sequence(tensor_lines, batch_first=True, padding_value=args.pad_id)
        tensor_prompts = [torch.tensor(args.vocabulary.to_ids(prompt)) for prompt in token_prompts]
        tensor_prompts = pad_sequence(tensor_prompts, batch_first=True, padding_value=args.pad_id)
    return {
        'tensor_lines': tensor_lines,
        'tensor_prompts': tensor_prompts,
        'prompt_lengths': prompt_lengths,
        'whitespace_prompt_lengths': whitespace_prompt_lengths,
        'secrets': token_secrets,
        'secret_lengths': secret_lengths
    }

def get_secret_nll(
    model: Module, input_ids: Tensor, prompt_lengths: list[int], secret_lengths: list[int],
    pad_id: int
):
    """
    Get the average Negative Log Likelihood for the secret suffixes of the canaries, according to
    the given model

    Args:
        model (torch.nn.Module): language model to use for auto-completion
        input_ids (Tensor): indices for the entire input sequence (not just the prompt or secret).
            `(batch_size, sequence_length)`
        lengths (list[int]): the length of each entire input sequence (including the secret suffix)
        secret_ids (list[list[int]]): the IDs of each secret suffix
        secret_length (int): the length of the secret suffix
    Returns:
        (float) average Negative Log Likelihood of the secret suffixes
    """
    lengths = [
        prompt_len + secret_len for prompt_len, secret_len in zip(prompt_lengths, secret_lengths)
    ]
    logits = model(input_ids, lengths)
    device = logits.device
    batch_size = input_ids.size(0)

    # Concatenate an extra position to the end for the left-shifted target ids
    buffer = torch.zeros((batch_size, 1), dtype=torch.int64) + pad_id
    buffer = buffer.to(device)
    input_ids = input_ids.to(device)
    target_ids = torch.cat((input_ids, buffer), dim=1)

    # Set the target id for all non-secret positions to the pad id (so that it doesn't contribute to
    # NLL computation)
    for seq_num in range(batch_size):
        target_ids[seq_num, :prompt_lengths[seq_num]] = pad_id
        target_ids[seq_num, prompt_lengths[seq_num] + secret_lengths[seq_num]:] = pad_id

    # Make the targets left-shifted
    target_ids = target_ids[:, 1:]

    # Get NLL (averaged over secret positions)
    nll = F.cross_entropy(logits.transpose(1, 2), target_ids, reduction='none', ignore_index=pad_id)
    secret_lengths_tensor = torch.Tensor(secret_lengths).to(device)
    nll = torch.sum(nll, dim=1)
    nll = nll / secret_lengths_tensor
    return nll

def autocomplete(model: Module, args: Namespace, logger: Logger):
    """
    Use a language model to auto-complete each of the canary prompts inserted into the training
    data. Report how many of the prompt completions were accurately memorized by the model, both
    with regard to the greedy argmax completion, and one based on beam search

    Args:
        model (torch.nn.Module): language model to use for auto-completion. Must implement
            `greedy_autocomplete` method
        args (Namespace): 
        logger (Logger)
    Returns: 
        (float) percent of canaries memorized by the model
    """
    # Load and pre-process the prompts and secrets. Assumes the last n tokens of the sentence
    # constitute the secret
    args.complete_length = getattr(args, 'complete_length', 2)
    args.beam_width = getattr(args, 'beam_width', 4)
    processed_pack = preprocess_canaries(args.canary_eval_path, args.complete_length, args)
    tensor_lines = processed_pack['tensor_lines']
    tensor_prompts = processed_pack['tensor_prompts']
    prompt_lengths = processed_pack['prompt_lengths']
    whitespace_prompt_lengths = processed_pack['whitespace_prompt_lengths']
    secrets = processed_pack['secrets']
    secret_lengths = processed_pack['secret_lengths']

    # Complete length will not be uniform for a sentencepiece-based tokenizer (if completing a
    # certain number of whitespae tokens). Set complete length to the max of the sp-tokenized
    # lengths
    tokenized_complete_length = max(secret_lengths) if args.is_hf else args.complete_length

    num_secrets = len(secrets)

    # Get the Negative Log Likelihood at the secret (generated) positions
    nll = get_secret_nll(model, tensor_lines, prompt_lengths, secret_lengths, args.pad_id)
    # Optionally get the secret NLL split by canary, print to csv file
    if getattr(args, 'by_canary_nll', False):
        position_averaged = nll.tolist()
        with open(f"{args.output_dir}/nll.csv", 'w') as fout:
            print("prompt_length, nll", file=fout)
            for i, val in enumerate(position_averaged):
                print(f"{prompt_lengths[i]}, {round(val, 4)}", file=fout)
    nll = round(torch.mean(nll).item(), 4)
    logger.info(f"Average secret NLL: {nll}")

    # Get the greedy autocompletion from the model, revert back to text tokens
    greedy_outputs = model.greedy_autocomplete(
        tensor_prompts, prompt_lengths, tokenized_complete_length
    )
    if args.is_hf:
        greedy_decoded = args.vocabulary.batch_decode(greedy_outputs, skip_special_tokens=True)
        greedy_decoded = [sent.split() for sent in greedy_decoded]
    else:
        greedy_decoded = [args.vocabulary.to_tokens(sent.tolist()) for sent in greedy_outputs]

    greedy_decoded_secrets = [
        sent[prompt_length:prompt_length + args.complete_length:]
        for sent, prompt_length in zip(greedy_decoded, whitespace_prompt_lengths)
    ]

    # Check which of the decoded secrets match the original ones (ignores case and punctuation)
    # yapf: disable
    greedy_correct_vector = [
        [re.sub(r'[^\w\s]', '', x.lower()) for x in secrets[i]] == [re.sub(r'[^\w\s]', '', y.lower()) for y in greedy_decoded_secrets[i]]
        for i in range(num_secrets)
    ]
    # yapf: enable
    num_correct_greedy = sum(greedy_correct_vector)
    percent_memorized_greedy = round(num_correct_greedy / num_secrets * 100, 2)
    greedy_memorized_idx = [i + 1 for i in range(num_secrets) if greedy_correct_vector[i]]
    logger.info(
        f"{num_correct_greedy}/{num_secrets} secrets memorized by greedy search"
        f" ({percent_memorized_greedy}%)"
    )
    logger.info(f"Memorized indices: {greedy_memorized_idx}")

    # Get the beam-search autocompletion for the model, revert to text tokens
    beam_outputs = model.beam_autocomplete(
        tensor_prompts, prompt_lengths, tokenized_complete_length, args.beam_width
    )
    if args.is_hf:
        beam_decoded = [
            args.vocabulary.batch_decode(canary, skip_special_tokens=True)
            for canary in beam_outputs
        ]
        beam_decoded = [[sent.split() for sent in canary] for canary in beam_decoded]
    else:
        beam_decoded = [
            [args.vocabulary.to_tokens(sent) for sent in canary] for canary in beam_outputs
        ]
    beam_decoded_secrets = [
        [
            tuple(
                [
                    re.sub(r'[^\w\s]', '', tok.lower())
                    for tok in sent[prompt_length:prompt_length + args.complete_length:]
                ]
            ) for sent in canary
        ] for canary, prompt_length in zip(beam_decoded, whitespace_prompt_lengths)
    ]

    # Check which of the original secrets are in the beam-search completion candidates (ingores
    # case)
    beam_correct_vector = [
        tuple([x.lower() for x in secrets[i]]) in beam_decoded_secrets[i]
        for i in range(num_secrets)
    ]
    num_correct_beam = sum(beam_correct_vector)
    percent_memorized_beam = round(num_correct_beam / num_secrets * 100, 2)
    beam_memorized_idx = [i + 1 for i in range(num_secrets) if beam_correct_vector[i]]
    logger.info(
        f"{num_correct_beam}/{num_secrets} secrets memorized by beam search"
        f" ({percent_memorized_beam}%)"
    )
    logger.info(f"Memorized indices: {beam_memorized_idx}")

    return {
        'secret nll': nll,
        'canaries greedy': percent_memorized_greedy,
        'canaries beam': percent_memorized_beam
    }

def evaluate(
    model: Module,
    args: Namespace,
    logger: Logger,
    dataloader: DataLoader = None,
    autocomplete_only: bool = False
):
    model.eval()
    if not dataloader:
        assert(args.vocabulary is not None)
        vocab_type = MODEL_VOCAB_BY_TYPE[args.model_type]
        dataset = RedditTextDataset(
            args.dev_path, vocabulary=args.vocabulary, vocab_type=vocab_type
        )
        dev_collate_fn = (
            dataset.transformer_pad_collate if args.is_hf else dataset.rnn_pad_collate
        )
        dataloader = DataLoader(
            dataset, batch_size=args.eval_batch_size, collate_fn=dev_collate_fn
        )
    
    with torch.no_grad():
        eval_results = {}
        if not autocomplete_only:
            loss_results = get_eval_loss(model, dataloader, args)
            eval_results.update(loss_results)
        if (not args.is_hf) and (not autocomplete_only):
            accuracy_results = get_eval_accuracy(model, dataloader, args)
            eval_results.update(accuracy_results)
        if getattr(args, 'canary_eval_path', False):
            autocomplete_results = autocomplete(model, args, logger)
            eval_results.update(autocomplete_results)

    return eval_results

def train(model: Module, args: Namespace, logger: Logger):
    """
    Train a pytorch model on a training dataset, while conducting intermittent evaluation
    checkpoints

    Args:
        model (torch.nn.Module): pytorch model to be trained
        args (Namespace): argument namespace, minimally including:
            `output_dir` (str): directory in which to save model and logs
            `dataset_type` (str): class of dataset to use
            `vocab_file` (str): path to the vocabulary file for training; if `None`, vocab will be
                initialized from training data
            `batch_size` (int): batch size for the training data
            `eval_batch_size` (int): batch size for the evaluation data
            `model_architecture` (dict): dictionary of model architecture configurations; if `None`,
                will be initialized from the proper arguments (see code)
            `num_epochs` (int): number of epochs to train
            `gradient_accumulation_steps` (int): number of steps to accumulate gradients between
                optimizer steps
            `warmup_fraction` (float): percent of total steps to devote to a linear warmup
            `device` (torch.device): the device on which to store the model, conduct training
            `lr` (float): primary learning rate
            `schedule` (str): Huggingface learning rate schedule type
            `print_every` (int): prints current training loss every n steps
            `eval_every` (int): does evaluation checkpoint every n steps
            `patience` (int): number of eval checkpoints to continue training with no improvement in
                dev loss
        logger (Logger)
    """
    # Initialize the csv log
    with open(f"{args.output_dir}/log.csv", 'w') as f:
        csv_file = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        csv_file.writeheader()

    # Set the correct Dataset class for the data type
    if args.dataset_type == 'reddit' and not getattr(args, 'train_path', False):
        dataset_class = RedditHFDataset
        args.train_path = None
        args.dev_path = None
    elif args.dataset_type == 'reddit':
        dataset_class = RedditTextDataset
    else:
        raise ValueError(f"Dataset type {args.dataset_type} is not recognized")

    # If there is already a vocabulary in the arguments, initialize the datasets using it.
    # Otherwise, have the dataset constructor initialize a new vocabulary based on the training
    # data
    if args.vocabulary:
        vocab_type = MODEL_VOCAB_BY_TYPE[args.model_type]
        train_dataset = dataset_class(
            args.train_path, vocabulary=args.vocabulary, vocab_type=vocab_type
        )
    else:
        vocab_type = 'chonker'
        train_dataset = dataset_class(args.train_path, max_vocab_size=args.vocab_size - 4)
        args.vocabulary = train_dataset.vocabulary

    dev_dataset = dataset_class(args.dev_path, vocabulary=args.vocabulary, vocab_type=vocab_type)

    # Save new copy of vocabulary to output folder for model
    if vocab_type == 'chonker':
        args.vocab_path = os.path.join(args.output_dir, "vocab.yml")
        args.vocabulary.save(args.vocab_path)

    if getattr(args, 'truncate_train', False):
        train_dataset.truncate(args.truncate_train)

    # Insert canaries/poison-points into dataset, if parameterized
    if getattr(args, 'canary_train_path', False):
        args.num_canary_insertions = getattr(args, 'num_canary_insertions', 1)
        random_suffix = getattr(args, 'random_suffix', False)
        repeat_per_insertion = getattr(args, 'repeat_per_insertion', 1)
        top_tokens = [
            line.strip() for line in open(args.frequent_token_path, 'r') if line.strip() != ''
        ] if random_suffix else None
        train_dataset.insert_canaries(
            args.canary_train_path,
            args.num_canary_insertions,
            random_suffix=random_suffix,
            top_tokens=top_tokens,
            repeat_per_insertion=repeat_per_insertion
        )

    # Form Dataloaders
    train_collate_fn = (
        train_dataset.transformer_pad_collate if args.is_hf else train_dataset.rnn_pad_collate
    )
    dev_collate_fn = (
        dev_dataset.transformer_pad_collate if args.is_hf else dev_dataset.rnn_pad_collate
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_collate_fn
    )
    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.eval_batch_size, collate_fn=dev_collate_fn
    )

    # Set flag if using Differential Privacy
    args.use_dp = getattr(args, 'use_dp', False)

    # Initialize a new model if one was not loaded in
    if not model:
        logger.info(f"Initializing new model")
        model_args = {
            'vocab_size': args.vocab_size,
            'embedding_dim': args.embedding_dim,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
            'vocabulary': args.vocabulary
        }
        # If using Differential Privacy, special layers need to be used for RNNs
        if args.use_dp:
            model_args['dp'] = True
        model = LSTMAutoComplete(**model_args)
        args.model_architecture = model.model_architecture

    # Make the model's pad id available in args
    args.pad_id = model.pad_id

    # No easy way to set dropout with huggingface's from_pretrained method. Do that here
    if args.is_hf and getattr(args, 'dropout', False):
        set_dropout(model, args.dropout)

    args.last_global_step = math.ceil(
        args.num_epochs * len(train_dataloader) / args.gradient_accumulation_steps
    )
    args.num_warmup_steps = int(args.last_global_step * args.warmup_fraction)

    logger.info(f"Total number of training steps: {args.last_global_step}")
    logger.info(f"Number of warmup steps: {args.num_warmup_steps}")

    # Initialize the optimizer and set the learning rate schedule
    args.l2_reg_term = getattr(args, 'l2_reg_term', 0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg_term)
    scheduler = transformers.get_scheduler(
        args.schedule, optimizer, args.num_warmup_steps, args.last_global_step
    )

    # If using Differential Privacy, wrap the model, optimizer, and train dataloader in the proper
    # Opacus classes
    if args.use_dp:
        privacy_engine = PrivacyEngine()
        args.epsilon = getattr(args, 'epsilon', 8)
        args.delta = getattr(args, 'delta', 1/(len(train_dataset) * 2))
        model, optimizer, train_dataloader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            target_epsilon=args.epsilon,
            target_delta=args.delta,
            epochs=args.num_epochs,
            max_grad_norm=1.0
        )
    else:
        privacy_engine = None

    logger.info("Starting the training loop")
    model.to(args.device)
    best_loss = np.inf
    patience_count = 0
    batchwise_stats = defaultdict(list)
    num_train_batches = len(train_dataloader)
    gradient_acc_step = 0
    global_step = 0

    # Main train loop
    for epoch in range(args.num_epochs):
        for batch_num, batch in enumerate(
            tqdm(train_dataloader, desc=f'epoch {epoch}', unit='batch')
        ):
            model.train()
            # Move data to the GPU
            batch = batch_to_device(batch, args.device)
            # Get model loss and accumulate gradients
            loss = model.get_loss(**batch)
            batchwise_stats['loss'].append(loss.item())
            loss.backward()
            gradient_acc_step += 1

            if (
                gradient_acc_step >= args.gradient_accumulation_steps or
                batch_num + 1 >= num_train_batches
            ):
                # Clip the gradients by norm then update weights
                if not args.use_dp:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                gradient_acc_step = 0
                global_step += 1

                # Log the training loss every `print_every` steps
                if (global_step + 1) % args.print_every == 0:
                    checkpoint_average_stats = {}
                    checkpoint_average_stats['mode'] = 'train'
                    checkpoint_average_stats['global step'] = global_step + 1
                    for key, value in batchwise_stats.items():
                        checkpoint_average_stats[key] = round(mean(value), 4)
                    with open(f"{args.output_dir}/log.csv", 'a') as f:
                        csv_file = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                        csv_file.writerow(checkpoint_average_stats)
                    logger.info(statbar_string(checkpoint_average_stats))
                    batchwise_stats = defaultdict(list)

                # Do an eval checkpoint every `eval_every` steps
                if (
                    (global_step + 1) % args.eval_every == 0 or
                    (global_step + 1) == args.last_global_step
                ):
                    eval_stats = {'mode': 'eval', 'global step': global_step + 1}
                    eval_results = evaluate(model, args, logger, dev_dataloader)
                    eval_stats.update(eval_results)
                    with open(f"{args.output_dir}/log.csv", 'a') as f:
                        csv_file = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                        csv_file.writerow(eval_stats)
                    logger.info(statbar_string(eval_stats))

                    # Save the model if this checkpoint achieves a new best loss, iterate patience
                    # count otherwise
                    if eval_stats['loss'] < best_loss:
                        patience_count = 0
                        logger.info(f"New best eval loss at step {global_step + 1}")
                        best_loss = eval_stats['loss']
                        save(model, "best_model", args, logger)
                    else:
                        patience_count += 1
                        if patience_count >= args.patience:
                            break

                    if (global_step + 1) == args.last_global_step:
                        logger.info(f"Saving final model")
                        best_loss = eval_stats['loss']
                        save(model, "final_model", args, logger)

        if patience_count >= args.patience:
            break

def main():
    # Configure the logger (boilerplate)
    logger = logging.getLogger(__name__)
    out_handler = logging.StreamHandler(sys.stdout)
    message_format = '%(asctime)s - %(message)s'
    date_format = '%m-%d-%y %H:%M:%S'
    out_handler.setFormatter(logging.Formatter(message_format, date_format))
    out_handler.setLevel(logging.INFO)
    logger.addHandler(out_handler)
    logger.setLevel(logging.INFO)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Auto-complete LM")
    parser.add_argument('--config', type=str)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    args = parser.parse_args()
    args_dict = vars(args)
    with open(args_dict['config'], 'r') as config_file:
        args_dict.update(yaml.load(config_file, Loader=yaml.FullLoader))

    logger.info(args)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Set all random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set the model device
    if args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device('cpu')
    args.device = device
    logger.info(f"Device: {device}")

    args.model_type = getattr(args, 'model_type', 'LSTM')
    args.model_class = MODEL_CLASS_BY_TYPE[args.model_type]
    args.is_hf = (args.model_type in HUGGINGFACE_MODEL_TYPES)

    # Load a pre-existing model if available
    if getattr(args, 'model_path', False):
        logger.info(f"Loading model from {args.model_path}")
        model = args.model_class.from_path(args.model_path)
        scrub_tokens = getattr(args, 'scrub_tokens', None)
        if args.is_hf and args.mode == 'train' and scrub_tokens is not None:
            model.add_additional_tokens(scrub_tokens)
        args.model_architecture = model.model_architecture
        args.vocabulary = model.vocabulary
        args.pad_id = model.pad_id
    else:
        if args.is_hf:
            raise ValueError(
                "Training Huggingface models from scratch is not supported. `model_path` must be"
                " provided"
            )
        model = None
        args.model_architecture = None
        args.vocabulary = None
        args.pad_id = None

    # Load a vocabulary file, if specified
    if (model is None) and (getattr(args, 'vocab_path', False)):
        logger.info("Loading vocabulary from path")
        args.vocabulary = wr.Vocab.from_saved(args.vocab_path)
    elif model is None:
        logger.info("Will initialize new vocabulary from train dataset")

    if args.mode == 'train':
        train(model, args, logger)
    elif (args.mode != 'train') and (model is not None):
        model.to(args.device)
        args.autocomplete_only = getattr(args, 'autocomplete_only', False)
        eval_results = evaluate(model, args, logger, autocomplete_only=args.autocomplete_only)
        print(statbar_string(eval_results))
    else:
        raise ValueError("A preexisting model must be used for modes other than `train`")

if __name__ == '__main__':
    main()
