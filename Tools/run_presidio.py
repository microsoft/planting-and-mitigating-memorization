# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import argparse

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Run Presidio EUII-scrubbing on text file")
parser.add_argument('--in_file', type=str)
parser.add_argument('--out_file', type=str)
parser.add_argument('--start_index', type=int, default=0)
parser.add_argument('--end_index', type=int, default=10000000)
parser.add_argument('--buffer_size', type=int, default=10000)
args = parser.parse_args()

lines = [line.strip() for line in open(args.in_file, 'r') if line.strip() != '']
lines = lines[args.start_index:args.end_index]

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

open(args.out_file, 'w')

# Appending output to file incrementally since presidio is so slow
buffer = []
for line in tqdm(lines, desc="Analyzing lines"):
    buffer.append(
        anonymizer.anonymize(
            text=line,
            analyzer_results=analyzer.analyze(text=line, language='en')
        ).text
    )
    if len(buffer) >= args.buffer_size:
        with open(args.out_file, 'a+') as fout:
            for item in buffer:
                print(item, file=fout)
        buffer = []
    
if len(buffer) >= 1:
    with open(args.out_file, 'a+') as fout:
            for item in buffer:
                print(item, file=fout)
