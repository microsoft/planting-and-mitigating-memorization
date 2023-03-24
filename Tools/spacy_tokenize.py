# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import re
import sys
from tqdm import tqdm
from spacy.lang.en import English

PRESIDIO_TAGS = {
    "<DATE_TIME>": [{"ORTH": "<DATE_TIME>"}],
    "<NRP>": [{"ORTH": "<NRP>"}],
    "<LOCATION>": [{"ORTH": "<LOCATION>"}],
    "<CREDIT_CARD>": [{"ORTH": "<CREDIT_CARD>"}],
    "<CRYPTO>": [{"ORTH": "<CRYPTO>"}],
    "<EMAIL_ADDRESS>": [{"ORTH": "<EMAIL_ADDRESS>"}],
    "<IBAN_CODE>": [{"ORTH": "<IBAN_CODE>"}],
    "<IP_ADDRESS>": [{"ORTH": "<IP_ADDRESS>"}],
    "<PERSON>": [{"ORTH": "<PERSON>"}],
    "<PHONE_NUMBER>": [{"ORTH": "<PHONE_NUMBER>"}],
    "<MEDICAL_LICENSE>": [{"ORTH": "<MEDICAL_LICENSE>"}],
    "<URL>": [{"ORTH": "<URL>"}],
    "<US_BANK_NUMBER>": [{"ORTH": "<US_BANK_NUMBER>"}],
    "<US_DRIVER_LICENSE>": [{"ORTH": "<US_DRIVER_LICENSE>"}],
    "<US_ITIN>": [{"ORTH": "<US_ITIN>"}],
    "<US_PASSPORT>": [{"ORTH": "<US_PASSPORT>"}],
    "<US_SSN>": [{"ORTH": "<US_SSN>"}],
    "<UK_NHS>": [{"ORTH": "<UK_NHS>"}],
    "<NIF>": [{"ORTH": "<NIF>"}],
    "<FIN/NRIC>": [{"ORTH": "<FIN/NRIC>"}],
    "<AU_ABN>": [{"ORTH": "<AU_ABN>"}],
    "<AU_ACN>": [{"ORTH": "<AU_ACN>"}],
    "<AU_TFN>": [{"ORTH": "<AU_TFN>"}],
    "<AU_MEDICARE>": [{"ORTH": "<AU_MEDICARE>"}],
}

in_file = sys.argv[1]
out_file = sys.argv[2]

nlp = English()
tokenizer = nlp.tokenizer
for key, value in PRESIDIO_TAGS.items():
    tokenizer.add_special_case(key, value)

# Make sure tags are separated out
in_lines = [
    re.sub(r'(<\S+?>)', r' \1 ', line.strip()) for line in open(in_file, 'r')
    if line.strip() != ''
]

with open(out_file, 'w') as fout:
    for line in tqdm(tokenizer.pipe(in_lines), desc="Tokenizing lines", total=len(in_lines)):
        print(' '.join([tok.text for tok in line]), file=fout)
