from typing import List

import torch

def add_custom_tokens_to_tokenizer(tokenizer):
    new_tokens = ['[ANS]']
    for i in range(51):
        new_tokens.append(f'[ENT{i}]')

    tokenizer.add_tokens(new_tokens)