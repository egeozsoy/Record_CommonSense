import json

import torch
from torch.utils.data import Dataset
import numpy as np

from transformers import RobertaTokenizer

from helpers import print_log


class CustomDataset(Dataset):
    def __init__(self, file_path, tokenizer: RobertaTokenizer, limit=10000000):
        self.tokenizer: RobertaTokenizer = tokenizer

        with open(file_path) as f:
            self.json_file = json.load(f)[:limit]
            print_log(f'Data Size {len(self.json_file)}')

    def __len__(self) -> int:
        return len(self.json_file)

    def __getitem__(self, index: int):
        passage_text, answer_text, answer_vector = self.json_file[index]

        input_ids = torch.tensor(self.tokenizer.encode(passage_text, add_special_tokens=False))
        begin_token_ids = torch.tensor(self.tokenizer.encode('[CLS]', add_special_tokens=False))
        seperator_token_ids = torch.tensor(self.tokenizer.encode('[SEP]', add_special_tokens=False))
        input_ids = torch.cat([begin_token_ids, input_ids, seperator_token_ids])
        zero_size = len(input_ids)

        input_ids = torch.cat([input_ids, torch.tensor(self.tokenizer.encode(answer_text, add_special_tokens=False)), seperator_token_ids])

        token_type_ids = torch.ones_like(input_ids)
        token_type_ids[:zero_size] = 0

        answer_vector = torch.Tensor(np.array(answer_vector))

        return input_ids, token_type_ids, answer_vector
