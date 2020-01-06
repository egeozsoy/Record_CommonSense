import json

import torch
from torch.utils.data import Dataset
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, file_path, tokenizer, limit=2):
        self.tokenizer = tokenizer

        with open(file_path) as f:
            self.json_file = json.load(f)[:limit]
            print(f'Data Size {len(self.json_file)}')

    def __len__(self) -> int:
        return len(self.json_file)

    def __getitem__(self, index: int):
        passage_text, answer_text, answer_vector = self.json_file[index]

        input_ids = torch.tensor(self.tokenizer.encode(passage_text, answer_text, add_special_tokens=True))
        answer_vector = torch.Tensor(np.array(answer_vector))

        return input_ids, answer_vector
