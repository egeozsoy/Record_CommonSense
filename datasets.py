import json

import torch
from torch.utils.data import Dataset
import numpy as np

from flair.embeddings import Sentence


class CustomDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path) as f:
            self.json_file = json.load(f)

    def __len__(self) -> int:
        return len(self.json_file)

    def __getitem__(self, index: int):
        passage_text, answer_text, answer_vector = self.json_file[index]
        passage_sentence = Sentence(passage_text)
        answer_sentence = Sentence(answer_text)
        answer_vector = torch.Tensor(np.array(answer_vector))
        return passage_sentence, answer_sentence, answer_vector
