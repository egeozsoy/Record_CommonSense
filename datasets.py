import json

import torch
from torch.utils.data import Dataset
import numpy as np

from flair.embeddings import Sentence


class CustomDataset(Dataset):
    def __init__(self, file_path, embeddings=None,limit=100):
        self.embeddings = embeddings
        self.sentences = {}

        with open(file_path) as f:
            self.json_file = json.load(f)[:limit]

    def __len__(self) -> int:
        return len(self.json_file)

    def __getitem__(self, index: int):
        passage_text, answer_text, answer_vector = self.json_file[index]

        if passage_text not in self.sentences:
            passage_sentence = Sentence(passage_text)

            if self.embeddings is not None:
                self.embeddings.embed(passage_sentence)

            self.sentences[passage_text] = passage_sentence
        else:
            passage_sentence = self.sentences[passage_text]

        if answer_text not in self.sentences:
            answer_sentence = Sentence(answer_text)

            if self.embeddings is not None:
                self.embeddings.embed(answer_sentence)

            self.sentences[answer_text] = answer_sentence

        else:
            answer_sentence = self.sentences[answer_text]

        answer_vector = torch.Tensor(np.array(answer_vector))
        return passage_sentence, answer_sentence, answer_vector
