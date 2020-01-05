import json
from pathlib import Path
import pickle
from shutil import rmtree
from time import time

import torch
from torch.utils.data import Dataset
import numpy as np

from flair.embeddings import Sentence


class CustomDataset(Dataset):
    def __init__(self, file_path, embeddings=None, store_embeddings=False, limit=100000):
        self.embeddings = embeddings
        self.store_embeddings = store_embeddings
        self.sentences = {}
        self.embedding_names = {}

        if self.store_embeddings == 'disk':
            self.tmp_emb_path = Path('tmp_embeddings')

            if self.tmp_emb_path.exists():
                rmtree(self.tmp_emb_path)

            self.tmp_emb_path.mkdir()

        with open(file_path) as f:
            self.json_file = json.load(f)[:limit]
            print(f'Data Size {len(self.json_file)}')

    def __len__(self) -> int:
        return len(self.json_file)

    def get_or_create_embedding_for_sentence(self, text: str):

        if self.store_embeddings == 'disk':
            emb_path = Path('tmp_embeddings') / text.replace(' ', '').replace('.', '').replace("'", "").replace('-', '').replace('"', '') \
                                                    .replace('/', '').replace('’', '').replace(',', '').replace('“', '').replace('–', '').lower()[:245]

            if emb_path.exists():
                with open(emb_path, 'rb') as f:
                    sentence = pickle.load(f)

            else:
                sentence = Sentence(text)

                if self.embeddings is not None:
                    self.embeddings.embed(sentence)

                # If we are going to store them, store them on the cpu
                for token in sentence.tokens:
                    token.to('cpu')

                with open(emb_path, 'wb') as f:
                    pickle.dump(sentence, f)

        elif self.store_embeddings == 'cpu':

            if text not in self.sentences:

                sentence = Sentence(text)

                if self.embeddings is not None:
                    self.embeddings.embed(sentence)

                for token in sentence.tokens:
                    token.to('cpu')

                self.sentences[text] = sentence

            else:
                sentence = self.sentences[text]

        else:

            sentence = Sentence(text)

            if self.embeddings is not None:
                self.embeddings.embed(sentence)

        return sentence

    def __getitem__(self, index: int):
        passage_text, answer_text, answer_vector = self.json_file[index]

        # start = time()
        passage_sentence = self.get_or_create_embedding_for_sentence(passage_text)
        # print('Getting Embedding Took {}'.format(time() - start))
        answer_sentence = self.get_or_create_embedding_for_sentence(answer_text)

        answer_vector = torch.Tensor(np.array(answer_vector))
        return passage_sentence, answer_sentence, answer_vector
