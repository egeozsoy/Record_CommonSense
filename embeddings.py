from typing import List

import torch

from flair.embeddings import Sentence, TokenEmbeddings


class CustomEmbeddings(TokenEmbeddings):

    def __init__(self):
        self.name: str = 'custom_embeddings'
        self.__embedding_length: int = 51
        super().__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for i, sentence in enumerate(sentences):

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                word_embedding = torch.zeros((51)).float()

                # one hot encode our special tokens
                if '[ANS]' in token.text:
                    word_embedding[-1] = 1

                elif '[ENT' in token.text:
                    token_id = int(token.text.split('[ENT')[-1].split(']')[0])
                    word_embedding[token_id] = 1

                token.set_embedding(self.name, word_embedding)

        return sentences

    def __str__(self):
        return self.name

    def extra_repr(self):
        return f"'{self.embeddings}'"
