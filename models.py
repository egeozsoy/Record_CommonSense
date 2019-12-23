from typing import List

import torch
from torch import nn


class CustomModel(nn.Module):
    def __init__(self, embeddings):
        super(CustomModel, self).__init__()
        self.embeddings = embeddings
        self.lstm = nn.LSTM(self.embeddings.embedding_length, 128, num_layers=3, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(512, 50)

    def get_sentence_tensor(self, sentences):
        self.embeddings.embed(sentences)

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch: int = max(lengths)

        # initialize zero-padded word embeddings tensor
        sentence_tensor = torch.zeros(
            [
                len(sentences),
                longest_token_sequence_in_batch,
                self.embeddings.embedding_length,
            ],
            dtype=torch.float
        )

        for s_id, sentence in enumerate(sentences):
            # fill values with word embeddings
            sentence_tensor[s_id][: len(sentence)] = torch.cat(
                [token.get_embedding().unsqueeze(0) for token in sentence], 0
            )

        return sentence_tensor

    def forward(self, passage_sentences, answer_sentences):
        passage_tensor = self.get_sentence_tensor(passage_sentences)
        answer_tensor = self.get_sentence_tensor(answer_sentences)

        # Pooling might be better, right now we are just taking the last element
        passage_output = self.lstm(passage_tensor)[0][:, -1, :]
        answer_output = self.lstm(answer_tensor)[0][:, -1, :]

        # Todo might be smart to filter out 0 paddings using something like padpackedtensors etc.
        final_output = self.linear(torch.cat([passage_output, answer_output], dim=-1))

        return final_output
