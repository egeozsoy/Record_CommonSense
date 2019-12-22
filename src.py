import json
import numpy as np

from torch import nn
import torch
from typing import List
from flair.embeddings import StackedEmbeddings, WordEmbeddings, FlairEmbeddings, Sentence, TokenEmbeddings

preprocess_data = False

if preprocess_data:

    with open('train.json') as f:
        json_file = json.load(f)
        datas = json_file['data']

        prepared_data = []

        for data in datas[:10]:
            # One element of data
            counter = 0
            entities = data['passage']['entities']

            entity_map = {}
            entity_ids = []
            replacements = {}
            passage_text: str = data['passage']['text']

            # maybe ignore upper lower case etc
            for entity in entities:
                entity_text = passage_text[entity['start']:entity['end'] + 1]

                if entity_text not in entity_map:
                    entity_map[entity_text] = counter
                    counter += 1

                id_value = entity_map[entity_text]

                entity_ids.append(id_value)

                replacements[entity_text] = '[ENT{}]'.format(id_value)

            for key, value in replacements.items():
                passage_text = passage_text.replace(key, value)

            # We might have multiple queries per text
            for question in data['qas']:
                answer_entities = question['answers']
                answer_entity_ids = []

                for answer_entity in answer_entities:
                    entity_text = answer_entity['text']
                    id_value = entity_map[entity_text]
                    answer_entity_ids.append(id_value)

                answer_text: str = question['query']

                for key, value in replacements.items():
                    answer_text = answer_text.replace(key, value)

                answer_text = answer_text.replace('@placeholder', '[ANS]')

                answer_vector = np.zeros((50))  # Assume certain amount of maximum entities

                try:
                    answer_vector[np.array(answer_entity_ids)] = 1

                except Exception as e:
                    continue

                prepared_data.append((passage_text, answer_text, list(answer_vector)))

    with open('train_processed.json', 'w') as f:
        json.dump(prepared_data, f)


class CustomModel(nn.Module):
    def __init__(self, embeddings):
        super(CustomModel, self).__init__()
        self.embeddings = embeddings

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
        passage_sentences = [passage_sentences]
        answer_sentences = [answer_sentences]

        passage_tensor = self.get_sentence_tensor(passage_sentences)
        answer_tensor = self.get_sentence_tensor(answer_sentences)

        print('h')


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


embeddings = StackedEmbeddings(
    [
        WordEmbeddings('glove'),
        FlairEmbeddings('news-forward'),
        FlairEmbeddings('news-backward'),
        CustomEmbeddings(),

    ]
)

model = CustomModel(embeddings)
prepared_data = []
special_tokens = ['[ENT{}]'.format(i) for i in range(51)] + ['[ANS]']

with open('train_processed.json') as f:
    json_file = json.load(f)

    for (passage_text, answer_text, answer_vector) in json_file:
        passage_sentence = Sentence(passage_text)
        answer_sentence = Sentence(answer_text)
        answer_vector = np.array(answer_vector)

        forward_pass = model(passage_sentence, answer_sentence)
