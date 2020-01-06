import json
import numpy as np
import torch

from configurations import device


def prepare_data(json_file_path):
    with open(json_file_path) as f:
        json_file = json.load(f)
        datas = json_file['data']

        prepared_data = []

        for data in datas:
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

    with open('{}_processed.json'.format(json_file_path.split('.json')[0]), 'w') as f:
        json.dump(prepared_data, f)


def pad_tensors(tensor_list):
    max_length = max(len(elem) for elem in tensor_list)

    padded_tensor = torch.zeros((len(tensor_list)), max_length, dtype=torch.long)

    valid_ids = torch.zeros((len(tensor_list)), max_length)

    for idx, elem in enumerate(tensor_list):
        padded_tensor[idx, :len(elem)] = elem
        valid_ids[idx, :len(elem)] = 1

    return padded_tensor.to(device=device), valid_ids.to(device=device)
