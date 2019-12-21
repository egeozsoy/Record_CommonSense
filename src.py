import json
import numpy as np
from transformers import BasicTokenizer
import collections
from transformers.tokenization_bert import whitespace_tokenize, _is_punctuation, load_vocab

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

prepared_data = []
special_tokens = ['[ENT{}]'.format(i) for i in range(51)] + ['[ANS]']


class CustomTokenizer(BasicTokenizer):

    def __init__(self, do_lower_case, additional_tokens, vocab_file='vocab.txt'):
        super(CustomTokenizer, self).__init__(do_lower_case, never_split=additional_tokens)

        self.vocab = load_vocab(vocab_file)

        for idx, additional_token in enumerate(additional_tokens):
            self.vocab[additional_token] = 100000 + idx

        self.unk_token = "[UNK]"
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])

    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        if never_split is not None and text in never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char) and char is not '[' and char is not ']':
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _convert_token_to_id(self, token):
        """ Converts a token (str/unicode) in an id using the vocab. """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = ' '.join(tokens).replace(' ##', '').strip()
        return out_string

    def convert_tokens_to_ids(self, tokens):
        all_ids = list(self._convert_token_to_id(token) for token in tokens)
        return all_ids

    def convert_ids_to_tokens(self, ids):
        all_ids = list(self._convert_id_to_token(id) for id in ids)
        return all_ids

    def tokenize(self, text, never_split=None):
        never_split = self.never_split + (never_split if never_split is not None else [])
        text = self._clean_text(text)
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in never_split:
                token = token.lower()
                token = self._run_strip_accents(token)

            # This is the only change to the standad version, we want to preserve our entities tokens

            split_tokens.extend(self._run_split_on_punc(token, never_split))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens


tokenizer = CustomTokenizer(do_lower_case=False, additional_tokens=special_tokens)

with open('train_processed.json') as f:
    json_file = json.load(f)

    for (passage_text, answer_text, answer_vector) in json_file:
        tokenized_passage_text = tokenizer.tokenize(passage_text)
        passage_text_ids = tokenizer.convert_tokens_to_ids(tokenized_passage_text)
        tokenized_answer_text = tokenizer.tokenize(answer_text)
        answer_text_ids = tokenizer.convert_tokens_to_ids(tokenized_answer_text)
        answer_vector = np.array(answer_vector)
