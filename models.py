from typing import List
from math import ceil

import torch
from torch import nn
from transformers import XLNetModel, RobertaModel

from configurations import device, model_name, maximum_allowed_length


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)

        if maximum_allowed_length > 512:
            old_position_embeddings = self.roberta.embeddings.position_embeddings
            self.roberta.embeddings.position_embeddings = nn.Embedding(maximum_allowed_length, self.roberta.config.hidden_size)
            # Use old weights
            self.roberta.embeddings.position_embeddings.weight.data[:old_position_embeddings.weight.shape[0]] = old_position_embeddings.weight.data

            self.roberta.config.max_position_embeddings = maximum_allowed_length

        self.dropout = nn.Dropout(0.1)

        self.fc = nn.Linear(self.roberta.config.hidden_size, 50)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, pooled_output = self.roberta.forward(input_ids, attention_mask=attention_mask)

        pooled_output = self.dropout(pooled_output)
        pooled_output = self.fc(pooled_output)

        return pooled_output


class CustomXLNETModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.xlnet = XLNetModel.from_pretrained(model_name, mem_len=1024)
        self.linear = nn.Linear(self.xlnet.d_model, 50)
        self.max_seq_length = 128  # How much we want to feed at once

    def forward(self, input_ids, attention_mask):
        # TODO make sure we are masking correctly
        mems = None

        for i in range(ceil(input_ids.shape[-1] / self.max_seq_length)):
            mini_input_ids = input_ids[:, i * self.max_seq_length:i * self.max_seq_length + self.max_seq_length]
            mini_attention_mask = attention_mask[:, i * self.max_seq_length:i * self.max_seq_length + self.max_seq_length]
            output, mems = self.xlnet(mini_input_ids, mems=mems, attention_mask=mini_attention_mask)

        # Pooling might be better, right now we are just taking the last element
        output = output[:, 0, :]

        output = self.linear(output)

        return output
