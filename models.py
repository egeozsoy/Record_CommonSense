from typing import List

import torch
from torch import nn
from transformers import XLNetModel

from configurations import device, model_name


'''
mems: (optional)
list of torch.FloatTensor (one for each layer): that contains pre-computed hidden-states (key and values in the attention blocks) as output by the model (see mems output below). Can be used to speed up sequential decoding and attend to longer context. To activate mems you need to set up config.mem_len to a positive value which will be the max number of tokens in the memory output by the model. E.g. model = XLNetModel.from_pretrained(‘xlnet-base-case, mem_len=1024) will instantiate a model which can use up to 1024 tokens of memory (in addition to the input it self).
'''


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.xlnet = XLNetModel.from_pretrained(model_name, mem_len=1024)
        self.linear = nn.Linear(self.xlnet.d_model, 50)

    def forward(self, input_ids,attention_mask):
        # TODO make sure we are masking correctly
        output = self.xlnet(input_ids,attention_mask=attention_mask)[0]

        # Pooling might be better, right now we are just taking the last element
        output = output[:, -1, :]

        output = self.linear(output)

        return output
