import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# model_name = 'xlnet-large-cased'
model_name = 'roberta-base'
batch_size = 4
accumulation_steps = 32 // batch_size

maximum_allowed_length = 1600
num_workers = 1
