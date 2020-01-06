import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model_name = 'xlnet-large-cased'
batch_size = 2
accumulation_steps = 16
