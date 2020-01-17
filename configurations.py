from pathlib import Path
import logging
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model_name = 'roberta-large'
batch_size = 2
accumulation_steps = 32 // batch_size

maximum_allowed_length = 1600
num_workers = 1
warmup_steps = 0

gdrive_path = Path('/content/gdrive/My Drive/Python Projects/Record_CommonSense')
log_path = gdrive_path / 'output.log'
model_path = gdrive_path / 'model.pth'
optimizer_path = gdrive_path / 'optimizer.pth'
scheduler_path = gdrive_path / 'scheduler.pth'

logging.basicConfig(filename=log_path, filemode='w', format='%(message)s', level=0)
