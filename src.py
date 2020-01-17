from pathlib import Path

from torch import nn
import torch
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup

from helpers import prepare_data, pad_tensors, nvidia_debug_output, print_log
from embeddings import add_custom_tokens_to_tokenizer
from models import CustomModel
from datasets import CustomDataset
from configurations import device, model_name, batch_size, accumulation_steps, num_workers, gdrive_path, model_path, warmup_steps, optimizer_path, \
    scheduler_path


class GradientAccumulator:
    def __init__(self, bs: int):
        self.bs: int = bs
        self.acc: int = 0

    def update_gradients(self, optimizer: Optimizer, scheduler, loss):
        loss.backward()

        if self.acc >= self.bs:
            self.acc = 0

            optimizer.step()
            scheduler.step()

            optimizer.zero_grad()

        else:
            self.acc += 1


preprocess_data: bool = False

if preprocess_data:
    prepare_data('train.json')
    prepare_data('dev.json')

model: nn.Module = CustomModel().to(device=device)

loss_fn: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
optimizer: Optimizer = AdamW(model.parameters(), lr=2e-5)

tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained(model_name)
add_custom_tokens_to_tokenizer(tokenizer)
# Because we added new tokens
model.roberta.resize_token_embeddings(len(tokenizer))

if model_path.exists():
    print_log('Loading Model')
    model.load_state_dict(torch.load(model_path.absolute()))

train_dataset: torch.utils.data.Dataset = CustomDataset('train_processed.json', tokenizer)
dev_dataset: torch.utils.data.Dataset = CustomDataset('dev_processed.json', tokenizer)
# collate fn overwrite is necessary as dataset is not returning tensors
train_loader: torch.utils.data.DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
                                                       collate_fn=lambda x: x)
dev_loader: torch.utils.data.DataLoader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
                                                     collate_fn=lambda x: x)

grad_accumulator: GradientAccumulator = GradientAccumulator(accumulation_steps)

prev_total_training_loss: float = 0.0
prev_total_validation_loss: float = 0.0

epochs: int = 5

total_training_steps = len(train_loader) // accumulation_steps * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps)

if optimizer_path.exists():
    print_log('Loading Optimizer')
    optimizer.load_state_dict(torch.load(optimizer_path.absolute()))

if scheduler_path.exists():
    print_log('Loading Scheduler')
    scheduler.load_state_dict(torch.load(scheduler_path.absolute()))

for epoch in range(epochs):
    total_loss: float = 0.0
    total_correct_guesses: float = 0.0
    total_total_guesses: float = 0.0
    total_training_loss: float = 0.0
    model.train()
    print_log(f'Epoch: {epoch}')
    print_log('Training Model')
    for idx, batch in enumerate(train_loader):
        input_ids, token_type_ids, y = list(zip(*batch))

        input_ids, token_type_ids, valid_ids = pad_tensors(input_ids, token_type_ids)
        y = torch.stack(y).to(device=device)

        output = model(input_ids, valid_ids, token_type_ids)

        loss = loss_fn(output, y)

        grad_accumulator.update_gradients(optimizer, scheduler, loss)

        total_loss += loss.item()
        total_training_loss += loss.item()

        # Using maximum of output, select corresponding elements from labels https://discuss.pytorch.org/t/how-to-index-a-tensor-with-another-tensor/25031
        y_selected = y.gather(1, torch.argmax(output, dim=1, keepdim=True))
        correct_guesses = float(torch.sum(y_selected).item())
        total_guesses = float(y_selected.shape[0])

        total_correct_guesses += correct_guesses
        total_total_guesses += total_guesses

        if idx % (len(train_loader) // 10) == 0 and idx > 0:
            print_log(f'Training Loss: {total_loss:.4f}, Acc: {total_correct_guesses / total_total_guesses:.4f}\n')

            print_log(torch.argmax(output, dim=1).cpu().numpy())
            print_log(torch.argmax(y, dim=1).cpu().numpy())

            total_loss = 0.0
            total_correct_guesses = 0.0
            total_total_guesses = 0.0

            if device.type != 'cpu' and epoch == 0:
                nvidia_debug_output()

    print_log(f'TL: {total_training_loss:.4f} PTL: {prev_total_training_loss:.4f}')
    prev_total_training_loss = total_training_loss

    total_loss = 0.0
    total_correct_guesses = 0.0
    total_total_guesses = 0.0
    total_validation_loss = 0.0
    model.eval()
    print_log('Validating Model')

    with torch.no_grad():
        for idx, batch in enumerate(dev_loader):
            input_ids, token_type_ids, y = list(zip(*batch))

            input_ids, token_type_ids, valid_ids = pad_tensors(input_ids, token_type_ids)
            y = torch.stack(y).to(device=device)

            output = model(input_ids, valid_ids, token_type_ids)

            loss = loss_fn(output, y)

            total_loss += loss.item()
            total_validation_loss += loss.item()

            # Using maximum of output, select corresponding elements from labels https://discuss.pytorch.org/t/how-to-index-a-tensor-with-another-tensor/25031
            y_selected = y.gather(1, torch.argmax(output, dim=1, keepdim=True))
            correct_guesses = float(torch.sum(y_selected).item())
            total_guesses = float(y_selected.shape[0])

            total_correct_guesses += correct_guesses
            total_total_guesses += total_guesses

            if idx % (len(dev_loader) // 10) == 0 and idx > 0:
                print_log(f'Validation Loss: {total_loss:.4f}, Acc: {total_correct_guesses / total_total_guesses:.4f}\n')

                total_loss = 0.0
                total_correct_guesses = 0.0
                total_total_guesses = 0.0

    print_log(f'Validation TL: {total_validation_loss:.4f} PTL: {prev_total_validation_loss:.4f}')
    prev_total_validation_loss = total_validation_loss

    if gdrive_path.exists():
        torch.save(model.state_dict(), model_path.absolute())
        torch.save(optimizer.state_dict(), optimizer_path.absolute())
        torch.save(scheduler.state_dict(), scheduler_path.absolute())

#  cp *.py /Users/egeozsoy/Google_Drive/Python\ Projects/Record_CommonSense/.
'''
function ClickConnect(){
console.log("Working");
document.querySelector("colab-toolbar-button#connect").click()
}
setInterval(ClickConnect,60000)
'''
