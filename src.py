from torch import nn
import torch
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW

from helpers import prepare_data
from embeddings import add_custom_tokens_to_tokenizer
from models import CustomModel
from datasets import CustomDataset
from configurations import device, model_name, batch_size, accumulation_steps
from transformers import XLNetTokenizer
from helpers import pad_tensors


class GradientAccumulator:
    def __init__(self, bs):
        self.bs = bs
        self.acc = 0

    def update_gradients(self, optimizer, loss):
        loss.backward()

        if self.acc >= self.bs:
            self.acc = 0
            optimizer.step()

            optimizer.zero_grad()

        else:
            self.acc += 1


preprocess_data = False

if preprocess_data:
    prepare_data('train.json')
    prepare_data('dev.json')

model = CustomModel().to(device=device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=5e-5)

tokenizer = XLNetTokenizer.from_pretrained(model_name)
add_custom_tokens_to_tokenizer(tokenizer)
# Because we added new tokens
model.xlnet.resize_token_embeddings(len(tokenizer))

train_dataset = CustomDataset('train_processed.json', tokenizer)
dev_dataset = CustomDataset('dev_processed.json', tokenizer)
# collate fn overwrite is necessary as dataset is not returning tensors
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=lambda x: x)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=lambda x: x)

grad_accumulator = GradientAccumulator(accumulation_steps)

for epoch in range(1000):
    total_loss = 0.0
    total_correct_guesses = 0.0
    total_total_guesses = 0.0
    model.train()
    print(f'Epoch: {epoch}')
    print('Training Model')
    for idx, batch in enumerate(train_loader):
        input_ids, y = list(zip(*batch))

        input_ids, valid_ids = pad_tensors(input_ids)
        y = torch.stack(y).to(device=device)

        output = model(input_ids, valid_ids)

        loss = loss_fn(output, y)

        grad_accumulator.update_gradients(optimizer, loss)

        total_loss += loss.item()

        # Using maximum of output, select corresponding elements from labels https://discuss.pytorch.org/t/how-to-index-a-tensor-with-another-tensor/25031
        y_selected = y.gather(1, torch.argmax(output, dim=1, keepdim=True))
        correct_guesses = float(torch.sum(y_selected).item())
        total_guesses = float(y_selected.shape[0])

        total_correct_guesses += correct_guesses
        total_total_guesses += total_guesses

        if idx % 1000 == 0:
            print(f'Training Loss: {total_loss:.4f}, Acc: {total_correct_guesses / total_total_guesses:.4f}\n')

            total_loss = 0.0
            total_correct_guesses = 0.0
            total_total_guesses = 0.0

    total_loss = 0.0
    total_correct_guesses = 0.0
    total_total_guesses = 0.0
    model.eval()
    print('Validating Model')

    with torch.no_grad():
        for idx, batch in enumerate(dev_loader):
            input_ids, y = list(zip(*batch))
            input_ids, valid_ids = pad_tensors(input_ids)
            y = torch.stack(y).to(device=device)

            output = model(input_ids, valid_ids)

            loss = loss_fn(output, y)

            total_loss += loss.item()

            # Using maximum of output, select corresponding elements from labels https://discuss.pytorch.org/t/how-to-index-a-tensor-with-another-tensor/25031
            y_selected = y.gather(1, torch.argmax(output, dim=1, keepdim=True))
            correct_guesses = float(torch.sum(y_selected).item())
            total_guesses = float(y_selected.shape[0])

            total_correct_guesses += correct_guesses
            total_total_guesses += total_guesses

            if idx % 1000 == 0:
                print(f'Validation Loss: {total_loss:.4f}, Acc: {total_correct_guesses / total_total_guesses:.4f}\n')

#  cp *.py /Users/egeozsoy/Google_Drive/Python\ Projects/Record_CommonSense/.
'''
function ClickConnect(){
console.log("Working");
document.querySelector("colab-toolbar-button#connect").click()
}
setInterval(ClickConnect,60000)
'''
