from torch import nn
import torch
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW

from flair.embeddings import StackedEmbeddings, WordEmbeddings, FlairEmbeddings

from helpers import prepare_data
from embeddings import CustomEmbeddings
from models import CustomModel
from datasets import CustomDataset
from configurations import device

preprocess_data = False

if preprocess_data:
    prepare_data('train.json')
    prepare_data('dev.json')

embeddings = StackedEmbeddings(
    [
        WordEmbeddings('glove'),
        FlairEmbeddings('news-forward'),  # Maybe do pooled
        FlairEmbeddings('news-backward'),
        CustomEmbeddings(),
    ]
)

model = CustomModel(embeddings).to(device=device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=1e-3)

train_dataset = CustomDataset('train_processed.json', embeddings)
dev_dataset = CustomDataset('dev_processed.json', embeddings)
# collate fn overwrite is necessary as dataset is not returning tensors
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True, collate_fn=lambda x: x)
dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True, collate_fn=lambda x: x)

for epoch in range(1000):
    total_loss = 0.0
    model.train()
    print(f'Epoch: {epoch}')
    print('Training Model')
    for idx, batch in enumerate(train_loader):
        passage_sentences, answer_sentences, y = list(zip(*batch))
        y = torch.stack(y).to(device=device)
        output = model(passage_sentences, answer_sentences)

        loss = loss_fn(output, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        if idx % 10 == 0:
            print(f'Training Loss: {total_loss}')
            total_loss = 0.0

            # Using maximum of output, select corresponding elements from labels https://discuss.pytorch.org/t/how-to-index-a-tensor-with-another-tensor/25031
            y_selected = y.gather(1, torch.argmax(output, dim=1, keepdim=True))
            correct_guesses = float(torch.sum(y_selected).item())
            total_guesses = float(y_selected.shape[0])

            print(f'Training Acc: {correct_guesses / total_guesses}\n')

    total_loss = 0.0
    model.eval()
    print('Validating Model')

    with torch.no_grad():
        for idx, batch in enumerate(dev_loader):
            passage_sentences, answer_sentences, y = list(zip(*batch))
            y = torch.stack(y).to(device=device)
            output = model(passage_sentences, answer_sentences)

            loss = loss_fn(output, y)

            total_loss += loss.item()

            if idx % 10 == 0:
                print(f'Validation Loss: {total_loss}')
                total_loss = 0.0

                # Using maximum of output, select corresponding elements from labels https://discuss.pytorch.org/t/how-to-index-a-tensor-with-another-tensor/25031
                y_selected = y.gather(1, torch.argmax(output, dim=1, keepdim=True))
                correct_guesses = float(torch.sum(y_selected).item())
                total_guesses = float(y_selected.shape[0])

                print(f'Validation Acc: {correct_guesses / total_guesses}\n')

#  cp *.py /Users/egeozsoy/Google_Drive/Python\ Projects/Record_CommonSense/.
'''
function ClickConnect(){
console.log("Working");
document.querySelector("colab-toolbar-button#connect").click()
}
setInterval(ClickConnect,60000)
'''
