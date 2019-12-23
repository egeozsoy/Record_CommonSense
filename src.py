from torch import nn
import torch
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW

from flair.embeddings import StackedEmbeddings, WordEmbeddings, FlairEmbeddings

from helpers import prepare_data
from embeddings import CustomEmbeddings
from models import CustomModel
from datasets import CustomDataset

preprocess_data = True

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

model = CustomModel(embeddings)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters())

train_dataset = CustomDataset('train_processed.json')
dev_dataset = CustomDataset('dev_processed.json')
# collate fn overwrite is necessary as dataset is not returning tensors
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1, pin_memory=True, collate_fn=lambda x: x)
dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=True, num_workers=1, pin_memory=True, collate_fn=lambda x: x)

for idx, batch in enumerate(train_loader):
    passage_sentences, answer_sentences, y = list(zip(*batch))
    y = torch.stack(y)
    output = model(passage_sentences, answer_sentences)

    loss = loss_fn(output, y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if idx % 10 == 0:
        print(loss.item())

for idx, batch in enumerate(dev_loader):
    passage_sentences, answer_sentences, y = list(zip(*batch))
    y = torch.stack(y)
    output = model(passage_sentences, answer_sentences)

    loss = loss_fn(output, y)

    if idx % 10 == 0:
        print(loss.item())
