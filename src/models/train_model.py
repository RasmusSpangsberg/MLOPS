import argparse
import sys
import os

import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset

from model import MyAwesomeModel


def train():
    print("Training day and night")

    data_root = r"data/processed/corruptmnist"
    train_images = torch.load(os.path.join(data_root, "train_image.pth"))
    train_labels = torch.load(os.path.join(data_root, "train_labels.pth"))

    data_set = TensorDataset(train_images, train_labels)
    train_loader = torch.utils.data.DataLoader(data_set, batch_size=64)

    model = MyAwesomeModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    epochs = 40
    for e in range(epochs):
        running_loss = 0
        print("epoch:", e)
        for images, labels in train_loader:
            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss = loss.item()
        print(running_loss)
    torch.save(model.state_dict(), 'models/checkpoint.pth')


if __name__ == '__main__':
    train()
