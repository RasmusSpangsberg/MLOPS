import os

import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
import numpy as np

from model import MyAwesomeModel
import matplotlib.pyplot as plt


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

    losses = []

    epochs = 30
    for e in range(epochs):
        print("epoch:", e)
        running_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print("loss:", running_loss)
        losses.append(running_loss)

    plt.plot(list(range(epochs)), losses)
    plt.xticks(np.arange(0, epochs, 1))
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig("reports/figures/training_run.png")

    torch.save(model.state_dict(), 'models/checkpoint.pth')


if __name__ == '__main__':
    train()
