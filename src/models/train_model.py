import argparse
import sys

import matplotlib.pyplot as plt
import torch
from model import MyAwesomeModel
from torch import nn, optim
from torch.utils.data import DataLoader


def train():
    print("Training started...")
    parser = argparse.ArgumentParser(description='Evaluation arguments')
    parser.add_argument('load_data_from', default="")
    args = parser.parse_args(sys.argv[1:])

    train_data = torch.load(args.load_data_from)
    train_loader = DataLoader(train_data, batch_size=64,
                              shuffle=False)

    model = MyAwesomeModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    losses = []

    epochs = 60
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
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig("reports/figures/training_run.png")

    torch.save(model.state_dict(), 'models/checkpoint.pth')


if __name__ == '__main__':
    train()
