import torch
import numpy as np
import os


def mnist():
    root = r"/Users/rasmusspangsberg/Documents/DTU/semester1/3uger/dtu_mlops/data/corruptmnist/"

    # train data set, we have 5 different data sets
    images = np.empty([0, 1, 28, 28], dtype=np.float32)
    labels = np.empty([0], dtype=np.long)
    for i in range(5):
        path = os.path.join(root, "train_" + str(i) + ".npz")
        with np.load(path) as data:
            a = data["images"]
            a = a.reshape(a.shape[0], 1, 28, 28)
            a = a.astype(np.float32)
            images = np.concatenate((images, a), axis=0)
            labels = np.concatenate((labels, data["labels"].astype(np.long)))

    train_data_set = torch.utils.data.TensorDataset(
        torch.from_numpy(images), torch.from_numpy(labels))
    train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=64, shuffle=True)

    # test data set, only a single data set
    path = os.path.join(root, "test.npz")

    with np.load(path) as data:
        images = data["images"]
        images = images.reshape(images.shape[0], 1, 28, 28)
        images = images.astype(np.float32)
        labels = data["labels"]

    test_data_set = torch.utils.data.TensorDataset(
        torch.from_numpy(images), torch.from_numpy(labels))
    test_loader = torch.utils.data.DataLoader(test_data_set, batch_size=len(test_data_set), shuffle=False)

    return train_loader, test_loader
