import torch
import numpy as np
import os


def mnist():
    raw_data_path = r"data\raw\corruptmnist"
    processed_data_path = r"data\processed\corruptmnist"

    if not os.path.exists(processed_data_path):
        os.mkdir(processed_data_path)

    # train data set, we have 5 different data sets
    images = np.empty([0, 1, 28, 28])
    labels = np.empty([0])
    for i in range(5):
        path = os.path.join(raw_data_path, "train_" + str(i) + ".npz")
        with np.load(path) as data:
            a = data["images"]
            a = a.reshape(a.shape[0], 1, 28, 28)
            images = np.concatenate((images, a), axis=0)
            labels = np.concatenate((labels, data["labels"]))

    # convert to torch tensors
    images = torch.from_numpy(images).type(torch.FloatTensor)
    labels = torch.from_numpy(labels).type(torch.LongTensor)

    torch.save(images, os.path.join(processed_data_path, "train_image.pth"))
    torch.save(labels, os.path.join(processed_data_path, "train_labels.pth"))

    # test data set, only a single data set
    path = os.path.join(raw_data_path, "test.npz")

    with np.load(path) as data:
        images = data["images"]
        images = images.reshape(images.shape[0], 1, 28, 28)
        labels = data["labels"]

    # convert to torch tensors
    images = torch.from_numpy(images).type(torch.FloatTensor)
    labels = torch.from_numpy(labels).type(torch.LongTensor)

    torch.save(images, os.path.join(processed_data_path, "test_image.pth"))
    torch.save(labels, os.path.join(processed_data_path, "test_labels.pth"))


if __name__ == "__main__":
    mnist()
