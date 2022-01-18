import torch
import os.path
import pytest


def check_all_labels_present(data_set):
    classes_present = []
    for _, label in data_set:
        if label not in classes_present:
            classes_present.append(label)

        if len(classes_present) == 10:
            break
    else:
        assert False, "All labels are not present"


@pytest.mark.skipif(not os.path.exists("data/processed/corruptmnist/"),
                    reason="Data files not found")
def test_data():
    train_data_set = torch.load("data/processed/corruptmnist/train.pth")
    test_data_set = torch.load("data/processed/corruptmnist/test.pth")

    assert len(train_data_set) == 25000, "Training data set length is incorrect"
    assert len(test_data_set) == 5000, "Test data set length is incorrect"

    x = next(iter(train_data_set))[0]
    assert x.shape == torch.Size([1, 28, 28]), "Train data set sample has incorrect shape"

    check_all_labels_present(train_data_set)
    check_all_labels_present(test_data_set)
