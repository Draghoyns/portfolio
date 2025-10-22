import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split


class Image:
    MNIST_SIZE = 28

    def __init__(self, data):
        self.pixels = np.array(data).reshape(self.MNIST_SIZE, self.MNIST_SIZE)


# utility function


def print_img(img: Image) -> None:
    plt.imshow(img.pixels, cmap="gray")
    plt.axis("off")
    plt.show()


def load_data(file_path: str, mode: str = "train"):
    data = pd.read_csv(file_path)

    if mode == "train":
        labels = data["label"]
        data = data.drop("label", axis=1)
        torch_data = torch.tensor(data.values, dtype=torch.float32)

        print(f"Data shape: {torch_data.shape}")

        torch_labels = torch.tensor(labels.values, dtype=torch.int8)

        return train, val
    else:
        return data

    # Create DataLoader for batching
    train_dataset = TensorDataset(
        torch.tensor(train.drop("label", axis=1).values, dtype=torch.float32),
        torch.tensor(train["label"].values, dtype=torch.int8),
    )
    val_dataset = TensorDataset(
        torch.tensor(val.drop("label", axis=1).values, dtype=torch.float32),
        torch.tensor(val["label"].values, dtype=torch.int8),
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    return 0


if __name__ == "__main__":

    train_path = "data/train.csv"
    train, val = load_data(train_path)

    # 2 possibilities
    # CNN from scratch
    # use a pre trained model and fine tune it for a few epochs
