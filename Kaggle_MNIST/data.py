import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, TensorDataset


MNIST_SIZE = 28


class Image:

    def __init__(self, data):
        self.pixels = np.array(data).reshape(MNIST_SIZE, MNIST_SIZE)


# utility function


def print_img(img: Image) -> None:
    plt.imshow(img.pixels, cmap="gray")
    plt.axis("off")
    plt.show()


def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    # normalize pixel values
    # one hot encoding of class labels
    return df


def load_train_val_test_data(file_path: str, params: dict, test_size: float = 0.2):
    df = pd.read_csv(file_path)
    df = preprocessing(df)

    batch_size = params["batch_size"]

    if "train" in file_path:
        labels = df["label"]
        df = df.drop("label", axis=1)

        if test_size != 0:

            X_train, X_val, y_train, y_val = train_test_split(
                df, labels, test_size=test_size, random_state=42
            )  # df

            train_loader = df_to_dl(
                X_train, y_train, shuffle=True, batch_size=batch_size
            )
            val_loader = df_to_dl(
                X_val, y_val, shuffle=False, batch_size=batch_size
            )  # No need to shuffle validation data
        else:
            train_loader = df_to_dl(df, labels, shuffle=True, batch_size=batch_size)
            val_loader = None

        return train_loader, val_loader

    elif "test" in file_path:
        test_loader = df_to_dl(df, None, batch_size=batch_size)
        return test_loader

    else:
        raise NotImplementedError("The file path does not correspond to training data.")


def df_to_dl(
    X: pd.DataFrame, y: pd.Series | None, shuffle: bool = False, batch_size=64
) -> DataLoader:
    # (N_data, C, H, W) for data
    #
    data = (
        torch.tensor(X.values).view(-1, 1, MNIST_SIZE, MNIST_SIZE).type(torch.float32)
    )
    if y is not None:
        label = torch.tensor(y.values).type(torch.int64)

        # create dataset
        ds = TensorDataset(data, label)
    else:
        ds = TensorDataset(data)

    # Create DataLoader for batching
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return dl
