import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm as tdqm

from data import load_train_val_test_data


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            1, 8, kernel_size=3, stride=1, padding=1
        )  # conv keep size
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # divide size by 2
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)  # which is why size = 7x7

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # flatten before fc layer
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


def evaluate(model, data_loader, device, cv: bool = False):
    if cv:
        acc = 0
        pass
    else:

        model.eval()
        num_correct = 0
        num_samples = 0
        with torch.no_grad():
            for batch, (x, y) in enumerate(data_loader):
                # Move data to the same device as model
                x, y = x.to(device), y.to(device)
                # per batch
                outputs = model(x)
                _, pred = outputs.max(1)
                num_correct += (pred == y).sum()
                num_samples += pred.size(0)

            acc = num_correct / num_samples * 100
            print(f"Validation accuracy: {acc:.2f}")

    model.train()
    return acc


def train_model(model, params, train_loader, device):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

    model.to(device)

    for epoch in range(params["num_epochs"]):
        print(f"\nEpoch {epoch+1}/{params['num_epochs']}")
        for batch, (data, labels) in enumerate(tdqm(train_loader)):
            data, labels = data.to(device), labels.to(device)

            # forward pass

            outputs = model(data)
            loss = criterion(outputs, labels)

            # backprop
            optimizer.zero_grad()
            loss.backward()

            # optimization
            optimizer.step()


def save_predictions(model, test_loader, device, output_path: str = "predictions.csv"):
    model.eval()
    model.to(device)
    predictions = []
    with torch.no_grad():
        for batch, (x,) in enumerate(test_loader):
            x = x.to(device)
            outputs = model(x)
            _, pred = outputs.max(1)
            predictions.extend(pred.cpu().numpy())

    df = pd.DataFrame(predictions, columns=["Label"])
    df.index.name = "ImageId"
    df.index += 1  # ImageId starts from 1
    df.to_csv(output_path)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":

    params = {
        "input_size": 784,  # 28x28 pixels (not directly used in CNN)
        "num_classes": 10,  # digits 0-9
        "learning_rate": 0.001,
        "batch_size": 64,
        "num_epochs": 25,  # Reduced for demonstration purposes
    }

    # data setup
    train_path = "data/train.csv"
    test_path = "data/test.csv"

    train_loader, val_loader = load_train_val_test_data(
        train_path, params, test_size=0.0
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")

    # model setup
    model = SimpleCNN(num_classes=params["num_classes"])

    # training loop
    print("\nTraining SimpleCNN ...")
    train_model(model, params, train_loader, device)

    # acc_dict = {}
    # for n_epochs in [10, 15, 20, 25, 50]:
    #     params["num_epochs"] = n_epochs
    #     train_model(model, params, train_loader, device)
    #     acc_dict[n_epochs] = evaluate(model, val_loader, device, cv=False)
    # print("\nValidation accuracies for different epochs:", acc_dict)

    test_loader = load_train_val_test_data(test_path, params)

    save_predictions(
        model, test_loader, device, output_path="SimpleCNN_predictions.csv"
    )
