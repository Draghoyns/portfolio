import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# 9 features is okay, we can keep them for now (even if some are irrelevant)


def analyze_data(data: pd.DataFrame) -> None:
    print(data.info())
    print(data.describe())


def preprocess_data(data: pd.DataFrame, mode: str = "train") -> pd.DataFrame:
    # drop features with too many missing values
    # if the number of NaN are above half the total, drop the column and print the column name
    for col in data.columns:
        if data[col].isna().sum() > len(data) / 2:
            print(f"Column '{col}' is dropped due to too many missing values")
            data = data.drop(columns=[col])
        else:
            if data[col].dtype == "object":
                data[col] = data[col].fillna(
                    data[col].mode()[0]
                )  # fill missing categorical values with mode
                # another strategy : fill with default value of same data type
            else:  # numerical column
                data[col] = data[col].fillna(data[col].median())
        # just make sure the same columns are in train and test set

    # very manual here
    # drop irrelevant features
    if mode == "train":
        data = data.drop(columns=["Name", "Ticket", "PassengerId"])
    else:  # test mode
        data = data.drop(columns=["Name", "Ticket"])

    # convert categorical features to numerical (one-hot encoding or label encoding)
    data = pd.get_dummies(data, columns=["Sex", "Embarked"], drop_first=True)
    # normalize numerical features if necessary
    # not for random forest
    return data


def train_test_data_processing(data_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(data_path + "train.csv")
    test_df = pd.read_csv(data_path + "test.csv")

    train = preprocess_data(train_df)
    test = preprocess_data(test_df, mode="test")

    return train, test


def random_forest(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:

    # make the random forest model with the available features (simple features) with the training data
    y_train = train["Survived"]
    X_train = train.drop(columns=["Survived"])

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(test.drop(columns=["PassengerId"]))

    return y_pred


def xgboosted(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    y_train = train["Survived"]
    X_train = train.drop(columns=["Survived"])

    model = XGBClassifier(objective="binary:logistic", random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(test.drop(columns=["PassengerId"]))

    return y_pred


def log_reg(train: pd.DataFrame, test: pd.DataFrame) -> None:
    pass


def save_pred(y_pred: np.ndarray, test: pd.DataFrame, filename: str) -> None:
    predictions = pd.DataFrame(
        {"PassengerId": test["PassengerId"], "Survived": y_pred}
    )  # keep headers

    predictions.to_csv(filename, index=False)


# you should split the dataset to have a validation set to evaluate your model


if __name__ == "__main__":
    data_path = "./data/"
    train, test = train_test_data_processing(data_path)

    y_pred = random_forest(train, test)
    save_pred(y_pred, test, "random_forest_predictions.csv")
    y_pred_xgb = xgboosted(train, test)
    save_pred(y_pred_xgb, test, "xgboosted_predictions.csv")
