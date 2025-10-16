import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# 9 features is okay, we can keep them for now (even if some are irrelevant)


def analyze_data(data: pd.DataFrame) -> None:
    print(data.info())
    print(data.describe())


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
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
    data = data.drop(columns=["Name", "Ticket", "PassengerId"])

    # convert categorical features to numerical (one-hot encoding or label encoding)
    data = pd.get_dummies(data, columns=["Sex", "Embarked"], drop_first=True)
    # normalize numerical features if necessary
    # not for random forest
    return data


def random_forest(data_path: str) -> None:

    train_df = pd.read_csv(data_path + "train.csv")
    test_df = pd.read_csv(data_path + "test.csv")

    train = preprocess_data(train_df)
    test = preprocess_data(test_df)

    # make the random forest model with the available features (simple features) with the training data
    y_train = train["Survived"]
    X_train = train.drop(columns=["Survived"])

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(test)

    predictions = pd.DataFrame(
        {"PassengerId": test_df["PassengerId"], "Survived": y_pred}
    )  # keep headers

    predictions.to_csv("predictions.csv", index=False)

    # predict the test data
    # save to predictions.csv


if __name__ == "__main__":
    data_path = "./data/"
    random_forest(data_path)
