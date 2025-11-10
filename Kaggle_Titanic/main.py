from xml.parsers.expat import model
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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

    y = train["Survived"]
    X = train.drop(columns=["Survived"])
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    print(
        "Random Forest trained.\nEvaluation on validation set: ",
        model.score(X_val, y_val),
    )

    print("Cross validation mean score: ", cross_val_score(model, X, y, cv=10).mean())

    param_grid = {
        "max_depth": [None, 3, 4, 5, 6],
        "min_samples_split": [2, 5, 10, 20],
        "n_estimators": [100, 150, 200],
    }

    best_model = do_grid_search(model, X, y, param_grid)
    best_model.fit(X, y)
    y_pred = best_model.predict(test.drop(columns=["PassengerId"]))

    return y_pred


def do_grid_search(model, X, y, param_grid):
    grid_search = GridSearchCV(model, param_grid, verbose=2)
    grid_search.fit(X, y)

    print("Best score obtained from Grid Search: ", grid_search.best_score_)
    print("Best parameters from Grid Search: ", grid_search.best_params_)
    best_model = grid_search.best_estimator_
    return best_model


def xgboosted(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    y_train = train["Survived"]
    X_train = train.drop(columns=["Survived"])

    model = XGBClassifier(objective="binary:logistic", random_state=42)

    # model.fit(X_train, y_train)

    param_grid = {
        "max_depth": [None, 3, 4, 5, 6],
        "n_estimators": [100, 150, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.5, 0.75, 1],
        "scale_pos_weight": [1, 1.25, 1.5],
        "num_parallel_tree": [1, 2, 5],
    }

    best_model = do_grid_search(model, X_train, y_train, param_grid)
    y_pred = best_model.predict(test.drop(columns=["PassengerId"]))
    # Best parameters from Grid Search:  {'learning_rate': 0.01, 'max_depth': None, 'n_estimators': 200, 'num_parallel_tree': 5, 'scale_pos_weight': 1, 'subsample': 0.75}

    return y_pred


def log_reg(train: pd.DataFrame, test: pd.DataFrame) -> None:
    pass


def save_pred(y_pred: np.ndarray, test: pd.DataFrame, filename: str) -> None:
    predictions = pd.DataFrame(
        {"PassengerId": test["PassengerId"], "Survived": y_pred}
    )  # keep headers

    predictions.to_csv(filename, index=False)


if __name__ == "__main__":
    data_path = "./data/"
    train, test = train_test_data_processing(data_path)

    # y_pred = random_forest(train, test)
    # save_pred(y_pred, test, "random_forest_predictions.csv")

    y_pred_xgb = xgboosted(train, test)
    save_pred(y_pred_xgb, test, "xgboosted_predictions.csv")
