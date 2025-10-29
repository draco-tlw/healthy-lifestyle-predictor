from os import replace

import numpy as np
import pandas as pd
from scipy.sparse import data
from tables import index


def map_exercise(row):
    row["value"] = 0
    if row["exercise"] == "high":
        row["value"] = 3
    elif row["exercise"] == "medium":
        row["value"] = 2
    elif row["exercise"] == "low":
        row["value"] = 1

    return row


def map_sugar_intake(row):
    row["value"] = 1
    if row["sugar_intake"] == "high":
        row["value"] = 3
    elif row["sugar_intake"] == "medium":
        row["value"] = 2

    return row


def map_smoking(row):
    row["value"] = 0
    if row["smoking"] == "yes":
        row["value"] = 1

    return row


def map_alcohol(row):
    row["value"] = 0
    if row["alcohol"] == "yes":
        row["value"] = 1

    return row


def map_married(row):
    row["value"] = 0
    if row["married"] == "yes":
        row["value"] = 1

    return row


def map_health_risk(row):
    row["value"] = 0
    if row["health_risk"] == "high":
        row["value"] = 1

    return row


def preprocess():
    dataset = pd.read_csv(
        "./lifestyle-and-health-risk-prediction-synthetic-dataset.csv"
    )

    exercise_table = (
        pd.DataFrame(dataset["exercise"].drop_duplicates())
        .apply(map_exercise, axis="columns")
        .set_index("exercise")
    )
    exercise_table.to_csv("./outputs/tables/exercise_table.csv")
    dataset["exercise"] = dataset["exercise"].map(
        lambda e: exercise_table.loc[e, "value"]
    )

    sugar_intake_table = (
        pd.DataFrame(dataset["sugar_intake"].drop_duplicates())
        .apply(map_sugar_intake, axis="columns")
        .set_index("sugar_intake")
    )
    sugar_intake_table.to_csv("./outputs/tables/sugar_intake_table.csv")
    dataset["sugar_intake"] = dataset["sugar_intake"].map(
        lambda si: sugar_intake_table.loc[si, "value"]
    )

    smoking_table = (
        pd.DataFrame(dataset["smoking"].drop_duplicates())
        .apply(map_smoking, axis="columns")
        .set_index("smoking")
    )
    smoking_table.to_csv("./outputs/tables/smoking_table.csv")
    dataset["smoking"] = dataset["smoking"].map(lambda s: smoking_table.loc[s, "value"])

    alcohol_table = (
        pd.DataFrame(dataset["alcohol"].drop_duplicates())
        .apply(map_alcohol, axis="columns")
        .set_index("alcohol")
    )
    alcohol_table.to_csv("./outputs/tables/alcohol_table.csv")
    dataset["alcohol"] = dataset["alcohol"].map(lambda a: alcohol_table.loc[a, "value"])

    married_table = (
        pd.DataFrame(dataset["married"].drop_duplicates())
        .apply(map_married, axis="columns")
        .set_index("married")
    )
    married_table.to_csv("./outputs/tables/married_table.csv")
    dataset["married"] = dataset["married"].map(lambda a: married_table.loc[a, "value"])

    health_risk_table = (
        pd.DataFrame(dataset["health_risk"].drop_duplicates())
        .apply(map_health_risk, axis="columns")
        .set_index("health_risk")
    )
    health_risk_table.to_csv("./outputs/tables/health_risk_table.csv")
    dataset["health_risk"] = dataset["health_risk"].map(
        lambda a: health_risk_table.loc[a, "value"]
    )

    profession_table: pd.DataFrame = (
        dataset.groupby("profession")
        .mean()
        .rename(columns={"health_risk": "health_risk_mean"})
    )
    profession_table = profession_table.loc[:, ["health_risk_mean"]].sort_values(
        "health_risk_mean"
    )
    profession_table.to_csv("./outputs/tables/profession_table.csv")
    dataset["profession"] = dataset["profession"].map(
        lambda p: profession_table.loc[p, "health_risk_mean"]
    )
    dataset.rename(columns={"profession": "profession_health_risk_mean"}, inplace=True)

    x_features = [
        "age",
        "weight",
        "height",
        "exercise",
        "sleep",
        "sugar_intake",
        "smoking",
        "alcohol",
        "married",
        "profession_health_risk_mean",
        "bmi",
    ]
    y_target = "health_risk"

    x_train: np.ndarray = dataset.loc[:, x_features].to_numpy()
    y_train: np.ndarray = dataset.loc[:, y_target].to_numpy()

    return x_train, y_train


# ------------------------
if __name__ == "__main__":
    preprocess()
