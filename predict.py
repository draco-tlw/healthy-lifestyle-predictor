import joblib
import numpy as np
import pandas as pd

exercise_table = pd.read_csv(
    "./outputs/tables/exercise_table.csv", index_col="exercise"
)
sugar_intake_table = pd.read_csv(
    "./outputs/tables/sugar_intake_table.csv", index_col="sugar_intake"
)
smoking_table = pd.read_csv("./outputs/tables/smoking_table.csv", index_col="smoking")
alcohol_table = pd.read_csv("./outputs/tables/alcohol_table.csv", index_col="alcohol")
married_table = pd.read_csv("./outputs/tables/married_table.csv", index_col="married")
profession_table = pd.read_csv(
    "./outputs/tables/profession_table.csv", index_col="profession"
)

model = joblib.load("./outputs/models/linear_logistic_regression_d7.joblib")


# "age",
# "weight",
# "height",
# "exercise",
# "sleep",
# "sugar_intake",
# "smoking",
# "alcohol",
# "married",
# "profession_health_risk_mean",
# "bmi",
def bmi(weight: float, height: float):
    return (weight / (height**2)) * 10**4


data_point = np.array(
    [
        22,  # age
        83,  # weight
        179,  # height
        exercise_table.loc["low", "value"],  # exercise
        7,  # sleep
        sugar_intake_table.loc["medium", "value"],  # sugar_intake
        smoking_table.loc["no", "value"],  # smoking
        alcohol_table.loc["no", "value"],  # alcohol
        married_table.loc["no", "value"],  # married
        profession_table.loc[
            "engineer", "health_risk_mean"
        ],  # profession_health_risk_mean
        bmi(83, 179),  # bmi
    ]
)

x = data_point.reshape(1, -1)

y = model.predict(x)
y_proba = model.predict_proba(x)[:, 1]

print(f"Input features: {data_point.tolist()}")
print("---")
print(f"Predicted health risk: {'high' if y[0] == 1 else 'low'}")
print(f"Predicted health risk proba: {y_proba[0]:.2f}")
