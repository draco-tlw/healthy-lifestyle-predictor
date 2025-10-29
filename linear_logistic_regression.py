import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from calculate_costs import calculate_costs
from plot import plot
from preprocess import preprocess

x_train, y_train = preprocess()

model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(5)),
        ("linear", LogisticRegression(max_iter=10000, verbose=True)),
    ]
)
model.fit(x_train, y_train)

model_name = "linear_logistic_regression_d5"
joblib.dump(model, f"./outputs/models/{model_name}.joblib")
print(f"Model saved successfully to '{model_name}.joblib'")

y_predict_classes = model.predict(x_train)

y_predict_probabilities = model.predict_proba(x_train)[:, 1]

plot(
    x_train,
    y_train,
    y_predict_probabilities,
    model_name,
    model_name,
)
calculate_costs(y_train, y_predict_probabilities)
