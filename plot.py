import matplotlib.pyplot as plt
import numpy as np


def plot(
    x_train: np.ndarray,
    y_train: np.ndarray,
    y_predict: np.ndarray,
    label: str,
    files_postfix: str,
):
    class_0 = y_train == 0
    class_1 = y_train == 1

    # --- health_risk vs age
    plt.figure(figsize=(40, 10))
    plt.scatter(
        x_train[class_0, 0],
        y_train[class_0],
        s=75,
        c="red",
        label="Actual Data (Class 0)",
    )
    plt.scatter(
        x_train[class_1, 0],
        y_train[class_1],
        s=75,
        c="blue",
        label="Actual Data (Class 1)",
    )
    plt.scatter(
        x_train[:, 0], y_predict, c="green", alpha=0.2, label="Predicted Probability"
    )
    plt.axhline(
        y=0.5, color="green", linestyle="--", label="Decision Boundary (0.5 Prob)"
    )
    plt.title(f"Health Risk vs. Age ({label})", fontsize=24)
    plt.xlabel("Age", fontsize=18)
    plt.ylabel("Health Risk", fontsize=18)
    plt.grid(True)
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
    plt.ticklabel_format(style="plain", axis="x")
    plt.savefig(f"outputs/plots/health_risk_vs_age_{files_postfix}_plot.png")

    # --- health_risk vs weight
    plt.figure(figsize=(40, 10))
    plt.scatter(
        x_train[class_0, 1],
        y_train[class_0],
        s=75,
        c="red",
        label="Actual Data (Class 0)",
    )
    plt.scatter(
        x_train[class_1, 1],
        y_train[class_1],
        s=75,
        c="blue",
        label="Actual Data (Class 1)",
    )
    plt.scatter(
        x_train[:, 1], y_predict, c="green", alpha=0.2, label="Predicted Probability"
    )
    plt.axhline(
        y=0.5, color="green", linestyle="--", label="Decision Boundary (0.5 Prob)"
    )
    plt.title(f"Health Risk vs. Weight ({label})", fontsize=24)
    plt.xlabel("Weight", fontsize=18)
    plt.ylabel("Health Risk", fontsize=18)
    plt.grid(True)
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
    plt.ticklabel_format(style="plain", axis="x")
    plt.savefig(f"outputs/plots/health_risk_vs_weight_{files_postfix}_plot.png")

    # --- health_risk vs height
    plt.figure(figsize=(40, 10))
    plt.scatter(
        x_train[class_0, 2],
        y_train[class_0],
        s=75,
        c="red",
        label="Actual Data (Class 0)",
    )
    plt.scatter(
        x_train[class_1, 2],
        y_train[class_1],
        s=75,
        c="blue",
        label="Actual Data (Class 1)",
    )
    plt.scatter(
        x_train[:, 2], y_predict, c="green", alpha=0.2, label="Predicted Probability"
    )
    plt.axhline(
        y=0.5, color="green", linestyle="--", label="Decision Boundary (0.5 Prob)"
    )
    plt.title(f"Health Risk vs. Height ({label})", fontsize=24)
    plt.xlabel("Height", fontsize=18)
    plt.ylabel("Health Risk", fontsize=18)
    plt.grid(True)
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
    plt.ticklabel_format(style="plain", axis="x")
    plt.savefig(f"outputs/plots/health_risk_vs_height_{files_postfix}_plot.png")

    # --- health_risk vs exercise
    plt.figure(figsize=(40, 10))
    plt.scatter(
        x_train[class_0, 3],
        y_train[class_0],
        s=75,
        c="red",
        label="Actual Data (Class 0)",
    )
    plt.scatter(
        x_train[class_1, 3],
        y_train[class_1],
        s=75,
        c="blue",
        label="Actual Data (Class 1)",
    )
    plt.scatter(
        x_train[:, 3], y_predict, c="green", alpha=0.2, label="Predicted Probability"
    )
    plt.axhline(
        y=0.5, color="green", linestyle="--", label="Decision Boundary (0.5 Prob)"
    )
    plt.title(f"Health Risk vs. Exercise ({label})", fontsize=24)
    plt.xlabel("Exercise", fontsize=18)
    plt.ylabel("Health Risk", fontsize=18)
    plt.grid(True)
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
    plt.ticklabel_format(style="plain", axis="x")
    plt.savefig(f"outputs/plots/health_risk_vs_exercise_{files_postfix}_plot.png")

    # --- health_risk vs sleep
    plt.figure(figsize=(40, 10))
    plt.scatter(
        x_train[class_0, 4],
        y_train[class_0],
        s=75,
        c="red",
        label="Actual Data (Class 0)",
    )
    plt.scatter(
        x_train[class_1, 4],
        y_train[class_1],
        s=75,
        c="blue",
        label="Actual Data (Class 1)",
    )
    plt.scatter(
        x_train[:, 4], y_predict, c="green", alpha=0.2, label="Predicted Probability"
    )
    plt.axhline(
        y=0.5, color="green", linestyle="--", label="Decision Boundary (0.5 Prob)"
    )
    plt.title(f"Health Risk vs. Sleep ({label})", fontsize=24)
    plt.xlabel("Sleep", fontsize=18)
    plt.ylabel("Health Risk", fontsize=18)
    plt.grid(True)
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
    plt.ticklabel_format(style="plain", axis="x")
    plt.savefig(f"outputs/plots/health_risk_vs_sleep_{files_postfix}_plot.png")

    # --- health_risk vs sugar_intake
    plt.figure(figsize=(40, 10))
    plt.scatter(
        x_train[class_0, 5],
        y_train[class_0],
        s=75,
        c="red",
        label="Actual Data (Class 0)",
    )
    plt.scatter(
        x_train[class_1, 5],
        y_train[class_1],
        s=75,
        c="blue",
        label="Actual Data (Class 1)",
    )
    plt.scatter(
        x_train[:, 5], y_predict, c="green", alpha=0.2, label="Predicted Probability"
    )
    plt.axhline(
        y=0.5, color="green", linestyle="--", label="Decision Boundary (0.5 Prob)"
    )
    plt.title(f"Health Risk vs. Sugar Intake ({label})", fontsize=24)
    plt.xlabel("Sugar Intake", fontsize=18)
    plt.ylabel("Health Risk", fontsize=18)
    plt.grid(True)
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
    plt.ticklabel_format(style="plain", axis="x")
    plt.savefig(f"outputs/plots/health_risk_vs_sugar_intake_{files_postfix}_plot.png")

    # --- health_risk vs smoking
    plt.figure(figsize=(40, 10))
    plt.scatter(
        x_train[class_0, 6],
        y_train[class_0],
        s=75,
        c="red",
        label="Actual Data (Class 0)",
    )
    plt.scatter(
        x_train[class_1, 6],
        y_train[class_1],
        s=75,
        c="blue",
        label="Actual Data (Class 1)",
    )
    plt.scatter(
        x_train[:, 6], y_predict, c="green", alpha=0.2, label="Predicted Probability"
    )
    plt.axhline(
        y=0.5, color="green", linestyle="--", label="Decision Boundary (0.5 Prob)"
    )
    plt.title(f"Health Risk vs. Smoking ({label})", fontsize=24)
    plt.xlabel("Smoking", fontsize=18)
    plt.ylabel("Health Risk", fontsize=18)
    plt.grid(True)
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
    plt.ticklabel_format(style="plain", axis="x")
    plt.savefig(f"outputs/plots/health_risk_vs_smoking_{files_postfix}_plot.png")

    # --- health_risk vs alcohol
    plt.figure(figsize=(40, 10))
    plt.scatter(
        x_train[class_0, 7],
        y_train[class_0],
        s=75,
        c="red",
        label="Actual Data (Class 0)",
    )
    plt.scatter(
        x_train[class_1, 7],
        y_train[class_1],
        s=75,
        c="blue",
        label="Actual Data (Class 1)",
    )
    plt.scatter(
        x_train[:, 7], y_predict, c="green", alpha=0.2, label="Predicted Probability"
    )
    plt.axhline(
        y=0.5, color="green", linestyle="--", label="Decision Boundary (0.5 Prob)"
    )
    plt.title(f"Health Risk vs. Alcohol ({label})", fontsize=24)
    plt.xlabel("Alcohol", fontsize=18)
    plt.ylabel("Health Risk", fontsize=18)
    plt.grid(True)
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
    plt.ticklabel_format(style="plain", axis="x")
    plt.savefig(f"outputs/plots/health_risk_vs_alcohol_{files_postfix}_plot.png")

    # --- health_risk vs married
    plt.figure(figsize=(40, 10))
    plt.scatter(
        x_train[class_0, 8],
        y_train[class_0],
        s=75,
        c="red",
        label="Actual Data (Class 0)",
    )
    plt.scatter(
        x_train[class_1, 8],
        y_train[class_1],
        s=75,
        c="blue",
        label="Actual Data (Class 1)",
    )
    plt.scatter(
        x_train[:, 8], y_predict, c="green", alpha=0.2, label="Predicted Probability"
    )
    plt.axhline(
        y=0.5, color="green", linestyle="--", label="Decision Boundary (0.5 Prob)"
    )
    plt.title(f"Health Risk vs. Married ({label})", fontsize=24)
    plt.xlabel("Married", fontsize=18)
    plt.ylabel("Health Risk", fontsize=18)
    plt.grid(True)
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
    plt.ticklabel_format(style="plain", axis="x")
    plt.savefig(f"outputs/plots/health_risk_vs_married_{files_postfix}_plot.png")

    # --- health_risk vs profession_health_risk_mean
    plt.figure(figsize=(40, 10))
    plt.scatter(
        x_train[class_0, 9],
        y_train[class_0],
        s=75,
        c="red",
        label="Actual Data (Class 0)",
    )
    plt.scatter(
        x_train[class_1, 9],
        y_train[class_1],
        s=75,
        c="blue",
        label="Actual Data (Class 1)",
    )
    plt.scatter(
        x_train[:, 9], y_predict, c="green", alpha=0.2, label="Predicted Probability"
    )
    plt.axhline(
        y=0.5, color="green", linestyle="--", label="Decision Boundary (0.5 Prob)"
    )
    plt.title(f"Health Risk vs. Profession Health Risk Mean ({label})", fontsize=24)
    plt.xlabel("Profession Health Risk Mean", fontsize=18)
    plt.ylabel("Health Risk", fontsize=18)
    plt.grid(True)
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
    plt.ticklabel_format(style="plain", axis="x")
    plt.savefig(
        f"outputs/plots/health_risk_vs_profession_health_risk_mean_{files_postfix}_plot.png"
    )

    # --- health_risk vs bmi
    plt.figure(figsize=(40, 10))
    plt.scatter(
        x_train[class_0, 10],
        y_train[class_0],
        s=75,
        c="red",
        label="Actual Data (Class 0)",
    )
    plt.scatter(
        x_train[class_1, 10],
        y_train[class_1],
        s=75,
        c="blue",
        label="Actual Data (Class 1)",
    )
    plt.scatter(
        x_train[:, 10], y_predict, c="green", alpha=0.2, label="Predicted Probability"
    )
    plt.axhline(
        y=0.5, color="green", linestyle="--", label="Decision Boundary (0.5 Prob)"
    )
    plt.title(f"Health Risk vs. BMI ({label})", fontsize=24)
    plt.xlabel("BMI", fontsize=18)
    plt.ylabel("Health Risk", fontsize=18)
    plt.grid(True)
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
    plt.ticklabel_format(style="plain", axis="x")
    plt.savefig(f"outputs/plots/health_risk_vs_bmi_{files_postfix}_plot.png")
