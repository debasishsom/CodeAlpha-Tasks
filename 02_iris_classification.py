# ============================================================
# CodeAlpha Internship - Task 1
# Iris Flower Classification
# File: 02_iris_classification.py
# ============================================================

# ------------------------------------------------------------
# Import Required Libraries
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib


# ------------------------------------------------------------
# 1. Load Iris Dataset
# ------------------------------------------------------------

print("=" * 70)
print("IRIS FLOWER CLASSIFICATION - CODEALPHA TASK 1")
print("=" * 70)

iris = load_iris()

X = iris.data
y = iris.target


# ------------------------------------------------------------
# 2. Create DataFrame
# ------------------------------------------------------------

df = pd.DataFrame(
    X,
    columns=iris.feature_names
)

df["target"] = y

df["species"] = df["target"].apply(
    lambda x: iris.target_names[x]
)


print("\nDataset Preview:")
print(df.head())


# ------------------------------------------------------------
# 3. Dataset Information
# ------------------------------------------------------------

print("\nDataset Shape:")
print(df.shape)

print("\nSpecies:")
print(iris.target_names)

print("\nSpecies Distribution:")
print(df["species"].value_counts())


# ------------------------------------------------------------
# 4. Check Missing Values
# ------------------------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())


# ------------------------------------------------------------
# 5. Separate Features and Target
# ------------------------------------------------------------

X = df[iris.feature_names]
y = df["target"]


# ------------------------------------------------------------
# 6. Split Dataset into Training and Testing Data
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining Data Size:")
print(X_train.shape)

print("\nTesting Data Size:")
print(X_test.shape)


# ------------------------------------------------------------
# 7. Feature Scaling
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ------------------------------------------------------------
# 8. Create Machine Learning Model
# ------------------------------------------------------------

model = LogisticRegression(
    max_iter=200
)


# ------------------------------------------------------------
# 9. Train the Model
# ------------------------------------------------------------

print("\nTraining the Logistic Regression model...")

model.fit(
    X_train_scaled,
    y_train
)

print("Model training completed successfully!")


# ------------------------------------------------------------
# 10. Make Predictions
# ------------------------------------------------------------

y_pred = model.predict(X_test_scaled)


# ------------------------------------------------------------
# 11. Calculate Accuracy
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


# ------------------------------------------------------------
# 12. Classification Report
# ------------------------------------------------------------

print("\nClassification Report:")

report = classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
)

print(report)


# ------------------------------------------------------------
# 13. Confusion Matrix
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ------------------------------------------------------------
# 14. Visualize Confusion Matrix
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.title("Iris Flower Classification - Confusion Matrix")
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 15. Test Individual Flower Prediction
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SAMPLE FLOWER PREDICTION")
print("=" * 70)

sample_flower = np.array([
    [5.1, 3.5, 1.4, 0.2]
])


sample_scaled = scaler.transform(
    sample_flower
)

prediction = model.predict(
    sample_scaled
)

predicted_species = iris.target_names[
    prediction[0]
]

print("\nInput Measurements:")
print("Sepal Length : 5.1 cm")
print("Sepal Width  : 3.5 cm")
print("Petal Length : 1.4 cm")
print("Petal Width  : 0.2 cm")

print(
    f"\nPredicted Flower Species: {predicted_species}"
)


# ------------------------------------------------------------
# 16. Save Trained Model
# ------------------------------------------------------------

joblib.dump(
    model,
    "iris_model.pkl"
)

joblib.dump(
    scaler,
    "iris_scaler.pkl"
)

print("\nTrained model saved as: iris_model.pkl")
print("Scaler saved as: iris_scaler.pkl")


# ------------------------------------------------------------
# 17. Save Results to Text File
# ------------------------------------------------------------

with open(
    "iris_results.txt",
    "w"
) as file:

    file.write(
        "CODEALPHA INTERNSHIP - TASK 1\n"
    )

    file.write(
        "IRIS FLOWER CLASSIFICATION\n"
    )

    file.write(
        "=" * 50 + "\n\n"
    )

    file.write(
        f"Model: Logistic Regression\n"
    )

    file.write(
        f"Training Samples: {len(X_train)}\n"
    )

    file.write(
        f"Testing Samples: {len(X_test)}\n"
    )

    file.write(
        f"Accuracy: {accuracy * 100:.2f}%\n\n"
    )

    file.write(
        "Classification Report:\n"
    )

    file.write(
        report
    )

    file.write(
        "\nConfusion Matrix:\n"
    )

    file.write(
        str(cm)
    )

    file.write(
        "\n\nSample Prediction:\n"
    )

    file.write(
        f"Predicted Species: {predicted_species}\n"
    )


# ------------------------------------------------------------
# 18. Final Message
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TASK 1 COMPLETED SUCCESSFULLY!")
print("=" * 70)

print("\nGenerated Files:")
print("1. iris_dataset.csv")
print("2. iris_model.pkl")
print("3. iris_scaler.pkl")
print("4. confusion_matrix.png")
print("5. iris_results.txt")

print("\nYou can now use these results for your CodeAlpha submission.")