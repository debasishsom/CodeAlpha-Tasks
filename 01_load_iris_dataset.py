# ============================================================
# CodeAlpha Internship - Task 1
# Iris Flower Classification
# File: 01_load_iris_dataset.py
# ============================================================

import pandas as pd
from sklearn.datasets import load_iris


# ------------------------------------------------------------
# 1. Load Iris Dataset
# ------------------------------------------------------------

iris = load_iris()


# ------------------------------------------------------------
# 2. Create DataFrame
# ------------------------------------------------------------

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)


# ------------------------------------------------------------
# 3. Add Target Column
# ------------------------------------------------------------

df["target"] = iris.target


# ------------------------------------------------------------
# 4. Add Flower Species Name
# ------------------------------------------------------------

df["species"] = df["target"].apply(
    lambda x: iris.target_names[x]
)


# ------------------------------------------------------------
# 5. Display Dataset Information
# ------------------------------------------------------------

print("=" * 60)
print("IRIS FLOWER DATASET")
print("=" * 60)

print("\nFirst 10 rows:")
print(df.head(10))


print("\nDataset Shape:")
print(df.shape)


print("\nColumn Names:")
print(df.columns.tolist())


print("\nDataset Information:")
print(df.info())


print("\nStatistical Summary:")
print(df.describe())


print("\nSpecies Distribution:")
print(df["species"].value_counts())


# ------------------------------------------------------------
# 6. Check Missing Values
# ------------------------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())


# ------------------------------------------------------------
# 7. Save Dataset
# ------------------------------------------------------------

df.to_csv("iris_dataset.csv", index=False)

print("\nIris dataset saved as: iris_dataset.csv")

print("\nDataset loading completed successfully!")