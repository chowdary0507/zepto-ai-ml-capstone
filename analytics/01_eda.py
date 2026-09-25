from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
CHART_DIR = BASE_DIR / "charts"
CHART_DIR.mkdir(exist_ok=True)

# Load the source data and keep a local copy for the next script.
df = sns.load_dataset("titanic")
df.to_csv(BASE_DIR / "titanic.csv", index=False)
print("Raw data shape:", df.shape)
print(df.info())
print(df.describe())

df = pd.read_csv(BASE_DIR / "titanic.csv")
print("\nMissing value percentages:")
print(df.isna().mean().mul(100).sort_values(ascending=False))

# `deck` is mostly missing, so it is left out. The smaller gaps are filled below.

# The two rows with missing embarked values are below the 5% threshold, so drop them.
df_clean = df.dropna(subset=["embarked", "embark_town"]).drop(columns=["deck"]).copy()
df_clean["age"] = df_clean["age"].fillna(df_clean["age"].median())

print("\nAge missing count after fill:", df_clean["age"].isna().sum())
print("Rows after dropping small missing-value gaps:", len(df_clean))

# Count outliers with the IQR rule.
for col in ["age", "fare"]:
    q1 = df_clean[col].quantile(0.25)
    q3 = df_clean[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = int(((df_clean[col] < lower) | (df_clean[col] > upper)).sum())
    print(f"{col} outliers using IQR rule: {count}")

fare_mean = df_clean["fare"].mean()
fare_median = df_clean["fare"].median()
fare_mode = df_clean["fare"].mode().iloc[0]
print(f"Fare mean={fare_mean:.3f}, median={fare_median:.3f}, mode={fare_mode:.3f}")
print("Fare distribution is right-skewed because mean > median > mode.")

# Bivariate survival rates, calculated with explicit boolean masks.
sex_masks = {sex: df_clean["sex"] == sex for sex in ["female", "male"]}
pclass_masks = {pclass: df_clean["pclass"] == pclass for pclass in [1, 2, 3]}
survival_by_sex = {sex: df_clean.loc[mask, "survived"].mean() for sex, mask in sex_masks.items()}
survival_by_pclass = {pclass: df_clean.loc[mask, "survived"].mean() for pclass, mask in pclass_masks.items()}
survival_by_sex_pclass = {}
for sex, sex_mask in sex_masks.items():
    for pclass, pclass_mask in pclass_masks.items():
        combined_mask = sex_mask & pclass_mask
        survival_by_sex_pclass[(sex, pclass)] = df_clean.loc[combined_mask, "survived"].mean()
print("\nSurvival by sex:")
print(pd.Series(survival_by_sex))
print("\nSurvival by pclass:")
print(pd.Series(survival_by_pclass))
print("\nSurvival by sex and pclass:")
print(pd.Series(survival_by_sex_pclass))

# Compare the main numeric columns.
correlation_columns = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
correlation_matrix = df_clean[correlation_columns].corr()
print("\nCorrelation matrix:")
print(correlation_matrix)

corr_abs = correlation_matrix.where(~np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)).stack().abs()
# Ignore the diagonal and sort the remaining pairs.
strongest_pairs = corr_abs[corr_abs.index.get_level_values(0) != corr_abs.index.get_level_values(1)].sort_values(ascending=False)
print("\nTop two absolute off-diagonal correlations:")
print(strongest_pairs.head(2))

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Titanic correlation heatmap")
plt.tight_layout()
plt.savefig(CHART_DIR / "corr_heatmap.png", dpi=200)
plt.close()

# Single-variable plots.
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.histplot(df_clean["age"], bins=25, kde=True, ax=axes[0, 0])
axes[0, 0].set_title("Age distribution")

sns.boxplot(x=df_clean["age"], ax=axes[0, 1])
axes[0, 1].set_title("Age boxplot")

sns.histplot(df_clean["fare"], bins=30, kde=True, ax=axes[1, 0])
axes[1, 0].set_title("Fare distribution")

sns.boxplot(x=df_clean["fare"], ax=axes[1, 1])
axes[1, 1].set_title("Fare boxplot")

plt.tight_layout()
plt.savefig(CHART_DIR / "age_fare_distribution.png", dpi=200)
plt.close()

# Save the two histograms separately for easy review.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df_clean["age"], bins=25, kde=True, ax=axes[0])
axes[0].set_title("Age histogram")
sns.histplot(df_clean["fare"], bins=30, kde=True, ax=axes[1])
axes[1].set_title("Fare histogram")
plt.tight_layout()
plt.savefig(BASE_DIR / "histograms.png", dpi=200)
plt.close()

# Save the boxplots separately so they are easy to find for review.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(y=df_clean["age"], ax=axes[0])
axes[0].set_title("Age boxplot")
sns.boxplot(y=df_clean["fare"], ax=axes[1])
axes[1].set_title("Fare boxplot")
plt.tight_layout()
plt.savefig(BASE_DIR / "boxplots.png", dpi=200)
plt.close()

# A few comparisons between variables.
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.barplot(data=df_clean, x="sex", y="survived", estimator="mean", ax=axes[0, 0])
axes[0, 0].set_title("Survival rate by sex")

sns.barplot(data=df_clean, x="pclass", y="survived", estimator="mean", ax=axes[0, 1])
axes[0, 1].set_title("Survival rate by passenger class")

sns.boxplot(data=df_clean, x="sex", y="age", ax=axes[1, 0])
axes[1, 0].set_title("Age by sex")

sns.scatterplot(data=df_clean, x="fare", y="age", hue="survived", alpha=0.7, ax=axes[1, 1])
axes[1, 1].set_title("Fare vs age by survival")

plt.tight_layout()
plt.savefig(CHART_DIR / "multivariate_story.png", dpi=200)
plt.close()

# Save the fare-age relationship as its own scatterplot.
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_clean, x="fare", y="age", hue="survived", alpha=0.7)
plt.title("Fare versus age by survival")
plt.tight_layout()
plt.savefig(BASE_DIR / "scatterplot.png", dpi=200)
plt.close()

# Check the standardized age and fare values.
for col in ["age", "fare"]:
    z_col = (df_clean[col] - df_clean[col].mean()) / df_clean[col].std(ddof=0)
    df_clean[f"{col}_zscore"] = z_col
    print(f"{col} z-score mean={z_col.mean():.6f}, std={z_col.std(ddof=0):.6f}")

# Save the cleaned data for modeling.
df_clean.to_csv(BASE_DIR / "clean_titanic.csv", index=False)

print("\nEDA complete. Charts saved to:", CHART_DIR)