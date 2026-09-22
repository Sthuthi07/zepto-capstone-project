import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)
CSV = os.path.join(BASE, "titanic.csv")

# Raw dataset is loaded from Seaborn exactly once.
df = sns.load_dataset("titanic")
df.to_csv(CSV, index=False)

print("INFO")
df.info()
print("\nDESCRIBE\n", df.describe(include="all"))
print("\nSHAPE:", df.shape)

missing = df.isna().mean().mul(100)
missing = missing[missing > 0]
print("\nMISSING PERCENTAGES\n", missing)

# Threshold-based cleaning.
# Threshold-based cleaning.
clean = df.copy()
for col in missing.index:
    rate = missing[col]
    if rate < 5:
        clean = clean.dropna(subset=[col])
    elif rate <= 30:
        if pd.api.types.is_numeric_dtype(clean[col]):
            clean[col] = clean[col].fillna(clean[col].median())
        else:
            clean[col] = clean[col].fillna(clean[col].mode()[0])
    else:
        # Retain highly-missing categorical columns using an explicit Missing label.
        if pd.api.types.is_categorical_dtype(clean[col]):
            clean[col] = clean[col].cat.add_categories(["Missing"])
        clean[col] = clean[col].fillna("Missing")
# Convert relevant categoricals to strings for robust downstream processing.
for col in ["sex", "embarked", "class", "who", "deck", "embark_town"]:
    if col in clean.columns:
        clean[col] = clean[col].astype(str)

def iqr_count(s):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())

age_outliers = iqr_count(clean["age"])
fare_outliers = iqr_count(clean["fare"])
fare_mean = clean["fare"].mean()
fare_median = clean["fare"].median()
fare_mode = clean["fare"].mode().iloc[0]
skew = "right-skewed" if fare_mean > fare_median > fare_mode else (
    "left-skewed" if fare_mean < fare_median < fare_mode else "not strictly classified by ordering"
)

print("\nOUTLIERS: age =", age_outliers, ", fare =", fare_outliers)
print("FARE mean/median/mode:", fare_mean, fare_median, fare_mode)
print("FARE distribution:", skew)

survival_sex = clean.groupby("sex")["survived"].mean()
survival_pclass = clean.groupby("pclass")["survived"].mean()
survival_both = clean.groupby(["sex", "pclass"])["survived"].mean()
print("\nSURVIVAL BY SEX\n", survival_sex)
print("\nSURVIVAL BY PCLASS\n", survival_pclass)
print("\nSURVIVAL BY SEX + PCLASS\n", survival_both)

corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr = clean[corr_cols].corr()
print("\nCORRELATION MATRIX\n", corr)

pairs = []
for i, a in enumerate(corr_cols):
    for b in corr_cols[i+1:]:
        pairs.append((abs(corr.loc[a, b]), a, b, corr.loc[a, b]))
top2 = sorted(pairs, reverse=True)[:2]
print("\nTWO STRONGEST CORRELATIONS\n", top2)

# Four+ distinct charts with interpretations in this file and analytics/README.md.
plt.figure(figsize=(8, 5))
sns.histplot(clean["age"], kde=True)
plt.title("Age Distribution")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "01_age_histogram.png"))
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(x=clean["age"])
plt.title("Age Box Plot")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "02_age_boxplot.png"))
plt.close()

plt.figure(figsize=(8, 5))
sns.histplot(clean["fare"], kde=True)
plt.title("Fare Distribution")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "03_fare_histogram.png"))
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(x=clean["fare"])
plt.title("Fare Box Plot")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "04_fare_boxplot.png"))
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(data=clean, x="sex", y="survived")
plt.title("Survival Rate by Sex")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "05_survival_sex.png"))
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(data=clean, x="pclass", y="survived")
plt.title("Survival Rate by Passenger Class")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "06_survival_pclass.png"))
plt.close()

plt.figure(figsize=(9, 6))
sns.boxplot(data=clean, x="pclass", y="fare", hue="survived")
plt.title("Fare by Class and Survival")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "07_fare_class_survival.png"))
plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f")
plt.title("Six-Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "08_correlation_heatmap.png"))
plt.close()

# Exploratory full-cleaned-data standardization only; not used for modeling.
scaler = StandardScaler()
z = scaler.fit_transform(clean[["age", "fare"]])
zdf = pd.DataFrame(z, columns=["age_z", "fare_z"])
print("\nSTANDARDIZATION BEFORE")
print(clean[["age", "fare"]].agg(["mean", "std"]))
print("\nSTANDARDIZATION AFTER")
print(zdf.agg(["mean", "std"]))

clean.to_csv(os.path.join(OUT, "clean_titanic.csv"), index=False)
with open(os.path.join(OUT, "eda_report.txt"), "w", encoding="utf-8") as f:
    f.write(f"Missing percentages:\n{missing.to_string()}\n\n")
    f.write(f"Age IQR outliers: {age_outliers}\nFare IQR outliers: {fare_outliers}\n")
    f.write(f"Fare mean={fare_mean:.4f}, median={fare_median:.4f}, mode={fare_mode:.4f}\n")
    f.write(f"Fare ordering interpretation: {skew}\n\n")
    f.write(f"Survival by sex:\n{survival_sex.to_string()}\n\n")
    f.write(f"Survival by pclass:\n{survival_pclass.to_string()}\n\n")
    f.write(f"Survival by sex and pclass:\n{survival_both.to_string()}\n\n")
    f.write(f"Correlation:\n{corr.to_string()}\n\n")
    f.write(f"Two strongest absolute correlations: {top2}\n")
    f.write("\nChart interpretations:\n")
    f.write("1. Age histogram: Age is concentrated among young and middle-aged passengers, with fewer observations at the extremes.\n")
    f.write("2. Fare histogram: Fare is strongly concentrated at lower values with a long upper tail, consistent with the mean exceeding the median.\n")
    f.write("3. Survival by sex: Survival rates differ substantially by sex, showing sex is informative for the target.\n")
    f.write("4. Survival by class: Passenger class is associated with survival, indicating socioeconomic/class position relates to outcome.\n")
    f.write("5. Fare by class and survival: Fare varies strongly by class, while survivors tend to appear more frequently among higher-fare observations within classes.\n")
    f.write("6. Correlation heatmap: The matrix shows relationships among the six required numeric features; the two strongest pairs are listed above.\n")
