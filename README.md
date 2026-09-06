# ==========================================
# TASK 5: EXPLORATORY DATA ANALYSIS (EDA)
# Dataset: Titanic Dataset
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Environment Setup & Data Loading
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.size'] = 10

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("=" * 50)
print("1. DATASET OVERVIEW & STATISTICAL SUMMARY")
print("=" * 50)

# Dataset Structure & Info
print("\n--- Dataset Info ---")
df.info()

# Missing Values Count
print("\n--- Missing Values Count ---")
print(df.isnull().sum())

# Descriptive Statistics
print("\n--- Numerical Summary ---")
print(df.describe().round(2))

# Target & Categorical Value Counts
print("\n--- Survival Distribution (%) ---")
print(df['Survived'].value_counts(normalize=True).round(3) * 100)

print("\n--- Passenger Class Distribution (%) ---")
print(df['Pclass'].value_counts(normalize=True).round(3) * 100)


# 2. Univariate Analysis (Distribution & Outliers)
print("\n" + "=" * 50)
print("2. GENERATING UNIVARIATE PLOTS...")
print("=" * 50)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Age Distribution
sns.histplot(df['Age'].dropna(), kde=True, ax=axes[0], color='teal', bins=30)
axes[0].set_title('Age Distribution of Passengers')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Count')

# Plot 2: Fare Outliers Boxplot
sns.boxplot(x=df['Fare'], ax=axes[1], color='coral')
axes[1].set_title('Fare Distribution & Outlier Detection')
axes[1].set_xlabel('Fare ($)')

plt.tight_layout()
plt.show()


# 3. Bivariate Analysis (Survival Drivers)
print("\n" + "=" * 50)
print("3. GENERATING BIVARIATE PLOTS...")
print("=" * 50)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 3: Survival Rate by Gender
sns.barplot(x='Sex', y='Survived', data=df, ax=axes[0], palette='Set2', errorbar=None)
axes[0].set_title('Survival Probability by Gender')
axes[0].set_ylabel('Survival Rate')

# Plot 4: Survival Rate by Passenger Class
sns.barplot(x='Pclass', y='Survived', data=df, ax=axes[1], palette='Set1', errorbar=None)
axes[1].set_title('Survival Probability by Passenger Class')
axes[1].set_ylabel('Survival Rate')

plt.tight_layout()
plt.show()


# 4. Multivariate Analysis (Heatmap & Pairplot)
print("\n" + "=" * 50)
print("4. GENERATING MULTIVARIATE PLOTS...")
print("=" * 50)

# Plot 5: Correlation Matrix
plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Feature Correlation Matrix Heatmap')
plt.show()

# Plot 6: Feature Pairplot
sns.pairplot(df[['Survived', 'Pclass', 'Age', 'Fare']].dropna(), hue='Survived', palette='husl')
plt.show()


# 5. Summary of Key Findings
print("\n" + "=" * 50)
print("5. SUMMARY OF KEY FINDINGS")
print("=" * 50)
print("""
* Gender Impact: Females had a significantly higher survival rate (~74%) compared to males (~19%).
* Socioeconomic Status: 1st Class passengers had the highest survival probability (>60%), while 3rd Class passengers were least likely to survive (<25%).
* Age Factor: Children had better survival odds across classes, consistent with standard evacuation protocols.
* Outliers & Missing Data: `Age` contains missing values needing median imputation; `Fare` presents high right-skewness with severe upper outliers.
""")
