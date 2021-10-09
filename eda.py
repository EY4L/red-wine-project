# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("dark")
sns.set_style("ticks")

# %% [markdown]
"""
# Read Data
- Use `dtype` for performance
- Rename `quality` to the target `label`
- Rename columns with spaces
"""

# %%  Read Dataset
df = pd.read_csv(
    Path("data", "winequality-red.csv",),
    dtype={
        "fixed acidity": float,
        "volatile acidity": float,
        "citric acid": float,
        "residual sugar": float,
        "chlorides": float,
        "free sulfur dioxide": float,
        "total sulfur dioxide": float,
        "density": float,
        "pH": float,
        "sulphates": float,
        "alcohol": float,
        "quality": int,
    },
).rename(
    columns={
        "fixed acidity": "fixed_acidity",
        "volatile acidity": "volatile_acidity",
        "citric acid": "citric_acid",
        "residual sugar": "residual_sugar",
        "free sulfur dioxide": "free_sulfur_dioxide",
        "total sulfur dioxide": "total_sulfur_dioxide",
        "quality": "label",
    }
)

df.head()
# %%
df.info()

#%%
df.describe()


# %% Save figure
def save_eda_fig(save_as):
    plt.savefig(
        Path("eda", save_as,), orientation="portrait", format="png",
    )


# %% [markdown]
"""
### Data preperation
- Check labels distribution. Labels are very imbalanced
- Transform labels into binary classification. Labels are balanced
- Check for duplicates, plot duplicates per label, and drop duplicated rows
- Check for missing values. No missing values for any features
- Save processed data for training
"""

# %% Class Distribution
fig, ax = plt.subplots()
df.groupby("label").count()["density"].plot(kind="bar")
ax.set_xlabel("Label")
ax.set_ylabel("No. of examples")
ax.set_title("Class Distribution")
save_eda_fig("labels_distribution.png")

# %%
# Changing labels
df["label"] = df["label"].map({3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1,})

# %% Duplicate removal
df_duplicated = df[df.duplicated(keep=False)]

fig, ax = plt.subplots()
df_duplicated.groupby("label").count()["density"].plot(kind="bar")
ax.set_xlabel("Label")
ax.set_ylabel("No. of duplicates")
save_eda_fig("duplicates_per_label.png")

df = df.drop_duplicates()
# %% Class Distribution after changing labels

fig, ax = plt.subplots()
df.groupby("label").count()["density"].plot(kind="bar")
ax.set_xlabel("Label")
ax.set_ylabel("No. of examples")
ax.set_title("Class Distribution")
save_eda_fig("labels_distribution.png")
# %% Check missing values
df.isna().sum()

# %%
# Save for training
df.to_csv(Path("data", "winequality-red-processed.csv",))


# %% [markdown]
"""
# Features visualization
- Plot boxplots for all features on a log scale.
- Plot scatter matrix to check for features distribution and correlation
- Plot boxplots and scatter graphs to inspect correlated variables
- Regression to examine further correlation
"""

# %% Feature Boxplot Distribution
fig, ax = plt.subplots()
df.boxplot(sym="r.", notch=True)
ax.set_yscale("log")
ax.tick_params(axis="x", labelrotation=90)
ax.set_title("Features Boxplots")
save_eda_fig("features_boxplots.png")

# %% Feature Correlation
fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)
ax.set_title("Features Correlation")
save_eda_fig("features_scatter_matrix.png")

# %% Selected features correlated with Quality

fig, axes = plt.subplots(1, 4, sharex=True, figsize=(22, 10))
sns.boxplot(
    ax=axes[0], y=df["alcohol"], x=df["label"],
)
axes[0].set(ylim=(8, 14.2), xlabel="Quality")

sns.boxplot(ax=axes[1], y=df["sulphates"], x=df["label"])
axes[1].set(ylim=(0.3, 1.15), xlabel="Quality")

sns.boxplot(ax=axes[2], y=df["citric_acid"], x=df["label"])
axes[2].set(ylim=(0, 0.8), xlabel="Quality")

sns.boxplot(ax=axes[3], y=df["volatile_acidity"], x=df["label"])
axes[3].set(ylim=(0.1, 1.25), xlabel="Quality")
save_eda_fig("Features_correlation_quality.png")

# %% Selected Features correlated with other features over Quality

fig, axes = plt.subplots(1, 4, figsize=(25, 10))
sns.scatterplot(
    ax=axes[0],
    y=df["pH"],
    x=df["citric_acid"],
    hue=df["label"],
    # palette=sns.color_palette("coolwarm", as_cmap=True),
    alpha=0.8,
)

sns.scatterplot(
    ax=axes[1],
    y=df["citric_acid"],
    x=df["fixed_acidity"],
    hue=df["label"],
    #  palette=sns.color_palette("coolwarm", as_cmap=True),
    alpha=0.82,
)
axes[1].set(ylim=(0, 0.8))

sns.scatterplot(
    ax=axes[2],
    y=df["pH"],
    x=df["fixed_acidity"],
    hue=df["label"],
    #  palette=sns.color_palette("coolwarm", as_cmap=True),
    alpha=0.84,
)
axes[2].set(xlim=(4.5, 14), ylim=(2.9, 3.8))

sns.scatterplot(
    ax=axes[3],
    y=df["free_sulfur_dioxide"],
    x=df["total_sulfur_dioxide"],
    hue=df["label"],
    # palette=sns.color_palette("coolwarm", as_cmap=True),
    alpha=0.78,
)
axes[3].set(xlim=(0, 150), ylim=(0, 50))
save_eda_fig("Features_correlation_features.png")


# %% Regression plot between total sulfur and free sulfur

sns.lmplot(
    y="total_sulfur_dioxide", x="free_sulfur_dioxide", hue="label", data=df,
)
ax.set(xlim=(0, 70), ylim=(0, 150))
save_eda_fig("Reg_plot.png")

# %%
