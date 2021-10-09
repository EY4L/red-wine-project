# %%
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics, set_config
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.tree import DecisionTreeClassifier

set_config(display="diagram")

# %% [markdown]
"""
# Read Data
"""
# %%
df = pd.read_csv(
    Path("data", "winequality-red-processed.csv",),
    index_col=0,
    dtype={
        "fixed_acidity": float,
        "volatile_acidity": float,
        "citric_acid": float,
        "chlorides": float,
        "free_sulfur_dioxide": float,
        "total_sulfur_dioxide": float,
        "density": float,
        "pH": float,
        "sulphates": float,
        "alcohol": float,
        "label": "category",
    },
)


df.head()

# %% [markdown]
"""
# Split data
- Use `label` as target labels
    - Number of classes is $2$
    - **Classes are balanced**
    - Stratify labels when splitting so their distribution in train/test data is similar
- Split $75/25$ for training/testing
- Create empty dict to store all classifiers. Once populated, an item will look like:
`'classifier name' : {
    'pipeline': ...,
    'params': ...,
    'best_score': ...,
    'best_params': ...,
    'best_estimator': ...,
    'best_estimator_params': ...,
    'testing_accuracy': ...,
    'testing_conf_matrix': ...,
}`
"""

# %%
y = df["label"]
num_class = len(y.unique())

le = LabelEncoder()
y = le.fit_transform(y)
le.classes_

joblib.dump(
    le, Path("models", "le.pkl",),
)

X = df.drop(["label"], axis=1)
num_features = X.columns.tolist()

if not Path("models").is_dir():
    Path("models").mkdir()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.75, shuffle=True, stratify=y, random_state=42,
)

classifiers = dict()

# %% [markdown]
"""
# Transformers
1. A pipeline for numerical features: `num`. This will have 2 step:
    - `KNNImputer` to fill-in missing values with the mean of neighboring samples (in case we have missing data in the future)
    - `RobustScaler` to scale features using statistics that are robust to outliers

- `remainder=drop` will be used to drop any extra features that might be added to the dataframe later as a safety guard. When adding new features, either pass them through a pipeline, or change to `remainder=passthrough`
"""

# %%
transformer = Pipeline(
    steps=[("imputer", KNNImputer(n_neighbors=5)), ("scaler", RobustScaler()),]
)

preprocessor = ColumnTransformer(
    transformers=[("num", transformer, num_features)], remainder="drop",
)

# %% [markdown]
"""
# Classifiers
"""

# %% [markdown]
"""
## Decision Tree

- Can be interpreted as a piecewise constand approximation
- Learns the value of target variable through decision rules generated from the features in the dataset
- Advantages
    - Requires little data and cost to use
    - Suited for numerical and categorical data and able to handle multi-output
- Disadvantages
    - Can easily overfit data
    - Can create biased trees
    - Can be unstable
"""

# %%
classifier = DecisionTreeClassifier(random_state=42,)

pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier),])

params = {
    "classifier__criterion": ["gini", "entropy"],
    "classifier__max_depth": [5, 6, 7],
    "classifier__max_features": ["sqrt", "log2"],
    "classifier__splitter": ["best", "random"],
}

classifiers["DecisionTreeClassifier"] = {
    "pipeline": pipeline,
    "params": params,
}

# %% [markdown]
"""
## Random Forest Ensemble

- Each tree is trained on a random subset of the training set with replacement (i.e. bootstrap aggregating or bagging)
- Each tree is trained on a random subset of features, the number of features to use is defined by `max_features`
- Advantages
    - No need for scaling
    - No need for dimensionality reduction (unless rotation is needed)
- Disadvantages
    - Sensitive to small variations in the training data
    - Over-fitting
    - Difficult to interpret
"""

# %%
classifier = RandomForestClassifier(
    bootstrap=True, oob_score=True, random_state=42, n_jobs=-1,
)

pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier),])

params = {
    "classifier__criterion": ["gini", "entropy"],
    "classifier__n_estimators": [200, 300, 400],
    "classifier__max_depth": [8, 9, 10],
    "classifier__max_features": ["sqrt", "log2"],
}

classifiers["RandomForestClassifier"] = {
    "pipeline": pipeline,
    "params": params,
}

# %% [markdown]
"""
## Adaptive Boosting
- AdaBoost uses decision tress with a single split and puts more weight on datapoints which are harder to classify and less on those handeled well
- By default uses 'SAAME.R' algorithm which converges faster than SAMME and outputs class probabilites
- Advantages
    - Flexible as can be used to improve weak classifiers
- Disadvantages
    - Data needs to be of high quality
    - Sensitive to Noisy data and outliers
    - Slow
"""

# %%
classifier = AdaBoostClassifier(random_state=42,)

pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier),])

params = {
    "classifier__algorithm": ["SAMME", "SAMME.R"],
    "classifier__n_estimators": [200, 300, 400],
    "classifier__learning_rate": [0.01, 0.1, 1],
}

classifiers["AdaBoostClassifier"] = {
    "pipeline": pipeline,
    "params": params,
}

# %% [markdown]
"""
# Training

## Grid search cross validation

- Use `accuracy` for scoring as binary classification
- Loop over each classifier in the classifiers dictionary
    - Train using training data
    - Pickle trained model
"""

# %%
for k, v in classifiers.items():

    print(f"\nRunning grid search with cross validation for {k}...")

    gs = GridSearchCV(
        v["pipeline"], v["params"], scoring="accuracy", cv=5, n_jobs=-1, verbose=2,
    )

    gs.fit(
        X_train, y_train,
    )

    joblib.dump(
        gs, Path("models", f"{k}.pkl",),
    )

# %% [markdown]
"""
# Validation

- Loop over each classifier in the classifiers dictionary
    - Load model
        - Store the grid search `best_score_`
        - Store the grid search `best_params_`
        - Store the grid search `best_estimator_`
        - Store the grid search `best_estimator_` params
    - Evalute using testing data
        - Store the `accuracy_score`
        - Store class-wise
"""

# %%
for k, v in classifiers.items():

    gs = joblib.load(Path("models", f"{k}.pkl",),)

    v["best_score"] = gs.best_score_
    v["best_params"] = gs.best_params_
    v["best_estimator"] = gs.best_estimator_
    v["best_estimator_params"] = gs.best_estimator_.named_steps[
        "classifier"
    ].get_params()

    print(f"Running evaluation on test data for {k}...")
    y_pred = gs.predict(X_test)

    v["testing_accuracy"] = accuracy_score(y_test, y_pred,)

    v["testing_conf_matrix"] = confusion_matrix(y_test, y_pred,)

# %% [markdown]
"""
- Create DataFrame of results from all classifiers and save into cvs
- Load LinearSVC and get features importance using the models `coef_` 
- Get features importance for RandomForestClassifier
"""

# %%
pd.DataFrame.from_dict(classifiers, orient="index",).to_csv(
    Path("models", "models.csv",)
)

# %%
gs = joblib.load(Path("models", "RandomForestClassifier.pkl",))

metrics.plot_roc_curve(gs, X_test, y_test)
plt.show()

importance = gs.best_estimator_.named_steps["classifier"].feature_importances_
features_importance = sorted(zip(importance, num_features), reverse=True)
print("RForest Feature_importances:", *features_importance, sep="\n")

# %%
gs = joblib.load(Path("models", "AdaBoostClassifier.pkl",))

metrics.plot_roc_curve(gs, X_test, y_test)
plt.show()


importance = gs.best_estimator_.named_steps["classifier"].feature_importances_
features_importance = sorted(zip(importance, num_features), reverse=True)
print("AdaBoost Feature_importances:", *features_importance, sep="\n")

# %%
gs = joblib.load(Path("models", "DecisionTreeClassifier.pkl",))

metrics.plot_roc_curve(gs, X_test, y_test)
plt.show()

importance = gs.best_estimator_.named_steps["classifier"].feature_importances_
features_importance = sorted(zip(importance, num_features), reverse=True)
print("Decision Tree Feature_importances:", *features_importance, sep="\n")


# %%
