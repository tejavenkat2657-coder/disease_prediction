"""
train_model.py
--------------
Preprocesses data, trains 3 ML models, saves the best one with Pickle.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data/disease_dataset.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")

# ── 2. Preprocess ─────────────────────────────────────────────────────────────
symptom_cols = [c for c in df.columns if c not in ("age_group", "gender", "disease")]
df[symptom_cols] = df[symptom_cols].fillna(0).astype(int)
df["age_group"]  = df["age_group"].fillna(df["age_group"].mode()[0])
df["gender"]     = df["gender"].fillna(df["gender"].mode()[0])

# ── 3. Encode ─────────────────────────────────────────────────────────────────
age_encoder    = LabelEncoder()
gender_encoder = LabelEncoder()
label_encoder  = LabelEncoder()

df["age_group_enc"] = age_encoder.fit_transform(df["age_group"])
df["gender_enc"]    = gender_encoder.fit_transform(df["gender"])
df["label"]         = label_encoder.fit_transform(df["disease"])

print(f"Classes → Age: {list(age_encoder.classes_)}  "
      f"Gender: {list(gender_encoder.classes_)}  "
      f"Disease: {list(label_encoder.classes_)}")

# ── 4. Split ──────────────────────────────────────────────────────────────────
FEATURE_COLS = symptom_cols + ["age_group_enc", "gender_enc"]
X = df[FEATURE_COLS].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

# ── 5. Train ──────────────────────────────────────────────────────────────────
models = {
    "Decision Tree":  DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest":  RandomForestClassifier(n_estimators=150, max_depth=10,
                                             random_state=42, n_jobs=-1),
    "Naive Bayes":    GaussianNB(),
}

results = {}
for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    cv_acc = cross_val_score(clf, X, y, cv=5, scoring="accuracy").mean()
    results[name] = {"model": clf, "accuracy": acc, "cv_acc": cv_acc}
    print(f"{name}: Test={acc*100:.2f}%  CV={cv_acc*100:.2f}%")

# ── 6. Best model ─────────────────────────────────────────────────────────────
comparison = pd.DataFrame({
    n: {"Test Accuracy": v["accuracy"], "CV Accuracy": v["cv_acc"]}
    for n, v in results.items()
}).T.sort_values("Test Accuracy", ascending=False)

best_name = comparison.index[0]
best_data = results[best_name]
print(f"\n✔ Best Model → {best_name}  ({best_data['accuracy']*100:.2f}%)")

# ── 7. Save ───────────────────────────────────────────────────────────────────
payload = {
    "model":          best_data["model"],
    "model_name":     best_name,
    "accuracy":       best_data["accuracy"],
    "feature_cols":   FEATURE_COLS,
    "symptom_cols":   symptom_cols,
    "age_encoder":    age_encoder,
    "gender_encoder": gender_encoder,
    "label_encoder":  label_encoder,
}

model_path = os.path.join(MODEL_DIR, "best_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(payload, f)

comparison.to_csv(os.path.join(MODEL_DIR, "model_comparison.csv"))
print(f"Model saved → {model_path}")
print("Done ✓")
