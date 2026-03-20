"""
generate_dataset.py
-------------------
Generates a realistic synthetic dataset for Malaria and Dengue disease prediction.
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

MALARIA_SYMPTOMS = [
    "fever", "chills", "sweating", "headache", "nausea",
    "vomiting", "muscle_pain", "fatigue", "anemia", "spleen_enlargement",
    "jaundice", "high_fever_cycles", "shivering", "loss_of_appetite", "dark_urine"
]

DENGUE_SYMPTOMS = [
    "fever", "severe_headache", "pain_behind_eyes", "joint_pain", "muscle_pain",
    "skin_rash", "mild_bleeding", "nausea", "vomiting", "fatigue",
    "low_platelet_count", "abdominal_pain", "swollen_glands", "loss_of_appetite", "dizziness"
]

ALL_SYMPTOMS = sorted(set(MALARIA_SYMPTOMS + DENGUE_SYMPTOMS))
AGE_GROUPS   = ["1-10", "10-20", "20-30", "30-40", "40-50", "50-60"]
GENDERS      = ["Male", "Female"]


def generate_record(disease: str) -> dict:
    record = {symptom: 0 for symptom in ALL_SYMPTOMS}

    if disease == "Malaria":
        core     = ["fever", "chills", "sweating", "headache", "fatigue"]
        optional = [s for s in MALARIA_SYMPTOMS if s not in core]
        present  = np.random.choice(optional, size=np.random.randint(3, 7), replace=False)
        for s in core + list(present):
            record[s] = 1
        for s in ["skin_rash", "pain_behind_eyes", "low_platelet_count", "joint_pain"]:
            if np.random.random() < 0.05:
                record[s] = 1
    else:
        core     = ["fever", "severe_headache", "pain_behind_eyes", "joint_pain", "skin_rash"]
        optional = [s for s in DENGUE_SYMPTOMS if s not in core]
        present  = np.random.choice(optional, size=np.random.randint(3, 7), replace=False)
        for s in core + list(present):
            record[s] = 1
        for s in ["chills", "high_fever_cycles", "anemia", "spleen_enlargement"]:
            if np.random.random() < 0.05:
                record[s] = 1

    record["age_group"] = np.random.choice(AGE_GROUPS)
    record["gender"]    = np.random.choice(GENDERS)
    record["disease"]   = disease
    return record


def build_dataset(n_malaria: int = 600, n_dengue: int = 600) -> pd.DataFrame:
    records = (
        [generate_record("Malaria") for _ in range(n_malaria)] +
        [generate_record("Dengue")  for _ in range(n_dengue)]
    )
    return pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = build_dataset()
    df.to_csv("data/disease_dataset.csv", index=False)
    print(f"Dataset saved → data/disease_dataset.csv  ({len(df)} rows)")
    print(df["disease"].value_counts())
