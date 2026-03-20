"""
predictor.py
------------
Reusable prediction function used by the Streamlit app.
Loads the saved Pickle model and exposes predict(), get_symptom_list(), get_model_info().
"""

import pickle
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "model" / "best_model.pkl"

_cache = {}


def _load():
    if "payload" not in _cache:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Please run generate_dataset.py and train_model.py first."
            )
        with open(MODEL_PATH, "rb") as f:
            _cache["payload"] = pickle.load(f)
    return _cache["payload"]


def get_symptom_list() -> list:
    payload = _load()
    return payload["symptom_cols"]


def get_model_info() -> dict:
    payload = _load()
    return {
        "model_name": payload["model_name"],
        "accuracy":   payload["accuracy"],
    }


def predict(selected_symptoms: list, age_group: str, gender: str) -> dict:
    """
    Parameters
    ----------
    selected_symptoms : list of symptom column names the user selected
    age_group         : one of '1-10','10-20','20-30','30-40','40-50','50-60'
    gender            : 'Male' or 'Female'

    Returns
    -------
    dict with keys: disease, confidence, probabilities
    """
    payload       = _load()
    model         = payload["model"]
    feature_cols  = payload["feature_cols"]
    symptom_cols  = payload["symptom_cols"]
    age_enc       = payload["age_encoder"]
    gender_enc    = payload["gender_encoder"]
    label_enc     = payload["label_encoder"]

    # Build feature vector
    row = {s: (1 if s in selected_symptoms else 0) for s in symptom_cols}

    try:
        row["age_group_enc"] = int(age_enc.transform([age_group])[0])
    except Exception:
        row["age_group_enc"] = 0

    try:
        row["gender_enc"] = int(gender_enc.transform([gender])[0])
    except Exception:
        row["gender_enc"] = 0

    X = np.array([[row[c] for c in feature_cols]])

    pred_idx = model.predict(X)[0]
    disease  = label_enc.inverse_transform([pred_idx])[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        conf  = float(proba[pred_idx])
        probs = {
            label_enc.inverse_transform([i])[0]: float(p)
            for i, p in enumerate(proba)
        }
    else:
        conf  = 1.0
        probs = {disease: 1.0}

    return {
        "disease":       disease,
        "confidence":    conf,
        "probabilities": probs,
    }
