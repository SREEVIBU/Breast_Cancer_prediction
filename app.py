import os
import pickle
from typing import Dict, Any, List

import streamlit as st
import pandas as pd
import numpy as np

# Paths
ROOT = os.path.dirname(__file__)
MODEL_PATH = os.path.join(ROOT, "breast_cancer.pkl")
DATA_PATH = os.path.join(ROOT, "data.csv")


@st.cache_resource
def load_artifacts():
    """Load model and optional artifacts from MODEL_PATH. Returns dict with keys: model, scaler, le, feature_names."""
    artifacts = {"model": None, "scaler": None, "le": None, "feature_names": None}
    if not os.path.exists(MODEL_PATH):
        st.warning(f"Model file not found at {MODEL_PATH}. Please create 'breast_cancer.pkl' in the project root.")
        return artifacts

    try:
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            artifacts["model"] = data.get("model") or data.get("estimator")
            artifacts["scaler"] = data.get("scaler")
            artifacts["le"] = data.get("label_encoder") or data.get("le")
            artifacts["feature_names"] = data.get("feature_names")
        else:
            artifacts["model"] = data
    except Exception as e:
        st.error(f"Failed to load model from {MODEL_PATH}: {e}")
    return artifacts


@st.cache_data
def load_dataset(nrows: int = 1000) -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH, nrows=nrows)


def infer_feature_names(df: pd.DataFrame) -> List[str]:
    drop = [c for c in ["id", "diagnosis", "Unnamed: 32"] if c in df.columns]
    return [c for c in df.columns if c not in drop]


def predict_from_array(model, scaler, le, X: np.ndarray):
    X_in = X.copy()
    if scaler is not None:
        X_in = scaler.transform(X_in)
    pred = model.predict(X_in)
    # return only the prediction (decoded if label encoder available)
    try:
        if le is not None and hasattr(le, "inverse_transform"):
            return le.inverse_transform(pred)[0]
        else:
            return int(pred[0]) if hasattr(pred[0], "__int__") else str(pred[0])
    except Exception:
        return str(pred[0])


def main():
    st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")
    st.title("Breast Cancer Prediction (Streamlit)")

    artifacts = load_artifacts()
    model = artifacts.get("model")
    scaler = artifacts.get("scaler")
    le = artifacts.get("le")
    feature_names = artifacts.get("feature_names")

    # If the model exposes feature_names_in_ use it (convert to list)
    if artifacts.get("model") is not None and hasattr(artifacts.get("model"), "feature_names_in_"):
        try:
            feature_names = list(getattr(artifacts.get("model"), "feature_names_in_"))
        except Exception:
            pass

    if model is None:
        st.error("Model not loaded. Place a valid 'breast_cancer.pkl' in the project folder.")

    df = load_dataset(2000)
    if df.empty:
        st.warning("data.csv not found or empty. Sample/manual inputs will be limited.")

    if feature_names is None and not df.empty:
        feature_names = infer_feature_names(df)

    st.sidebar.header("Input")
    st.sidebar.markdown("Only manual input mode is enabled for this app.")
    st.sidebar.markdown("---")
    if model is not None:
            st.sidebar.success(f"Model loaded: {type(model).__name__}")
            st.sidebar.write(f"Expected features: {getattr(model, 'n_features_in_', len(feature_names) if feature_names else 'unknown')}")

    # Manual input mode only
    st.subheader("Manual input")
    # Determine final required features
    required_features = feature_names
    if model is not None and hasattr(model, "n_features_in_"):
        if required_features is None:
            # can't infer names, but we know number
            required_count = int(getattr(model, "n_features_in_"))
            st.warning(f"Model expects {required_count} features but names are not available. Please include feature_names in the pickle.")
            return
        else:
            # ensure count matches
            if len(required_features) != int(getattr(model, "n_features_in_")):
                st.error(f"Feature count mismatch: model expects {int(getattr(model, 'n_features_in_'))} features but {len(required_features)} were inferred.")
                st.stop()

    if not required_features:
        st.error("Feature names not available; please include `feature_names` in the pickle.")
        return

    # build a form
    with st.form("manual_form"):
        cols = st.columns(3)
        inputs = {}
        defaults = {}
        if not df.empty:
            defaults = df[feature_names].mean().to_dict()

        for i, fname in enumerate(feature_names):
            col = cols[i % 3]
            val = defaults.get(fname, 0.0)
            inputs[fname] = col.number_input(fname, value=float(val), format="%.5f")

        submit = st.form_submit_button("Predict")
        if submit:
            X = np.array([inputs[f] for f in required_features], dtype=float).reshape(1, -1)
            # validate feature length against model
            if model is not None and hasattr(model, "n_features_in_") and X.shape[1] != int(getattr(model, "n_features_in_")):
                st.error(f"Input feature length {X.shape[1]} does not match model expectation {int(getattr(model, 'n_features_in_'))}.")
            else:
                res = predict_from_array(model, scaler, le, X)
                st.success(f"Prediction: {res}")

    # Display only a short instruction
    st.markdown("---")
    st.info("Prediction shown above. Run with: `streamlit run app.py`")


if __name__ == "__main__":
    main()
