import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# --------------------------------------------------------
# 1. Load model pipeline and dataset schema
# --------------------------------------------------------
@st.cache_resource
def load_model_and_schema():
    # load the pickled pipeline
    pkl_path = Path("loan_knn.pkl")
    if not pkl_path.exists():
        st.error("loan_knn.pkl not found. Place it in the same folder as app.py.")
        st.stop()

    with open(pkl_path, "rb") as f:
        model = pickle.load(f)

    # load CSV only to get feature names & options (not to train)
    csv_path = Path("loan_data.csv")
    if not csv_path.exists():
        st.error("loan_data.csv not found. Place it in the same folder as app.py.")
        st.stop()

    df = pd.read_csv(csv_path)

    # >>> use the same target column name as in Kaggle <<<
    target_col = "loan_status"   # change if different

    X = df.drop(columns=[target_col])

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "bool", "category"]
    ).columns.tolist()

    # distinct values for each categorical feature to build selectboxes
    cat_values = {
        col: sorted(df[col].dropna().unique()) for col in categorical_features
    }

    feature_cols = X.columns.tolist()

    return model, feature_cols, numeric_features, categorical_features, cat_values, target_col


model, feature_cols, numeric_features, categorical_features, cat_values, target_col = (
    load_model_and_schema()
)

# --------------------------------------------------------
# 2. Streamlit UI
# --------------------------------------------------------
st.title("Loan Approval Prediction (K-NN, Pickle Pipeline)")

st.markdown(
    """
This app uses a **K-Nearest Neighbors (KNN)** classifier trained on a loan approval
dataset. The preprocessing + KNN pipeline is loaded from a pickle file.
"""
)

st.subheader("Enter applicant details")

user_input = {}

# numeric inputs
for col in numeric_features:
    user_input[col] = st.number_input(
        col,
        value=0.0,
        help=f"Numeric feature: {col}",
    )

# categorical inputs
for col in categorical_features:
    options = cat_values.get(col, [])
    if options:
        user_input[col] = st.selectbox(
            col,
            options=options,
            help=f"Categorical feature: {col}",
        )
    else:
        user_input[col] = st.text_input(col)

# build DataFrame in the exact same order as training features
input_df = pd.DataFrame([{col: user_input[col] for col in feature_cols}])

if st.button("Predict loan status"):
    pred = model.predict(input_df)[0]

    st.success(f"Predicted {target_col}: **{pred}**")

    # show class probabilities if supported
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]
        class_labels = model.named_steps["knn"].classes_
        st.write("Class probabilities:")
        for label, p in zip(class_labels, proba):

            st.write(f"- {label}: {p:.3f}")
