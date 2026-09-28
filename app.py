from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st


MODEL_PATH = Path(__file__).resolve().with_name("svm_model.pkl")

# These names and this order must match the SVM training data.
MODEL_FEATURE_NAMES = [
    "BMI",
    "Number.of.staples",
    "Ondansetron",
    "Male",
    "Chest.drains",
]

# The model keeps its original field name; only the user-facing label is corrected.
DISPLAY_FEATURE_NAMES = [
    "BMI",
    "Number.of.staplers",
    "Ondansetron",
    "Male",
    "Chest.drains",
]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.set_page_config(page_title="Postoperative CRP Risk Predictor", page_icon="🩺")
st.title("Postoperative CRP Risk Predictor")

st.subheader("Patient information")
col1, col2 = st.columns(2)
with col1:
    bmi = st.number_input("BMI", min_value=0.0, value=25.0, step=0.1)
    number_of_staplers = st.number_input(
        "Number.of.staplers", min_value=0.0, value=0.0, step=1.0
    )
    ondansetron = st.number_input(
        "Ondansetron", min_value=0.0, value=0.0, step=1.0
    )
with col2:
    male = st.selectbox(
        "Male (0=No, 1=Yes)",
        options=[0, 1],
        format_func=lambda value: "No (0)" if value == 0 else "Yes (1)",
    )
    chest_drains = st.number_input(
        "Chest.drains", min_value=0.0, value=0.0, step=1.0
    )

feature_values = [
    bmi,
    number_of_staplers,
    ondansetron,
    male,
    chest_drains,
]
features = pd.DataFrame([feature_values], columns=MODEL_FEATURE_NAMES)

if st.button("Predict"):
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    class_index = int(np.flatnonzero(model.classes_ == predicted_class)[0])
    probability = float(predicted_proba[class_index])

    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")
    st.write(f"Model probability for the predicted class: {probability:.1%}")

    if predicted_class == 1:
        st.warning(
            "The model predicts a higher-risk result. "
            "Please consult your healthcare provider for further evaluation."
        )
    else:
        st.success(
            "The model predicts a lower-risk result. "
            "Continue routine follow-up and healthy habits."
        )

    # Kernel SHAP supports this SVM model. Use support vectors as the background.
    background = getattr(model, "support_vectors_", features.to_numpy())
    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(features, nsamples="auto")

    # Binary classifiers may return either a list (one array per class) or a
    # 3-D array depending on the installed SHAP version.
    if isinstance(shap_values, list):
        class_shap_values = np.asarray(shap_values[class_index])[0]
    else:
        shap_values = np.asarray(shap_values)
        class_shap_values = (
            shap_values[0, :, class_index]
            if shap_values.ndim == 3
            else shap_values[0]
        )

    expected_value = np.asarray(explainer.expected_value)
    base_value = float(expected_value[class_index]) if expected_value.ndim else float(expected_value)
    display_features = features.copy()
    display_features.columns = DISPLAY_FEATURE_NAMES
    shap.force_plot(
        base_value,
        class_shap_values,
        display_features,
        matplotlib=True,
        show=False,
    )
    figure = plt.gcf()
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)
