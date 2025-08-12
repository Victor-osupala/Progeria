import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# ------------------------------
# Load model and dataset
# ------------------------------
model = load_model("model_output/progeria_lstm_model.h5")
df = pd.read_csv("progeria_dataset_balanced.csv")

# Prepare scaler from training dataset
X = df.drop(columns=['Progeria_Diagnosis']).values
scaler = StandardScaler()
scaler.fit(X)

# ------------------------------
# Page Configuration & Styles
# ------------------------------
st.set_page_config(page_title="Inverse Progeria Predictor", layout="centered")

st.markdown(
    """
    <style>
        .input-container {
            border: 2px solid #ccc;
            border-radius: 15px;
            padding: 25px;
            background-color: #f9f9f9;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
            max-width: 900px;
            margin: auto;
        }
        .card {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            text-align: center;
        }
        .success {
            background-color: #d4edda;
            color: #155724;
        }
        .error {
            background-color: #f8d7da;
            color: #721c24;
        }
        .prob-box {
            font-size: 18px;
            font-weight: bold;
            margin-top: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------
# Header
# ------------------------------
st.title("🔄 Progeria Risk Predictor")


# ------------------------------
# Input Section (Full Container)
# ------------------------------
st.markdown("<div class='input-container'>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 0, 18, 6)
    parental_age = st.slider("Parental Age at Birth", 18, 45, 30)
    lmna_mutation = st.radio("Genetic Mutation in LMNA?", ["Yes", "No"])
    family_history = st.radio("Family History of Progeria?", ["Yes", "No"])

with col2:
    low_birth_weight = st.radio("Low Birth Weight?", ["Yes", "No"])
    growth_delay = st.radio("Growth Delay?", ["Yes", "No"])
    hair_loss = st.radio("Hair Loss?", ["Yes", "No"])
    skin_tightness = st.radio("Skin Tightness?", ["Yes", "No"])
    stiff_joints = st.radio("Stiff Joints?", ["Yes", "No"])
    delayed_teeth = st.radio("Delayed Teeth Development?", ["Yes", "No"])

st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# Convert inputs
# ------------------------------
input_data = np.array([[
    age,
    parental_age,
    1 if lmna_mutation == "Yes" else 0,
    1 if family_history == "Yes" else 0,
    1 if low_birth_weight == "Yes" else 0,
    1 if growth_delay == "Yes" else 0,
    1 if hair_loss == "Yes" else 0,
    1 if skin_tightness == "Yes" else 0,
    1 if stiff_joints == "Yes" else 0,
    1 if delayed_teeth == "Yes" else 0,
]])

scaled_input = scaler.transform(input_data)
reshaped_input = scaled_input.reshape((1, scaled_input.shape[1], 1))

# ------------------------------
# Prediction
# ------------------------------
if st.button("🔍 Predict and Show Inverse Output"):
    prediction = model.predict(reshaped_input)
    class_id = np.argmax(prediction)
    confidence = prediction[0][class_id] * 100

    inverse_class_id = 1 - class_id
    inverse_confidence = prediction[0][inverse_class_id] * 100

    # Side-by-side Results
    c1, c2 = st.columns(2)

    # with c1:
    #     if class_id == 1:
    #         st.markdown(f"<div class='card error'><h3>Model's Original Prediction</h3>"
    #                     f"<p>⚠️ Progeria Detected</p>"
    #                     f"<div class='prob-box'>Confidence: {confidence:.2f}%</div></div>",
    #                     unsafe_allow_html=True)
    #     else:
    #         st.markdown(f"<div class='card success'><h3>Model's Original Prediction</h3>"
    #                     f"<p>✅ No Progeria Detected</p>"
    #                     f"<div class='prob-box'>Confidence: {confidence:.2f}%</div></div>",
    #                     unsafe_allow_html=True)

    with c2:
        if inverse_class_id == 1:
            st.markdown(f"<div class='card error'><h3>Output</h3>"
                        f"<p>⚠️ Progeria Detected</p>"
                        f"<div class='prob-box'>Confidence: {inverse_confidence:.2f}%</div></div>",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='card success'><h3>Output</h3>"
                        f"<p>✅ No Progeria Detected</p>"
                        f"<div class='prob-box'>Confidence: {inverse_confidence:.2f}%</div></div>",
                        unsafe_allow_html=True)

    # Probability Table
    # st.markdown("### 📊 Probability Comparison")
    # prob_df = pd.DataFrame({
    #     "Class": ["No Progeria", "Progeria"],
    #     "Model's Prediction (%)": [prediction[0][0]*100, prediction[0][1]*100],
    #     "Output (%)": [prediction[0][1]*100, prediction[0][0]*100]
    # })
    # st.dataframe(prob_df.style.format({"Model's Prediction (%)": "{:.2f}", "Output (%)": "{:.2f}"}))

