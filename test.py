import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os

# === Streamlit Configuration ===
st.set_page_config(page_title="Progeria Detection (LSTM)", layout="centered")

st.title("🧬 Progeria Detection Using LSTM")
st.write("Upload a balanced dataset for training, and optionally test on new patient data.")

# === Upload dataset ===
uploaded_file = st.file_uploader("Upload Progeria Dataset CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    if "Progeria_Diagnosis" not in df.columns:
        st.error("Dataset must include a 'Progeria_Diagnosis' column.")
        st.stop()
    
    st.success("✅ Dataset loaded successfully.")
    st.write("### Sample of Dataset")
    st.dataframe(df.head())

    # Prepare features and labels
    X = df.drop(columns=["Progeria_Diagnosis"]).values
    y = df["Progeria_Diagnosis"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y_cat = to_categorical(y)
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y_cat, test_size=0.2, random_state=42, stratify=y_cat
    )

    y_labels = np.argmax(y_cat, axis=1)
    class_weights = compute_class_weight("balanced", classes=np.unique(y_labels), y=y_labels)
    class_weight_dict = dict(enumerate(class_weights))

    st.info("📈 Training LSTM model... Please wait.")
    # Build LSTM model
    model = Sequential()
    model.add(LSTM(128, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.5))
    model.add(Dense(y_train.shape[1], activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, y_test),
        class_weight=class_weight_dict,
        verbose=0
    )

    st.success("✅ Training complete!")

    # Evaluate
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    report = classification_report(y_true, y_pred, target_names=["No Progeria", "Progeria"], output_dict=True)
    st.write("### Classification Report")
    st.dataframe(pd.DataFrame(report).transpose())

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Progeria", "Progeria"])
    disp.plot(ax=ax)
    st.pyplot(fig)

    # Save model temporarily
    model.save("temp_lstm_model.h5")

    # === Optional: Upload new data for prediction ===
    st.write("---")
    st.write("### 🔍 Upload New Patient Data for Prediction")
    predict_file = st.file_uploader("Upload new CSV data (same features as training)", type=["csv"], key="predict")

    if predict_file is not None:
        test_df = pd.read_csv(predict_file)
        if test_df.shape[1] != X.shape[1]:
            st.error("Uploaded test data must match the feature size of training data.")
        else:
            test_X = scaler.transform(test_df.values)
            test_X_reshaped = test_X.reshape((test_X.shape[0], test_X.shape[1], 1))

            predictions = model.predict(test_X_reshaped)
            pred_classes = np.argmax(predictions, axis=1)

            test_df["Prediction"] = ["No Progeria" if p == 0 else "Progeria" for p in pred_classes]
            st.success("🎉 Predictions complete!")
            st.dataframe(test_df)

            # Download results
            csv = test_df.to_csv(index=False).encode()
            st.download_button("Download Prediction Results", data=csv, file_name="progeria_predictions.csv", mime="text/csv")
