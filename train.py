import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight

# Load dataset
df = pd.read_csv("progeria_dataset_balanced.csv")

# Display class distribution
print("Class Distribution:")
print(df['Progeria_Diagnosis'].value_counts())

# Separate features and target
X = df.drop(columns=['Progeria_Diagnosis']).values
y = df['Progeria_Diagnosis'].values

# One-hot encode target
y_cat = to_categorical(y)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape input for LSTM: (samples, timesteps, features)
X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_cat, test_size=0.2, random_state=42)

# Compute class weights to handle imbalance
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(enumerate(class_weights))

# Build LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.5))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=class_weight_dict,
    verbose=1
)

# Predict and evaluate
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Output directory
output_dir = "model_output_lstm"
os.makedirs(output_dir, exist_ok=True)

# Save trained model
model.save(os.path.join(output_dir, "progeria_lstm_model.h5"))

# Save model parameters
with open(os.path.join(output_dir, "model_parameters.txt"), "w") as f:
    f.write("Model Parameters:\n")
    f.write(f"Input shape: {X_train.shape[1:]}\n")
    f.write("LSTM units: 128\n")
    f.write("Dropout rate: 0.5\n")
    f.write("Optimizer: adam\n")
    f.write("Epochs: 50\n")
    f.write("Batch size: 32\n")

# Save classification report
report = classification_report(y_true, y_pred, target_names=["No Progeria", "Progeria"])
with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
    f.write("Classification Report:\n")
    f.write(report)

# Save confusion matrix image
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Progeria", "Progeria"])
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()
