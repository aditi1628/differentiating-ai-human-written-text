import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Load saved true labels (y_test) and predictions (y_pred) for the ERTC model
model = "LR"  # Change to "LR" or "LSTM" for respective models

if model == "ERTC":
    y_test = np.load('y_test_ertc.npy')  # Replace with your actual saved y_test file for ERTC model
    y_pred = np.load('ertc_predictions.npy')  # Saved predictions for ERTC
    title = "Confusion Matrix of ERTC Model"
elif model == "LR":
    y_test = np.load('y_test_lr.npy')  # Replace with your actual saved y_test file for LR model
    y_pred = np.load('lr_predictions.npy')  # Saved predictions for Logistic Regression
    title = "Confusion Matrix of Logistic Regression Model"
elif model == "LSTM":
    y_test = np.load('y_test_lstm.npy')  # Replace with your actual saved y_test file for LSTM model
    y_pred = np.load('lstm_predictions.npy')  # Saved predictions for LSTM
    title = "Confusion Matrix of LSTM Model"

# Class names (adjust based on your encoding)
class_names = ['Human-written', 'AI-generated']

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title(title)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
