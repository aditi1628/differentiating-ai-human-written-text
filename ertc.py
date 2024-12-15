import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold  # Import KFold here
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox

# Load datasets
article_data = pd.read_csv('article_level_data.csv')
sentence_data = pd.read_csv('sentence_level_data.csv')

article_data.rename(columns={'article': 'text', 'class': 'label'}, inplace=True)
sentence_data.rename(columns={'sentence': 'text', 'class': 'label'}, inplace=True)

data = pd.concat([article_data, sentence_data], ignore_index=True)
data.dropna(subset=['text', 'label'], inplace=True)

X = data['text']
y = data['label']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = ExtraTreesClassifier(n_estimators=100, random_state=42)

# Cross-validation to track accuracy during training
cross_val_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring='accuracy')

# Calculate training and validation accuracy separately
train_accuracies = []
val_accuracies = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in kf.split(X_train_tfidf):
    model.fit(X_train_tfidf[train_idx], y_train[train_idx])
    val_accuracy = accuracy_score(y_train[val_idx], model.predict(X_train_tfidf[val_idx]))
    train_accuracies.append(accuracy_score(y_train[train_idx], model.predict(X_train_tfidf[train_idx])))
    val_accuracies.append(val_accuracy)

# Average training and validation accuracies
avg_train_accuracy = np.mean(train_accuracies)
avg_val_accuracy = np.mean(val_accuracies)
print(f"Average Training Accuracy: {avg_train_accuracy:.4f}")
print(f"Average Validation Accuracy: {avg_val_accuracy:.4f}")

# Fit the model
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy (ERTC): {accuracy:.4f}")

# Save results
np.save('ertc_test_accuracy.npy', accuracy)
np.save('ertc_predictions.npy', y_pred)
np.save('y_test_ertc.npy', y_test)  # Save y_test for confusion matrix plotting

# Save training history (for plotting)
history_data = {
    'estimators_range': np.arange(1, len(train_accuracies) + 1),
    'train_accuracies': train_accuracies,
    'val_accuracies': val_accuracies
}
assert len(history_data['estimators_range']) == len(history_data['train_accuracies'])
assert len(history_data['estimators_range']) == len(history_data['val_accuracies'])

np.save('ertc_history.npy', history_data)  # Save history data for plotting

# GUI Application
def predict_text():
    user_input = text_input.get("1.0", "end-1c")
    if user_input.strip():
        user_vector = vectorizer.transform([user_input])
        prediction = model.predict(user_vector)
        result = "AI-Generated" if prediction[0] == 1 else "Human-Written"
        messagebox.showinfo("Prediction", f"The text is predicted to be: {result}")
    else:
        messagebox.showwarning("Input Error", "Please enter some text to analyze.")

# Tkinter setup
root = tk.Tk()
root.title("ERTC Text Classifier")

frame = tk.Frame(root, padx=10, pady=10)
frame.pack(pady=20)

label = tk.Label(frame, text="Enter text for classification:")
label.pack()

text_input = tk.Text(frame, height=10, width=50)
text_input.pack(pady=10)

predict_button = tk.Button(frame, text="Predict", command=predict_text)
predict_button.pack()

accuracy_label = tk.Label(frame, text=f"Test Accuracy: {accuracy:.4f}")
accuracy_label.pack(pady=10)

root.mainloop()
