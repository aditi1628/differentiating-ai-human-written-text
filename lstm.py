import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

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

max_words = 5000
max_len = 200
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_padded = pad_sequences(X_train_seq, maxlen=max_len)
X_test_padded = pad_sequences(X_test_seq, maxlen=max_len)

model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    SpatialDropout1D(0.2),
    LSTM(100, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model and save history
history = model.fit(X_train_padded, y_train, epochs=5, batch_size=64, validation_data=(X_test_padded, y_test), verbose=2)

# Save the training history for later plotting
np.save('history.npy', history.history)

# Make predictions and calculate accuracy
y_pred = (model.predict(X_test_padded) > 0.5).astype("int32")
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy (LSTM): {accuracy:.4f}")

# Save results
np.save('lstm_test_accuracy.npy', accuracy)
np.save('lstm_predictions.npy', y_pred)
np.save('y_test_lstm.npy', y_test)

# GUI Application
def predict_text():
    user_input = text_input.get("1.0", "end-1c")
    if user_input.strip():
        user_seq = tokenizer.texts_to_sequences([user_input])
        user_padded = pad_sequences(user_seq, maxlen=max_len)
        prediction = (model.predict(user_padded) > 0.5).astype("int32")
        result = "AI-Generated" if prediction[0][0] == 1 else "Human-Written"
        messagebox.showinfo("Prediction", f"The text is predicted to be: {result}")
    else:
        messagebox.showwarning("Input Error", "Please enter some text to analyze.")

# Tkinter setup
root = tk.Tk()
root.title("LSTM Text Classifier")

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
