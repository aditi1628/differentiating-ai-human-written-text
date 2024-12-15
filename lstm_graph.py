import matplotlib.pyplot as plt
import numpy as np

# Load the training history
history = np.load('history.npy', allow_pickle=True).item()

# Plot training & validation accuracy values
plt.figure(figsize=(10, 6))
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('LSTM Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()
