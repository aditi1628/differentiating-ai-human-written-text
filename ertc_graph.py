import numpy as np
import matplotlib.pyplot as plt

# Load the training accuracy data
history_data = np.load('ertc_history.npy', allow_pickle=True).item()

# Extract data from history
estimators_range = history_data['estimators_range']
train_accuracies = history_data['train_accuracies']
val_accuracies = history_data['val_accuracies']

# Check that the lengths match
if len(estimators_range) == len(train_accuracies) == len(val_accuracies):
    # Plot training & validation accuracy values
    plt.figure(figsize=(10, 6))
    plt.plot(estimators_range, train_accuracies, label='Train Accuracy')
    plt.plot(estimators_range, val_accuracies, label='Validation Accuracy')
    plt.title('Extra Trees Classifier Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Estimators')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()
else:
    print("Error: Lengths of the arrays do not match.")
