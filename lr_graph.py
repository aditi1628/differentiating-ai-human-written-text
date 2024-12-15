import numpy as np
import matplotlib.pyplot as plt

# Load the cross-validation scores (this should have been generated during the model evaluation)
cv_scores = np.load('logistic_cv_scores.npy')

# Plot cross-validation accuracy values
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cv_scores) + 1), cv_scores, marker='o', linestyle='-', color='b')
plt.title('Logistic Regression Cross-Validation Accuracy')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.xticks(range(1, len(cv_scores) + 1))
plt.grid(True)
plt.show()
