
#logistic regression model used 


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'article_level_data.csv'  # Update this path if necessary
df = pd.read_csv(file_path)

# Step 1: Split the dataset into training and testing sets
X = df['article']  # Features (the articles)
y = df['class']    # Labels (0: human-written, 1: AI-generated)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Convert articles into numerical features
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Step 3: Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Step 4: Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Step 5: Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print a detailed classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Step 6: Plot the confusion matrix as a heatmap
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Human-written', 'AI-generated'], yticklabels=['Human-written', 'AI-generated'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
