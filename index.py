from datasets import load_dataset

# Load the dataset
dataset = load_dataset("iamketan25/essay-instructions-dataset")

# View a sample of the dataset
print(dataset)

# Inspect the columns and types
print(dataset['train'].features)
import pandas as pd
import re

# Load datasets-
dataset = pd.read_csv('article_level_data.csv')
dataset_2 = pd.read_csv('sentence_level_data.csv')

# Function to clean text
def clean_text(text):
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.lower().strip()  # Convert to lowercase and strip whitespace

# Clean the article-level dataset
dataset['article'] = dataset['article'].apply(clean_text)
dataset = dataset.drop(columns=['Unnamed: 0'])

# Clean the sentence-level dataset
dataset_2['sentence'] = dataset_2['sentence'].apply(clean_text)
dataset_2 = dataset_2.drop(columns=['Unnamed: 0'])

# Check for missing values
print("Missing values in articles:\n", dataset.isnull().sum())
print("Missing values in sentences:\n", dataset_2.isnull().sum())

# Summary of cleaned datasets
print("Number of articles:", len(dataset))
print("Number of sentences:", len(dataset_2))
