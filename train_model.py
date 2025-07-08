import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Load dataset
DATASET_PATH = 'C:\\Users\\Admin\\Desktop\\rifthan\\imdb\\IMDB Dataset.csv'

df = pd.read_csv(DATASET_PATH, encoding='utf-8')

# Convert sentiment to binary
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Basic preprocessing
def preprocess(text):
    return text.lower().strip()

df['review'] = df['review'].apply(preprocess)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=42
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# Save model and vectorizer
MODEL_PATH = 'model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'

with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)

with open(VECTORIZER_PATH, 'wb') as f:
    pickle.dump(vectorizer, f)

print(f"✅ Training complete. Model saved to {MODEL_PATH}, Vectorizer saved to {VECTORIZER_PATH}")
