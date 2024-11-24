# Model training logic

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pickle

# Load dataset
data = pd.read_csv("data/dummy_data.csv")

# Combine subject and body for training
data["text"] = data["subject"] + " " + data["body"]

# Prepare features and labels
X = data["text"]
y = data["category"]

# Build model pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train model
model.fit(X, y)

# Save model
with open("model/email_categorizer.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")
