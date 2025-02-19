import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Define categories
CATEGORIES = {
    "Counselling/Consultation": ["Information Request"],
    "Business": ["Web Shop Order", "Course Confirmation"],
    "Communication Type": ["Office Visit", "Phone Call", "Email", "Facebook"],
}
all_categories = [category for subcategories in CATEGORIES.values() for category in subcategories]
CATEGORY_TO_INT = {category: idx for idx, category in enumerate(all_categories)}
INT_TO_CATEGORY = {idx: category for category, idx in CATEGORY_TO_INT.items()}

model_dir = 'model/trained_roberta'

# Example of loading the model again for inference
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# # Predicting the category for a new text
# print("Enter <subject> FROM <sender>: <body>")
# new_text = input()
# # new_text = "Question About My Recent Order FROM lina@gmail.com: Hello, I recently ordered a set of sensory-friendly clothing from your shop. I wanted to check if the order has been shipped and if there’s a tracking number available. Could you please provide an update?"
# inputs = tokenizer(new_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
# with torch.no_grad():
#     outputs = model(**inputs)
#     predictions = torch.argmax(outputs.logits, dim=-1)
#     predicted_label = INT_TO_CATEGORY[predictions.item()]
#     print(f"Predicted Category: {predicted_label}")

# Predict from file json
df = pd.read_json("data/AI_data2.json")  # Training data file
texts = [email["subject"] + " FROM " + email["sender"] + ": " + email["body"] for email in df.to_dict(orient="records")]
expected_categories = [email["expected_category"] for email in df.to_dict(orient="records")]
labels = [CATEGORY_TO_INT[email["expected_category"]] for email in df.to_dict(orient="records")]

score = 0
for i in range(50):
    new_text = texts[i]
    inputs = tokenizer(new_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        predicted_label = INT_TO_CATEGORY[predictions.item()]
        print(f"Predicted Category: {predicted_label}, Expected Category: {expected_categories[i]}")
        if predicted_label == expected_categories[i]:
            score = score + 1

print(f"{score}/{len(texts)}")
