import os
import sys
import openai
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv  # Load environment variables

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_dir = 'model/trained_roberta'
if not os.path.exists(model_dir):
    raise Exception("Model does not exist. Create by 'python model/roberta_train.py")

# Example of loading the model again for inference
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Get the absolute path of the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Initialize Flask app
app = Flask(__name__, static_folder=PROJECT_ROOT, static_url_path="")
#
# # Load environment variables from .env file
# load_dotenv()
#
# # Get API Key from environment variables
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#
# if not OPENAI_API_KEY:
#     raise ValueError("Missing OpenAI API Key. Set it in the .env file.")
#
# # Initialize OpenAI client with the correct structure
# client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Updated predefined categories with new structure
CATEGORIES = {
    "Counselling/Consultation": ["Information Request"],
    "Business": ["Web Shop Order", "Course Confirmation"],
    "Communication Type": ["Office Visit", "Phone Call", "Email", "Facebook"],
}
all_categories = [category for subcategories in CATEGORIES.values() for category in subcategories]
CATEGORY_TO_INT = {category: idx for idx, category in enumerate(all_categories)}
INT_TO_CATEGORY = {idx: category for category, idx in CATEGORY_TO_INT.items()}



# Function to get the category and sub-category from OpenAI API
def predict_category(subject, body, sender):
    # Predicting the category for a new text
    text = subject + " FROM " + sender + ": " + body
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        predicted_label = INT_TO_CATEGORY[predictions.item()]
        # print(f"Predicted Category: {predicted_label}")
        return predicted_label


# Serve the index.html page with no-cache header to prevent caching issues
@app.route("/")
def index():
    response = send_from_directory(PROJECT_ROOT, "index.html")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# API endpoint for prediction
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    subject = data.get("subject", "").strip()
    sender = data.get("sender", "").strip()
    body = data.get("body", "").strip()

    if not sender or not subject or not body:
        return jsonify({"error": "Subject, sender, and body are required."}), 400

    # Get prediction
    category = predict_category(subject, body, sender)
    return jsonify({"category": category})


if __name__ == "__main__":
    app.run(debug=True)
