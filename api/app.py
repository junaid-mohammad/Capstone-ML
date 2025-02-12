import os
import sys
import json
import requests
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv  # Load environment variables

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Get the absolute path of the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Initialize Flask app
app = Flask(__name__, static_folder=PROJECT_ROOT, static_url_path="")

# Load environment variables from .env file
load_dotenv()

# Get API Key from environment variables
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_ENDPOINT = os.getenv("DEEPSEEK_API_ENDPOINT")
DEEPSEEK_MODEL = 'deepseek-r1:1.5b'
DEEPSEEK_TEMPERATURE = 0.6  # Response randomness (0 = deterministic, 1 = more creative); Documentation recommends 0.5 to 0.7

if not DEEPSEEK_API_KEY or not DEEPSEEK_API_ENDPOINT:
    raise ValueError("Missing DeepSeek API Key or Endpoint. Set them in the .env file.")

# Updated predefined categories with NGO-specific focus
CATEGORIES = [
    "General Inquiry",
    "Patient Support & Counseling Request",
    "Funding & Donations",
    "Appointment Scheduling",
    "Volunteer & Internship Applications",
    "Legal & Compliance",
    "Awareness & Advocacy",
    "Educational Resources",
    "Family & Caregiver Support",
    "Spam & Irrelevant Messages",
]


# Function to get the category from OpenAI API
def predict_category(subject, body, sender):
    prompt = f"""
    You are an AI email categorization assistant for a small mental health and wellness NGO. Classify the following email into one of the predefined categories: {', '.join(CATEGORIES)}.

    Sender: {sender}
    Subject: {subject}
    Body: {body}

    Respond with only the category name.
    """

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": DEEPSEEK_MODEL,
        "temperature": DEEPSEEK_TEMPERATURE,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    # Print the request details
    print("\n[DEBUG] Sending request to DeepSeek API:")
    print("URL:", DEEPSEEK_API_ENDPOINT)
    print("Headers:", json.dumps(headers, indent=4))
    print("Payload:", json.dumps(payload, indent=4))

    try:
        response = requests.post(DEEPSEEK_API_ENDPOINT, json=payload, headers=headers)

        # Print the response details
        print("\n[DEBUG] Response from DeepSeek API:")
        print("Status Code:", response.status_code)
        print("Response Body:", response.text)

        # Check if the request was successful
        if response.status_code != 200:
            return f"Error: API request failed with status {response.status_code} - {response.text}"

        response_data = response.json()

        # Extract only the last line after the <think> block to get the predicted category from the response
        full_response = response_data["choices"][0]["message"]["content"].strip()
        category = full_response.split("\n")[-1].strip()  # Extract the last line

        print("\n[DEBUG] Predicted Category:", category)

        # Validate that the response is one of the predefined categories
        if category not in CATEGORIES:
            category = "Uncategorized"  # Fallback if the AI provides unexpected output

        return category

    except Exception as e:
        return f"Error: {str(e)}"


# Serve the index.html page
@app.route("/")
def index():
    return send_from_directory(PROJECT_ROOT, "index.html")


# API endpoint for prediction
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    sender = data.get("sender", "").strip()
    subject = data.get("subject", "").strip()
    body = data.get("body", "").strip()

    if not sender or not subject or not body:
        return jsonify({"error": "Sender, subject, and body are required."}), 400

    # Get prediction
    category = predict_category(subject, body, sender)
    return jsonify({"category": category})


if __name__ == "__main__":
    app.run(debug=True)