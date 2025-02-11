import os
import sys
import openai
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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API Key. Set it in the .env file.")

# Initialize OpenAI client with the correct structure
client = openai.OpenAI(api_key=OPENAI_API_KEY)

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

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Cheapest model
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )

        category = response.choices[0].message.content.strip()

        # Validate that the response is one of the predefined categories
        if category not in CATEGORIES:
            category = "Uncategorized"  # Fallback if ChatGPT gives an unexpected output

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
