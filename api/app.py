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

# Updated predefined categories with new structure
CATEGORIES = {
    "Counselling/Consultation": ["Information Request"],
    "Business": ["Web Shop Order", "Course Confirmation"],
    "Communication Type": ["Office Visit", "Phone Call", "Email", "Facebook"],
}


# Function to get the category and sub-category from OpenAI API
def predict_category(subject, body, sender):
    prompt = f"""
    You are an AI email categorization assistant for a mental health and wellness NGO. 
    Classify the following email into one of the main categories and corresponding sub-categories from this structure:
    - Counselling/Consultation: Information Request
    - Business: Web Shop Order, Course Confirmation
    - Communication Type: Office Visit, Phone Call, Email, Facebook

    The email may be written in Icelandic, English, or a mix of both. Provide your response in this format:
    "[Main Category]: [Specific Sub-Category] e.g. Counselling/Consultation: Information Request"

    Subject: {subject}
    Sender: {sender}
    Body: {body}

    Respond with only the category and sub-category.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Cheapest model
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0,
            max_tokens=50,
        )

        category_response = response.choices[0].message.content.strip()
        return category_response

    except Exception as e:
        return f"Error: {str(e)}"


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
