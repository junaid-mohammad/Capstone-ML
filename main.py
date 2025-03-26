import os
import sys
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
import requests

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Initialize Flask app
app = Flask(__name__, static_folder="static", static_url_path="/static")

'''
The model load time (when switching) is an additional ~10-50 seconds overhead. The DEEPSEEK_MODEL available are:
- deepseek-r1:1.5b (~20-60s prediction time)
- deepseek-r1:7b (~3-8 minutes prediction time)
'''

# Load environment variables
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_ENDPOINT = os.getenv("DEEPSEEK_API_ENDPOINT")
DEEPSEEK_MODEL = 'deepseek-r1:1.5b'
DEEPSEEK_TEMPERATURE = 0.6
APPEND_MAIN_CATEGORY = False  # Set to True to include main category in response (not recommended)

if not DEEPSEEK_API_KEY or not DEEPSEEK_API_ENDPOINT:
    raise ValueError("Missing DeepSeek API Key or Endpoint. Set them in the .env file.")

# Sub-category → Main category mapping
CATEGORY_MAP = {
    "Information Request": "Counselling/Consultation",
    "Web Shop Order": "Business",
    "Course Confirmation": "Business",
    "Office Visit": "Communication Type",
    "Phone Call": "Communication Type",
    "Email": "Communication Type",
    "Facebook": "Communication Type",
}

# Function to get the category and sub-category from OpenAI API
def predict_category(subject, sender, body):
    prompt = f"""
    You are an AI assistant trained to classify emails for a mental health and wellness NGO.

    Classify the following email into exactly one of the following categories:
    - Information Request
    - Web Shop Order
    - Course Confirmation

    The email may be written in English, Icelandic, or both.

    Email:
    Subject: {subject}
    Sender: {sender}
    Body: {body}

    Your response must ALWAYS have the prediction on the LAST and SEPARATE line.
    Do not include labels like "Prediction:", quotes, or markdown.
    Do not add any extra formatting, markdown, or blank lines after the result.
    No extra commentary.

    Example (last line only):
    Information Request
    """

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": DEEPSEEK_MODEL,
        "temperature": DEEPSEEK_TEMPERATURE,
        "messages": [{"role": "user", "content": prompt}]
    }

    print("\n[DEBUG] Sending request to DeepSeek API...")
    try:
        response = requests.post(DEEPSEEK_API_ENDPOINT, json=payload, headers=headers)
        print("[DEBUG] Status Code:", response.status_code)

        if response.status_code != 200:
            return f"Error: API request failed - {response.status_code}: {response.text}"

        full_response = response.json()["choices"][0]["message"]["content"].strip()
        subcategory = full_response.split("\n")[-1].strip()
        print(f"[DEBUG] Subcategory predicted: {subcategory}")

        if subcategory in CATEGORY_MAP:
            main_category = CATEGORY_MAP[subcategory]
            return f"{main_category}: {subcategory}" if APPEND_MAIN_CATEGORY else subcategory
        elif subcategory.lower() == "uncategorized":
            return "Uncategorized"
        else:
            print("[WARNING] Invalid subcategory. Full response was:\n", full_response)
            return "Uncategorized"

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
    category_string = predict_category(subject, body, sender)

    # Convert to list format, e.g., "Business: Web Shop Order" → ["Business", "Web Shop Order"]
    category_list = (
        [part.strip() for part in category_string.split(":")]
        if ":" in category_string
        else [category_string]
    )

    return jsonify(category_list)

if __name__ == "__main__":
    app.run(debug=True)