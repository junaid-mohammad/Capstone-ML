import os
import sys
import requests
from flask import Flask, request, jsonify, send_from_directory

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Get the absolute path of the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Initialize Flask app
app = Flask(__name__, static_folder=PROJECT_ROOT, static_url_path="")

APPEND_MAIN_CATEGORY = False  # Set to True to include main category in output
CATEGORY_MAP = {
    "Information Request": "Counseling/Consultation",
    "Web Shop Order": "Business",
    "Course Confirmation": "Business",
    "Office Visit": "Communication Type",
    "Phone Call": "Communication Type",
    "Email": "Communication Type",
    "Facebook": "Communication Type",
}

global OLLAMA_MODEL
OLLAMA_SERVER = "127.0.0.1:11434"
OLLAMA_API_URL = f"http://{OLLAMA_SERVER}/api/chat"
OLLAMA_MODEL = "deepseek-r1:1.5b"

# Get the category and sub-category from the locally hosted model
def predict_category(subject, body, sender):
    prompt = f"""
    You are an AI assistant trained to classify emails for a mental health and wellness NGO.
    Your task is to classify the email below into exactly one of the following categories:
    - Information Request
    - Web Shop Order
    - Course Confirmation

    The email may be written in English, Icelandic, or both.

    Email to classify:
    Subject: {subject}
    Sender: {sender}
    Body: {body}

    Your response must always have the prediction on the LAST and SEPARATE line. 
    Do not include tags like "Prediction:" or quotes.
    Do not add any extra formatting, markdown, or blank lines after the result.
    
    Example (last line only): 
    Course Confirmation
    
    Every email must be classified into exactly one of the categories.
    """

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload)

        if response.status_code != 200:
            return f"Error: API request failed with status {response.status_code} - {response.text}"

        response_data = response.json()
        full_response = response_data.get("message", {}).get("content", "").strip()
        subcategory = full_response.split("\n")[-1].strip()  # Extract the last line

        # Check if subcategory is valid
        if subcategory in CATEGORY_MAP:
            main_category = CATEGORY_MAP[subcategory]
            print(f"[DEBUG] Predicted: {main_category}: {subcategory}")

            if APPEND_MAIN_CATEGORY:
                return f"{main_category}: {subcategory}"
            else:
                return subcategory

        elif subcategory == "Uncategorized":
            return "Uncategorized"
        else:
            print("[DEBUG] Invalid subcategory detected, falling back to full response:\n", full_response)
            return "Uncategorized"

    except Exception as e:
        return f"Error: {str(e)}"

# Serve the index.html page
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
    
    global OLLAMA_MODEL # Override the model if specified in the request header
    
    # Use the model specified in the request headers, or fallback to the default
    model_override = request.headers.get("X-Model")  
    if model_override:
        OLLAMA_MODEL = model_override  # Modify the global variable safely
        # print (f"\n[DEBUG] Model override detected: {OLLAMA_MODEL}")

    # Get prediction
    category = predict_category(subject, body, sender)
    return jsonify({"category": category})

if __name__ == "__main__":
    app.run(debug=True)