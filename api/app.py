import os
import sys
import json
import requests
from flask import Flask, request, jsonify, send_from_directory

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Get the absolute path of the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Initialize Flask app
app = Flask(__name__, static_folder=PROJECT_ROOT, static_url_path="")

# Updated predefined categories with new structure
CATEGORIES = {
    "Counseling/Consultation": ["Information Request"],
    "Business": ["Web Shop Order", "Course Confirmation"],
    "Communication Type": ["Office Visit", "Phone Call", "Email", "Facebook"],
}

# Ollama Local API URL
global OLLAMA_MODEL
OLLAMA_SERVER = "127.0.0.1:11434"
OLLAMA_API_URL = f"http://{OLLAMA_SERVER}/api/chat"
OLLAMA_MODEL = "deepseek-r1:1.5b"

# Get the category and sub-category from the locally hosted model
def predict_category(subject, body, sender):
    prompt = f"""
    You are an AI assistant trained to classify emails for a mental health and wellness NGO.
    Your task is to classify the email below into exactly into one of the following [Main Category]: [Sub-Category] formats:
    - Information Request
    - Web Shop Order
    - Course Confirmation

    The email may be written in English, Icelandic, or both.

    Email to classify:
    Subject: {subject}
    Sender: {sender}
    Body: {body}

    The last line of the output format must be a single line, Do NOT add explanations, tags, or extra formatting, exactly in this format:
    Main Category: Sub-Category

    Example:
    Counseling/Consultation: Information Request
    Business: Course Confirmation
    Business: Web Shop Order
    
    If the email does not fit any of the categories, please classify it as "Uncategorized".
    """

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    try:
        # Send the request to the Ollama Local API
        print("[DEBUG] Sending request to Ollama Local API:")
        print("URL:", OLLAMA_API_URL)
        print("Payload:", json.dumps(payload, indent=4))

        response = requests.post(OLLAMA_API_URL, json=payload)

        # Check if the request was successful
        if response.status_code != 200:
            return f"Error: API request failed with status {response.status_code} - {response.text}"

        response_data = response.json()
        print("[DEBUG] API Response:", json.dumps(response_data, indent=4))

        # Extract the model's response
        if "message" in response_data and "content" in response_data["message"]:
            full_response = response_data["message"]["content"].strip()
            category = full_response.split("\n")[-1].strip()  # Extract the last line
        else:
            return f"Error: Unexpected API response format - {response_data}"

        try:
            # Extract main category and sub-category
            main_category, sub_category = category.split(":", 1)
            main_category, sub_category = main_category.strip(), sub_category.strip()

            # Validate that the response is a known category
            if main_category in CATEGORIES and sub_category in CATEGORIES[main_category]:
                print("[DEBUG] Predicted Category:", main_category, "-", sub_category)
                return category

            print("[WARNING] Model returned an unknown category. Logging response.")
            print("[DEBUG] Full Response:", full_response, sub_category, main_category)
            return "Uncategorized"

        except ValueError as e:
            print("\n[ERROR] Issue extracting category. Logging response.")
            print("[DEBUG] Full Response:", full_response)
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
        print (f"\n[DEBUG] Model override detected: {OLLAMA_MODEL}")

    # Get prediction
    category = predict_category(subject, body, sender)
    return jsonify({"category": category})

if __name__ == "__main__":
    app.run(debug=True)