import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, request, jsonify, send_from_directory
from model.predict_category import predict_category  # Import the function

# Get the absolute path of the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Initialize Flask app
app = Flask(__name__, static_folder=PROJECT_ROOT, static_url_path="")


# Serve the index.html page
@app.route("/")
def index():
    return send_from_directory(PROJECT_ROOT, "index.html")


# API endpoint for prediction
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    subject = data.get("subject", "")
    body = data.get("body", "")

    if not subject or not body:
        return jsonify({"error": "Subject and body are required."}), 400

    # Get prediction
    category = predict_category(subject, body)
    return jsonify({"category": category})


if __name__ == "__main__":
    app.run(debug=True)
