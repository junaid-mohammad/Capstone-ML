# Inference script for model prediction
import pickle
import os

# Load the trained model
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
with open(os.path.join(PROJECT_ROOT, "model/email_categorizer.pkl"), "rb") as f:
    model = pickle.load(f)


# Prediction function
def predict_category(subject, body):
    text = subject + " " + body
    return model.predict([text])[0]
