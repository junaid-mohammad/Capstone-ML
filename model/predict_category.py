# Inference script
import pickle

# Load the trained model
with open("model/email_categorizer.pkl", "rb") as f:
    model = pickle.load(f)


def predict_category(subject, body):
    text = subject + " " + body
    return model.predict([text])[0]


# Example usage
if __name__ == "__main__":
    subject = "Budget Update"
    body = "Please find the budget report attached."
    print("Predicted Category:", predict_category(subject, body))
