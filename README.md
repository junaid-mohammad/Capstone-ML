# Capstone Project - Machine Learning Email Categorization

## Overview

This project is designed to categorize incoming emails for a Polish NPO based on the sender, subject, and body of the emails. The machine learning model suggests a category for each email, which users can accept, modify, or replace with a new category. User feedback is incorporated to improve the model continuously.

The project uses Python for machine learning and text processing, with a lightweight setup to ensure cross-platform compatibility for macOS and Windows users. Additionally, a Flask-based API provides backend functionality, and a simple web interface enables users to interact with the model.

---

## Setup Instructions

### Prerequisites

- **Python 3.10+** installed on your system
- Git installed
- A terminal or command line interface (e.g., Terminal for macOS, Command Prompt or PowerShell for Windows)

---

### Step-by-Step Guide to Set Up and Run the Project

#### 1. Clone the Repository

Open a terminal and run the following commands:

```bash
git clone https://github.com/junaid-mohammad/Capstone-ML.git
cd Capstone-ML
```

#### 2. Create a Virtual Environment

- **macOS/Linux:**

  ```bash
  python3 -m venv machine_learning_env
  source machine_learning_env/bin/activate
  ```

- **Windows:**
  ```bash
  python3 -m venv machine_learning_env
  machine_learning_env\Scripts\activate
  ```

#### 3. Install Dependencies

With the virtual environment activated, install the required Python libraries:

```bash
pip install -r requirements.txt
```

#### 4. Train the Model

Run the following command to train the email categorization model:

```bash
python3 model/train_model.py
```

#### 5. Test the Model - Start the Flask API

To enable API calls and serve the web interface, start the Flask app:

```bash
python3 api/app.py
```

This will start the local development server at `http://127.0.0.1:5000`.

#### 6. Access the Web Interface

Open a browser and navigate to:

```
http://127.0.0.1:5000
```

Here, you can enter an email subject and body into the form, submit the data, and see the predicted category displayed below the form.

---

### **Confirm Model is Trained and API and Web Interface Work**

After completing the above steps:

1. Ensure the model is trained without errors during the training step.

2. Test the model using different email inputs on the browser http://127.0.0.1:5000/ to verify predictions.

3. **Test the `/predict` API Endpoint**:

   **Note**: Testing `/predict` directly with Postman or `curl` allows you to validate the API independently of the web interface. Use this to confirm the model works before testing the HTML form.

   - Use a tool like Postman or `curl` to send POST requests directly to the API.
   - Example request:
     ```bash
     curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"subject": "Meeting Update", "body": "Please attend the meeting at 10 AM"}'
     ```
   - Expected response:
     ```json
     {
       "category": "Task Assignment"
     }
     ```

4. **Test the Web Interface**:
   - Open `http://127.0.0.1:5000` in a browser.
   - Enter test data into the form and verify the predicted category appears below.

---

## Project Structure

This section explains the purpose of each folder and file in the project.

```
Capstone-ML/
│
├── model/
│   ├── train_model.py         # Trains the email categorization model and saves it as `email_categorizer.pkl`
│   ├── predict_category.py    # Provides reusable prediction logic for the trained model
│   ├── feedback_loop.py       # Script (to be developed) for handling user feedback
│   ├── email_categorizer.pkl  # Serialized trained model used for predictions
│
├── data/
│   ├── dummy_data.csv         # Example dataset for initial model training
│
├── api/
│   ├── app.py                 # Flask API to serve predictions and the web interface
│
├── index.html                 # Web interface for email categorization
│
├── utils/
│   ├── preprocess.py          # Utility functions for text preprocessing
│
├── tests/
│   ├── test_model.py          # Unit tests for ensuring model functionality
│
├── machine_learning_env/      # Virtual environment folder (ignored by Git)
│
├── requirements.txt           # List of dependencies required for the project
├── .gitignore                 # Specifies files and folders to ignore in Git
└── README.md                  # Documentation for the project
```

### Key Components

1. **model/**

   - Contains scripts for model training, testing, and continuous improvement.
   - `train_model.py` trains the model using the dataset in `data/`.
   - `predict_category.py` Contains reusable logic for predicting categories based on email subject and body. Imported into `app.py` for serving predictions via the API.

2. **data/**

   - Holds the datasets for model training.
   - `dummy_data.csv` is a sample dataset for initial testing and training.

3. **api/**

   - `app.py` - Flask application that serves:
     - **API endpoints** for predictions (e.g., `/predict`).
     - **Static files**, including `index.html`, for the web interface.

4. **`index.html`**:

   - A user-friendly interface to input email data and see predicted categories.
   - Interacts with the Flask API using JavaScript.

5. **utils/**

   - Contains helper scripts for tasks like text preprocessing or data formatting.

6. **tests/**

   - Includes unit tests to validate different aspects of the project (e.g., model accuracy, preprocessing correctness).

7. **requirements.txt**

   - Lists all Python dependencies for the project. Use this file to install the required libraries.

8. **.gitignore**
   - Ensures unnecessary or large files, like the `machine_learning_env/` folder, are not tracked by Git.

---

## Future Development

1. **Feedback Loop:**
   - Develop `feedback_loop.py` to incorporate user feedback into model improvement.
2. **API Integration:**
   - Add endpoints for advanced functionality, like logging predictions or retrieving prediction history.
3. **Dataset Expansion:**
   - Replace the dummy dataset with real-world email data to improve the model's accuracy.

---

## Contributors

- Junaid Arif
- Shyam Desai
- Minji Chang
- Harry Park
