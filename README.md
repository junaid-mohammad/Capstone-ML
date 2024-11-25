# Capstone Project - Machine Learning Email Categorization

## Overview

This project is designed to categorize incoming emails for a Polish NPO based on the sender, subject, and body of the emails. The machine learning model suggests a category for each email, which users can accept, modify, or replace with a new category. User feedback is incorporated to improve the model continuously.

The project uses Python for machine learning and text processing, with a lightweight setup to ensure cross-platform compatibility for macOS and Windows users.

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
  python -m venv machine_learning_env
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
python model/train_model.py
```

#### 5. Test the Model

To test the model, run:

```bash
python model/predict_category.py
```

You will be prompted to input a subject and body. The model will output the predicted category for the given input.

---

### Confirm Everything Works

After completing the above steps:

1. Ensure the model is trained without errors during the training step.
2. Test the model using different email inputs to verify predictions.

---

## Project Structure

This section explains the purpose of each folder and file in the project.

```
Capstone-ML/
│
├── model/
│   ├── train_model.py         # Script to train the email categorization model
│   ├── predict_category.py    # Script to test predictions from the trained model
│   ├── feedback_loop.py       # Script (to be developed) for handling user feedback
│
├── data/
│   ├── dummy_data.csv         # Example dataset for initial model training
│
├── api/
│   ├── app.py                 # Flask API for integrating the machine learning model with a frontend (to be developed)
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
   - `predict_category.py` allows you to test the trained model's predictions.

2. **data/**

   - Holds the datasets for model training.
   - `dummy_data.csv` is a sample dataset for initial testing and training.

3. **api/**

   - Will house the backend Flask API for serving the model to a frontend or other services.

4. **utils/**

   - Contains helper scripts for tasks like text preprocessing or data formatting.

5. **tests/**

   - Includes unit tests to validate different aspects of the project (e.g., model accuracy, preprocessing correctness).

6. **requirements.txt**

   - Lists all Python dependencies for the project. Use this file to install the required libraries.

7. **.gitignore**
   - Ensures unnecessary or large files, like the `machine_learning_env/` folder, are not tracked by Git.

---

## Future Development

1. **Feedback Loop:**
   - Develop `feedback_loop.py` to incorporate user feedback into model improvement.
2. **API Integration:**
   - Build a Flask-based API to interact with the machine learning model and serve predictions.
3. **Dataset Expansion:**
   - Replace the dummy dataset with real-world email data to improve the model's accuracy.

---

## Contributors

- Junaid Arif
- Shyam Desai
- Minji Chang
- Harry Park
