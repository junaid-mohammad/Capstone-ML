# Capstone Project - Machine Learning Email Categorization (Updated for GPT-based Model)

## Overview

This project is designed to categorize incoming emails for a Polish NPO based on the sender, subject, and body of the emails. Initially, we used a custom-built machine learning model trained on a local dataset (`dummy_data.csv`). However, after extensive evaluation, we transitioned to using a **GPT-based categorization model** for its superior accuracy, flexibility, and ease of integration.

The current implementation leverages OpenAI’s **GPT API**, which classifies emails into one of ten predefined categories relevant to the NGO’s focus on mental health and wellness.

Despite its advantages, this approach raises potential **privacy concerns** due to data processing on external servers. Therefore, it is considered a **prototype** and may eventually be replaced by a **self-hosted LLM** for better control and privacy.

---

## Why the Shift to GPT?

The primary reasons for transitioning to GPT-based categorization include:

1. **Higher Accuracy and Robustness** – Pretrained models like GPT are better at understanding complex language patterns.
2. **Faster Development and Integration** – Avoided the time-consuming process of building and fine-tuning a custom model.
3. **Scalability and Flexibility** – GPT can easily handle new categories and adapt to changing requirements.
4. **Simplified Maintenance** – No need to retrain and deploy models regularly.

However, this transition was also guided by practical limitations of the initial approach, such as limited data availability and concerns about prediction accuracy using a small dataset.

---

## Setup Instructions (Updated)

### Prerequisites

- **Python 3.10+** installed on your system.
- Git installed.
- A terminal or command line interface (e.g., Terminal for macOS, Command Prompt or PowerShell for Windows).
- **OpenAI API Key** – You must generate your own API key from [OpenAI's platform](https://platform.openai.com/) and add it to a `.env` file.

---

### Step-by-Step Guide to Set Up and Run the Project

#### 1. Clone the Repository

```bash
git clone https://github.com/junaid-mohammad/Capstone-ML.git
cd Capstone-ML
```

#### 2. Create a Virtual Environment

```bash
python3 -m venv machine_learning_env
source machine_learning_env/bin/activate  # On macOS/Linux
machine_learning_env\Scripts\activate  # On Windows
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Add Your OpenAI API Key

1. Create a `.env` file in the project root directory.
2. Add the following line to your `.env` file:

   ```
   OPENAI_API_KEY=your-api-key-here
   ```

This API key is required for the GPT-based email categorization.

#### 5. Start the Flask API

```bash
python api/app.py
```

The Flask app will start at `http://127.0.0.1:5000`.

#### 6. Access the Web Interface

Open a browser and navigate to:

```
http://127.0.0.1:5000
```

Enter the sender’s email, email subject, and body. The predicted category will be displayed below the form.

---

## Project Structure

Here’s an updated overview of the project structure:

```
Capstone-ML/
│
├── api/
│   ├── app.py                 # Flask API to serve predictions and the web interface.
│
├── data/
│   ├── AI_data.json           # Example dataset generated using AI
│
├── static/
│   ├── styles.css             # CSS for the modern black-themed user interface.
│
├── index.html                 # Web interface for email categorization.
│
├── .env                       # Environment file to store your OpenAI API key.
├── requirements.txt           # List of dependencies required for the project.
├── .gitignore                 # Specifies files and folders to ignore in Git.
└── README.md                  # Documentation for the project.
```

### Key Components

1. **api/app.py** – Contains the Flask API for handling requests and interacting with OpenAI’s API for predictions.

2. **data/AI_data.json** – Sample dataset for testing the AI model.

3. **index.html** – A user-friendly web interface for entering email details and receiving predictions.

4. **static/styles.css** – Custom CSS for a modern and intuitive user experience.

5. **.env** – Stores the OpenAI API key securely.

---

## Supported Categories

The system categorizes emails into the following categories:

1. 📋 **General Inquiry** – General questions about the NGO’s mission or services.
2. 🧠 **Patient Support & Counseling Request** – Seeking mental health resources or therapy guidance.
3. 💰 **Funding & Donations** – Related to fundraising, donor inquiries, and grants.
4. 📅 **Appointment Scheduling** – Requests to schedule or reschedule therapy sessions.
5. 🤝 **Volunteer & Internship Applications** – Applications for volunteering or internships.
6. ⚖️ **Legal & Compliance** – Topics related to patient rights, GDPR, or legal matters.
7. 📢 **Awareness & Advocacy** – Requests for partnerships, campaigns, or media inquiries.
8. 📚 **Educational Resources** – Requests for mental health guides, articles, or workshops.
9. 👨‍👩‍👧 **Family & Caregiver Support** – Guidance for caregivers supporting individuals with mental health conditions.
10. 🛑 **Spam & Irrelevant Messages** – Advertisements, promotions, or unrelated emails.

---

## Privacy Concerns and Future Plans

While the GPT-based solution demonstrates the potential for automated email categorization, it also introduces significant privacy concerns:

1. **Data Privacy Risks** – Since the data is processed on OpenAI’s servers, sensitive information could be exposed.
2. **External Dependency** – Relying on an external service may not align with the NGO’s long-term goals.

### Next Steps:

1. **Explore Self-Hosted Open-Source LLMs** – This would allow the NGO to maintain full control over its data.
2. **Refine Data Anonymization Processes** – Ensure sensitive data is adequately anonymized before processing.
3. **Develop Feedback and Logging Systems** – Improve the categorization accuracy through continuous learning from user feedback.

---

## Contributors

- Junaid Arif
- Shyam Desai
- Minji Chang
- Harry Park

---
