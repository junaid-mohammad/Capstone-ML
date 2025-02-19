import os
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import torch

# Define categories
CATEGORIES = {
    "Counselling/Consultation": ["Information Request"],
    "Business": ["Web Shop Order", "Course Confirmation"],
    "Communication Type": ["Office Visit", "Phone Call", "Email", "Facebook"],
}
all_categories = [category for subcategories in CATEGORIES.values() for category in subcategories]
CATEGORY_TO_INT = {category: idx for idx, category in enumerate(all_categories)}
INT_TO_CATEGORY = {idx: category for category, idx in CATEGORY_TO_INT.items()}

model_dir = 'model/trained_roberta'

# Check if the model already exists
if os.path.exists(model_dir):
    print("Loading pre-trained model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
else:
    print("Training a new model...")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=len(CATEGORIES))


class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            # add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),  # Remove the batch dimension
            'attention_mask': encoding['attention_mask'].squeeze(),  # Mask for padding tokens
            'labels': torch.tensor(label, dtype=torch.long)  # Convert label to tensor
        }


# Load the json data
df = pd.read_json("data/AI_data2.json")  # Training data file
texts = [email["subject"] + " FROM " + email["sender"] + ": " + email["body"] for email in df.to_dict(orient="records")]
labels = [CATEGORY_TO_INT[email["expected_category"]] for email in df.to_dict(orient="records")]

train_dataset = CustomDataset(texts, labels, tokenizer, max_len=128)

training_args = TrainingArguments(
    output_dir='./model/trained_roberta',  # output directory to save the model
    save_strategy="no",
    num_train_epochs=3,  # number of training epochs
    per_device_train_batch_size=8,  # batch size during training
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir='./model/roberta_logs',  # directory for storing logs
    logging_steps=10,  # log every 10 steps
)
# Define your custom dataset (train_dataset)
train_dataset = CustomDataset(texts, labels, tokenizer, max_len=256)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Start training locally
trainer.train()

# Save the trained model locally
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)


# # Example of loading the model again for inference
# model = AutoModelForSequenceClassification.from_pretrained(model_dir)
# tokenizer = AutoTokenizer.from_pretrained(model_dir)
#
# # Predicting the category for a new text
# new_text = "Question About My Recent Order FROM lina@gmail.com: Hello, I recently ordered a set of sensory-friendly clothing from your shop. I wanted to check if the order has been shipped and if there’s a tracking number available. Could you please provide an update?"
# inputs = tokenizer(new_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
# with torch.no_grad():
#     outputs = model(**inputs)
#     predictions = torch.argmax(outputs.logits, dim=-1)
#     predicted_label = INT_TO_CATEGORY[predictions.item()]
#     print(f"Predicted Category: {predicted_label}")
#     print("answer: Web Shop Order")