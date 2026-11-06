# ==========================================
# MODEL EVALUATION & PERFORMANCE METRICS
# ==========================================

import os
import ast
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader

print("--- 1. LOADING SINGLE-FILE MODEL & TEST DATA ---")
model_file = "emotion_model.pth"
tokenizer_path = "./tokenizer_cache"

if not os.path.exists(model_file):
    raise FileNotFoundError(f"Model file '{model_file}' not found. Run training first!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

# Load test data
data_dir = "Ecommerce Emotion_Train_Test_80_20"
test_path = os.path.join(data_dir, "test.csv")
if not os.path.exists(test_path):
    for root, dirs, files in os.walk("."):
        if "test.csv" in files:
            test_path = os.path.join(root, "test.csv")

test_df = pd.read_csv(test_path)
text_col = 'text' if 'text' in test_df.columns else test_df.columns[0]
label_col = 'labels' if 'labels' in test_df.columns else test_df.columns[1]

test_df = test_df[[text_col, label_col]].dropna()

def parse_label(val):
    if isinstance(val, str):
        val = val.strip()
        if val.startswith('[') and val.endswith(']'):
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list) and len(parsed) > 0:
                    return int(parsed[0])
            except Exception:
                pass
    try:
        return int(val)
    except Exception:
        return 0

test_df[label_col] = test_df[label_col].apply(parse_label)
true_labels = test_df[label_col].tolist()

print("\n--- 2. TOKENIZING TEST DATA ---")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
test_dataset = Dataset.from_pandas(test_df)

def tokenize_function(examples):
    return tokenizer(examples[text_col], truncation=True, max_length=128)

tokenized_test = test_dataset.map(tokenize_function, batched=True, remove_columns=[text_col])
if label_col != 'labels':
    tokenized_test = tokenized_test.rename_column(label_col, 'labels')
tokenized_test.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
test_dataloader = DataLoader(tokenized_test, batch_size=32, collate_fn=data_collator)

print("\n--- 3. LOADING MODEL WEIGHTS & EVALUATING ---")
checkpoint = torch.load(model_file, map_location=device)
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint['model_checkpoint'], 
    num_labels=checkpoint['num_labels']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

all_predictions = []
with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1)
        all_predictions.extend(preds.cpu().tolist())

print("\n--- 4. PERFORMANCE METRICS ---")
accuracy = accuracy_score(true_labels, all_predictions)
print(f"\nOverall Test Accuracy: {accuracy * 100:.2f}%\n")
print(classification_report(true_labels, all_predictions, zero_division=0))