# ==========================================
# INTERACTIVE EMOTION CLASSIFIER INFERENCE
# ==========================================

import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("--- 1. LOADING SINGLE-FILE MODEL & TOKENIZER ---")
model_file = "emotion_model.pth"
tokenizer_path = "./tokenizer_cache"

if not os.path.exists(model_file):
    raise FileNotFoundError(f"Could not find model file '{model_file}'. Run 'train_emotion.py' first!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

# Load checkpoint dictionary
checkpoint = torch.load(model_file, map_location=device)

# Initialize model architecture and load weights
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint['model_checkpoint'], 
    num_labels=checkpoint['num_labels']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Load label mapping if available
label_mapping = None
for root, dirs, files in os.walk("."):
    if "labels.json" in files:
        with open(os.path.join(root, "labels.json"), "r") as f:
            label_mapping = json.load(f)
        break

print("\n--- 2. READY FOR TESTING ---")
print("Type any sentence below to test the emotion classifier. Type 'exit' or 'quit' to stop.\n")

while True:
    user_input = input("Enter a comment/sentence: ").strip()
    
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting testing tool.")
        break
        
    if not user_input:
        continue

    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=128).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)
        predicted_class_id = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class_id].item() * 100

    if label_mapping and str(predicted_class_id) in label_mapping:
        emotion_name = label_mapping[str(predicted_class_id)]
        print(f"👉 Predicted Emotion: {emotion_name.upper()} (ID: {predicted_class_id}) | Confidence: {confidence:.2f}%\n")
    else:
        print(f"👉 Predicted Class ID: {predicted_class_id} | Confidence: {confidence:.2f}%\n")