import os
import re
import sys
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import extract_features

MAX_LEN = 300  # Define the same max length used during training

def clean_text(text):
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_tokenizer(path):
    with open(path, 'r') as f:
        return tokenizer_from_json(json.load(f))

def predict_essay(fold):
    model_path = f"model/{fold}/aes_model.h5"
    tokenizer_path = f"model/{fold}/tokenizer.json"

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}. Train first!")
        return

    model = load_model(model_path)
    tokenizer = load_tokenizer(tokenizer_path)

    essay = input("✏️ Enter your essay:\n")
    cleaned = clean_text(essay)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')

    features = extract_features(cleaned).reshape(1, -1)

    pred = model.predict([padded, features])[0][0]
    print(f"\n📊 Predicted Score: {round(pred, 2)}")

if __name__ == "__main__":
    fold = sys.argv[1] if len(sys.argv) > 1 else "fold_0"
    predict_essay(fold)
