import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Concatenate
from preprocess import load_fold_data, extract_features

fold = sys.argv[1] if len(sys.argv) > 1 else "fold_0"
fold_path = f"data/{fold}"
model_dir = f"model/{fold}"

# Load Data
(train_texts, train_scores), (dev_texts, dev_scores), _ = load_fold_data(fold_path)

# Tokenization
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)
X_train_seq = pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=300)
X_val_seq = pad_sequences(tokenizer.texts_to_sequences(dev_texts), maxlen=300)

# NLP-based Features
X_train_features = np.array([extract_features(text) for text in train_texts])
X_val_features = np.array([extract_features(text) for text in dev_texts])

# Labels
y_train = np.array(train_scores)
y_val = np.array(dev_scores)

# Model Definition
text_input = Input(shape=(300,), name="text_input")
embedding = Embedding(input_dim=10000, output_dim=128, input_length=300)(text_input)
lstm_out = LSTM(64)(embedding)
lstm_out = Dropout(0.5)(lstm_out)

features_input = Input(shape=(3,), name="features_input")
merged = Concatenate()([lstm_out, features_input])
dense = Dense(32, activation='relu')(merged)
output = Dense(1, activation='linear')(dense)

model = Model(inputs=[text_input, features_input], outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train Model
model.fit([X_train_seq, X_train_features], y_train, epochs=5, batch_size=32, validation_data=([X_val_seq, X_val_features], y_val))

# Save Model
os.makedirs(model_dir, exist_ok=True)
model.save(f"{model_dir}/aes_model.h5")

print(f"✅ Model trained and saved to {model_dir}")
