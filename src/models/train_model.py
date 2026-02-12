import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

DATA_PATH = "data/factory_guard_data.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train_tfidf, y_train)
joblib.dump(rf, os.path.join(MODEL_DIR, "rf_model.pkl"))

# XGBoost
xgb = XGBClassifier(eval_metric="logloss")
xgb.fit(X_train_tfidf, y_train)
joblib.dump(xgb, os.path.join(MODEL_DIR, "xgb_model.pkl"))

# BiLSTM
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=20)
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=20)

model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=20),
    LSTM(64),
    Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train_seq, y_train, epochs=3, verbose=0)

model.save(os.path.join(MODEL_DIR, "bilstm_model.keras"))
joblib.dump(tokenizer, os.path.join(MODEL_DIR, "tokenizer.pkl"))

print("✅ Models trained and saved.")
