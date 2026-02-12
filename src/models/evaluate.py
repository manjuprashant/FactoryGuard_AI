import os
import pandas as pd
import joblib
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

DATA_PATH = "data/factory_guard_data.csv"
MODEL_DIR = "models"

df = pd.read_csv(DATA_PATH)
X = df["text"]
y = df["label"]

vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
rf = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
xgb = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
tokenizer = joblib.load(os.path.join(MODEL_DIR, "tokenizer.pkl"))
bilstm = load_model(os.path.join(MODEL_DIR, "bilstm_model.keras"))

X_tfidf = vectorizer.transform(X)
print("Random Forest Report")
print(classification_report(y, rf.predict(X_tfidf)))

print("XGBoost Report")
print(classification_report(y, xgb.predict(X_tfidf)))

X_seq = pad_sequences(tokenizer.texts_to_sequences(X), maxlen=20)
bilstm_preds = (bilstm.predict(X_seq) > 0.5).astype("int32")

print("BiLSTM Report")
print(classification_report(y, bilstm_preds))
