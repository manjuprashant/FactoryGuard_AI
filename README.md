# 🏭 Factory Guard AI

AI-based industrial safety monitoring system that classifies factory incidents using ML and DL models.

## 🚀 Features
- Random Forest
- XGBoost
- BiLSTM (TensorFlow)
- ROC Curve
- Confusion Matrix
- Model Comparison

## ⚙️ Setup

### 1. Create Virtual Environment
py -3.11 -m venv venv
venv\Scripts\activate

### 2. Install Requirements
pip install -r requirements.txt

### 3. Generate Dataset
py src/models/dataset_generate.py

### 4. Train Models
py src/models/train_model.py

### 5. Evaluate
py src/models/evaluate.py
