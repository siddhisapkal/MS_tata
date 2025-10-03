# -------------------------------
# 0️⃣ Install / Upgrade Dependencies
# -------------------------------
import subprocess
import sys

def install_packages():
    """Install required packages"""
    packages = ['transformers', 'datasets', 'scikit-learn', 'torch', 'pandas', 'numpy']
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', package, '--quiet'])
            print(f"✓ {package} installed/upgraded")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")

# Uncomment the line below to install packages
# install_packages()

# -------------------------------
# 1️⃣ Load JSON Dataset
# -------------------------------
import json
import pandas as pd

json_file = "tata_motors_risk_analysis.json"  # change if needed
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

print(f"Original dataset size: {len(df)}")
print(f"Columns: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head())

# -------------------------------
# 2️⃣ Remove duplicates
# -------------------------------
df = df.drop_duplicates(subset=['Title', 'Explanation'])
print(f"\nDataset size after deduplication: {len(df)}")

# -------------------------------
# 3️⃣ Construct 'text' column for ML
# -------------------------------
df['text'] = df['Title'].fillna('') + ' ' + df['Explanation'].fillna('')
df = df[df['text'].str.strip() != '']  # remove empty text
print(f"Dataset size after cleaning text: {len(df)}")
print("\nSample text data:")
print(df['text'].head())

# -------------------------------
# 4️⃣ Encode categorical targets
# -------------------------------
from sklearn.preprocessing import LabelEncoder

le_risk = LabelEncoder()
df['Risk_Type_enc'] = le_risk.fit_transform(df['Risk_Type'])

le_sev = LabelEncoder()
df['Severity_enc'] = le_sev.fit_transform(df['Severity'])

print(f"\nRisk Types: {le_risk.classes_}")
print(f"Severity Levels: {le_sev.classes_}")

# -------------------------------
# 5️⃣ Train/Test Split
# -------------------------------
from sklearn.model_selection import train_test_split

# Risk_Type
X_train, X_test, y_risk_train, y_risk_test = train_test_split(
    df['text'], df['Risk_Type_enc'], test_size=0.2, random_state=42
)

# Severity
_, _, y_sev_train, y_sev_test = train_test_split(
    df['text'], df['Severity_enc'], test_size=0.2, random_state=42
)

# Risk_Score
_, _, y_score_train, y_score_test = train_test_split(
    df['text'], df['Risk_Score'], test_size=0.2, random_state=42
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# -------------------------------
# 6️⃣ Baseline ML: TF-IDF + RandomForest
# -------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_squared_error, classification_report
import numpy as np

print("\n" + "="*50)
print("BASELINE ML MODELS (TF-IDF + RandomForest)")
print("="*50)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Risk_Type classification
print("\nTraining Risk_Type classifier...")
clf_risk = RandomForestClassifier(n_estimators=100, random_state=42)
clf_risk.fit(X_train_vec, y_risk_train)
y_risk_pred = clf_risk.predict(X_test_vec)
risk_f1 = f1_score(y_risk_test, y_risk_pred, average='weighted')
print(f"Risk_Type F1 Score: {risk_f1:.4f}")

# Severity classification
print("\nTraining Severity classifier...")
clf_sev = RandomForestClassifier(n_estimators=100, random_state=42)
clf_sev.fit(X_train_vec, y_sev_train)
y_sev_pred = clf_sev.predict(X_test_vec)
sev_f1 = f1_score(y_sev_test, y_sev_pred, average='weighted')
print(f"Severity F1 Score: {sev_f1:.4f}")

# Risk_Score regression
print("\nTraining Risk_Score regressor...")
reg_score = RandomForestRegressor(n_estimators=100, random_state=42)
reg_score.fit(X_train_vec, y_score_train)
y_score_pred = reg_score.predict(X_test_vec)
rmse = np.sqrt(mean_squared_error(y_score_test, y_score_pred))
print(f"Risk_Score RMSE: {rmse:.4f}")

# Detailed classification reports
print("\nDetailed Risk_Type Classification Report:")
print(classification_report(y_risk_test, y_risk_pred, target_names=le_risk.classes_))

print("\nDetailed Severity Classification Report:")
print(classification_report(y_sev_test, y_sev_pred, target_names=le_sev.classes_))

# -------------------------------
# 7️⃣ Transformers Fine-Tuning
# -------------------------------
print("\n" + "="*50)
print("TRANSFORMERS FINE-TUNING")
print("="*50)

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from sklearn.metrics import f1_score as sklearn_f1_score

model_name = "distilbert-base-uncased"
print(f"Loading tokenizer and model: {model_name}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize
    print("Tokenizing data...")
    train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=256)
    test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=256)
    
    # Dataset class
    class RiskDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item
    
    train_dataset = RiskDataset(train_encodings, list(y_risk_train))
    test_dataset = RiskDataset(test_encodings, list(y_risk_test))
    
    # TrainingArguments
    training_args = TrainingArguments(
        output_dir='./results_risk',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True
    )
    
    # Model
    model_risk = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(le_risk.classes_)
    )
    
    # Metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = torch.argmax(torch.tensor(logits), axis=1).numpy()
        f1 = sklearn_f1_score(labels, preds, average='weighted')
        return {"f1": f1}
    
    # Trainer
    trainer = Trainer(
        model=model_risk,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    print("Starting transformer training...")
    trainer.train()
    
    # Evaluate the model
    print("\nEvaluating transformer model...")
    eval_results = trainer.evaluate()
    print(f"Transformer F1 Score: {eval_results['eval_f1']:.4f}")
    
    # Make predictions
    predictions = trainer.predict(test_dataset)
    y_pred_transformer = torch.argmax(torch.tensor(predictions.predictions), axis=1).numpy()
    
    print("\nTransformer Risk_Type Classification Report:")
    print(classification_report(y_risk_test, y_pred_transformer, target_names=le_risk.classes_))
    
except Exception as e:
    print(f"Error in transformer training: {e}")
    print("Make sure you have the required packages installed:")
    print("pip install transformers torch datasets scikit-learn")

# -------------------------------
# 8️⃣ Model Comparison and Results Summary
# -------------------------------
print("\n" + "="*50)
print("MODEL COMPARISON SUMMARY")
print("="*50)

print(f"Baseline RandomForest Risk_Type F1: {risk_f1:.4f}")
print(f"Baseline RandomForest Severity F1: {sev_f1:.4f}")
print(f"Baseline RandomForest Risk_Score RMSE: {rmse:.4f}")

try:
    print(f"Transformer DistilBERT Risk_Type F1: {eval_results['eval_f1']:.4f}")
except:
    print("Transformer results not available")

print("\n" + "="*50)
print("ANALYSIS COMPLETE")
print("="*50)
