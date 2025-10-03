# Enhanced Risk Analysis ML Pipeline with Complete Transformer Training
# =================================================================

import json
import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_squared_error, classification_report, confusion_matrix
# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️ Matplotlib/Seaborn not available - skipping visualizations")
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    AutoModelForTokenClassification, Trainer, TrainingArguments,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# =================================================================
# 1. Enhanced Data Loading & Preprocessing
# =================================================================

def load_and_preprocess_data(json_file="tata_motors_risk_analysis.json"):
    """Load and preprocess the risk analysis dataset"""
    print("🔄 Loading and preprocessing data...")
    
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    print(f"Original dataset size: {len(df)}")
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['Title', 'Explanation'])
    print(f"After deduplication: {len(df)}")
    
    # Enhanced text preprocessing
    df['text'] = df['Title'].fillna('') + ' ' + df['Explanation'].fillna('')
    df = df[df['text'].str.strip() != '']
    
    # Add additional features
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    df['has_affected_nodes'] = df['Affected_Nodes'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)
    
    print(f"Final dataset size: {len(df)}")
    return df

# =================================================================
# 2. Advanced NLP Preprocessing
# =================================================================

import re

# Optional NLTK imports with fallbacks
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠️ NLTK not available - using basic text preprocessing")

def download_nltk_data():
    """Download required NLTK data"""
    if not NLTK_AVAILABLE:
        return
    
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("📥 Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)

def advanced_text_preprocessing(text):
    """Advanced NLP preprocessing with fallbacks"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    if NLTK_AVAILABLE:
        try:
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words]
            
            # Lemmatization
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
            
            return ' '.join(tokens)
        except Exception as e:
            print(f"⚠️ NLTK preprocessing failed: {e}, using basic preprocessing")
    
    # Fallback: basic preprocessing
    tokens = text.split()
    # Simple stopword removal (basic English stopwords)
    basic_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
    tokens = [word for word in tokens if word not in basic_stopwords]
    
    return ' '.join(tokens)

# =================================================================
# 3. Enhanced Baseline Models
# =================================================================

def train_baseline_models(df):
    """Train enhanced baseline models with additional features"""
    print("\n🔧 Training Enhanced Baseline Models...")
    
    # Prepare features
    X_text = df['text']
    X_processed = df['text'].apply(advanced_text_preprocessing)
    
    # Additional features
    X_additional = df[['text_length', 'word_count', 'has_affected_nodes']].values
    
    # Encode targets
    le_risk = LabelEncoder()
    le_sev = LabelEncoder()
    df['Risk_Type_enc'] = le_risk.fit_transform(df['Risk_Type'])
    df['Severity_enc'] = le_sev.fit_transform(df['Severity'])
    
    # Train/test split
    X_train, X_test, y_risk_train, y_risk_test = train_test_split(
        X_text, df['Risk_Type_enc'], test_size=0.2, random_state=42, stratify=df['Risk_Type_enc']
    )
    
    _, _, y_sev_train, y_sev_test = train_test_split(
        X_text, df['Severity_enc'], test_size=0.2, random_state=42, stratify=df['Severity_enc']
    )
    
    _, _, y_score_train, y_score_test = train_test_split(
        X_text, df['Risk_Score'], test_size=0.2, random_state=42
    )
    
    # Additional features split
    X_add_train, X_add_test, _, _ = train_test_split(
        X_additional, df['Risk_Type_enc'], test_size=0.2, random_state=42
    )
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Combine text and additional features
    from scipy.sparse import hstack
    X_train_combined = hstack([X_train_vec, X_add_train])
    X_test_combined = hstack([X_test_vec, X_add_test])
    
    # Train models
    models = {}
    
    # Risk Type Classification
    clf_risk = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf_risk.fit(X_train_combined, y_risk_train)
    y_risk_pred = clf_risk.predict(X_test_combined)
    risk_f1 = f1_score(y_risk_test, y_risk_pred, average='weighted')
    
    # Severity Classification
    clf_sev = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf_sev.fit(X_train_combined, y_sev_train)
    y_sev_pred = clf_sev.predict(X_test_combined)
    sev_f1 = f1_score(y_sev_test, y_sev_pred, average='weighted')
    
    # Risk Score Regression
    reg_score = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    reg_score.fit(X_train_combined, y_score_train)
    y_score_pred = reg_score.predict(X_test_combined)
    rmse = np.sqrt(mean_squared_error(y_score_test, y_score_pred))
    
    models = {
        'vectorizer': vectorizer,
        'clf_risk': clf_risk,
        'clf_sev': clf_sev,
        'reg_score': reg_score,
        'le_risk': le_risk,
        'le_sev': le_sev
    }
    
    results = {
        'risk_f1': risk_f1,
        'sev_f1': sev_f1,
        'score_rmse': rmse,
        'y_risk_test': y_risk_test,
        'y_risk_pred': y_risk_pred,
        'y_sev_test': y_sev_test,
        'y_sev_pred': y_sev_pred,
        'y_score_test': y_score_test,
        'y_score_pred': y_score_pred
    }
    
    print(f"✅ Risk Type F1: {risk_f1:.4f}")
    print(f"✅ Severity F1: {sev_f1:.4f}")
    print(f"✅ Risk Score RMSE: {rmse:.4f}")
    
    return models, results

# =================================================================
# 4. Multi-Task Transformer Training
# =================================================================

class MultiTaskRiskDataset(Dataset):
    """Multi-task dataset for joint classification and regression"""
    def __init__(self, encodings, risk_labels, severity_labels, score_labels):
        self.encodings = encodings
        self.risk_labels = risk_labels
        self.severity_labels = severity_labels
        self.score_labels = score_labels
    
    def __len__(self):
        return len(self.risk_labels)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['risk_labels'] = torch.tensor(self.risk_labels[idx])
        item['severity_labels'] = torch.tensor(self.severity_labels[idx])
        item['score_labels'] = torch.tensor(self.score_labels[idx], dtype=torch.float)
        return item

class MultiTaskModel(torch.nn.Module):
    """Multi-task transformer model"""
    def __init__(self, model_name, num_risk_classes, num_severity_classes):
        super().__init__()
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_risk_classes
        )
        self.risk_classifier = torch.nn.Linear(self.transformer.config.hidden_size, num_risk_classes)
        self.severity_classifier = torch.nn.Linear(self.transformer.config.hidden_size, num_severity_classes)
        self.score_regressor = torch.nn.Linear(self.transformer.config.hidden_size, 1)
        self.dropout = torch.nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask, risk_labels=None, severity_labels=None, score_labels=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        risk_logits = self.risk_classifier(pooled_output)
        severity_logits = self.severity_classifier(pooled_output)
        score_predictions = self.score_regressor(pooled_output).squeeze(-1)
        
        loss = None
        if risk_labels is not None and severity_labels is not None and score_labels is not None:
            risk_loss = torch.nn.functional.cross_entropy(risk_logits, risk_labels)
            severity_loss = torch.nn.functional.cross_entropy(severity_logits, severity_labels)
            score_loss = torch.nn.functional.mse_loss(score_predictions, score_labels)
            loss = risk_loss + severity_loss + score_loss
        
        return {
            'loss': loss,
            'risk_logits': risk_logits,
            'severity_logits': severity_logits,
            'score_predictions': score_predictions
        }

def train_transformer_models(df, model_name="distilbert-base-uncased"):
    """Train transformer models for all tasks"""
    print(f"\n🚀 Training Transformer Models: {model_name}")
    
    # Download NLTK data
    download_nltk_data()
    
    # Prepare data
    le_risk = LabelEncoder()
    le_sev = LabelEncoder()
    df['Risk_Type_enc'] = le_risk.fit_transform(df['Risk_Type'])
    df['Severity_enc'] = le_sev.fit_transform(df['Severity'])
    
    # Train/test split
    X_train, X_test, y_risk_train, y_risk_test = train_test_split(
        df['text'], df['Risk_Type_enc'], test_size=0.2, random_state=42, stratify=df['Risk_Type_enc']
    )
    
    _, _, y_sev_train, y_sev_test = train_test_split(
        df['text'], df['Severity_enc'], test_size=0.2, random_state=42, stratify=df['Severity_enc']
    )
    
    _, _, y_score_train, y_score_test = train_test_split(
        df['text'], df['Risk_Score'], test_size=0.2, random_state=42
    )
    
    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=256)
    test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=256)
    
    # Create datasets
    train_dataset = MultiTaskRiskDataset(train_encodings, list(y_risk_train), list(y_sev_train), list(y_score_train))
    test_dataset = MultiTaskRiskDataset(test_encodings, list(y_risk_test), list(y_sev_test), list(y_score_test))
    
    # Initialize model
    model = MultiTaskModel(model_name, len(le_risk.classes_), len(le_sev.classes_))
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f'./results_{model_name}',
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f'./logs_{model_name}',
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2
    )
    
    # Custom trainer for multi-task learning
    class MultiTaskTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            outputs = model(**inputs)
            loss = outputs['loss']
            return (loss, outputs) if return_outputs else loss
        
        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                outputs = model(**inputs)
                loss = outputs['loss']
                logits = {
                    'risk_logits': outputs['risk_logits'],
                    'severity_logits': outputs['severity_logits'],
                    'score_predictions': outputs['score_predictions']
                }
            return (loss, logits, None)
    
    # Train
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    print("🔄 Starting transformer training...")
    trainer.train()
    
    # Evaluate
    print("📊 Evaluating transformer model...")
    eval_results = trainer.evaluate()
    
    # Make predictions
    predictions = trainer.predict(test_dataset)
    
    # Extract predictions
    risk_preds = torch.argmax(torch.tensor(predictions.predictions['risk_logits']), dim=1).numpy()
    severity_preds = torch.argmax(torch.tensor(predictions.predictions['severity_logits']), dim=1).numpy()
    score_preds = predictions.predictions['score_predictions']
    
    # Calculate metrics
    risk_f1 = f1_score(y_risk_test, risk_preds, average='weighted')
    severity_f1 = f1_score(y_sev_test, severity_preds, average='weighted')
    score_rmse = np.sqrt(mean_squared_error(y_score_test, score_preds))
    
    print(f"✅ Transformer Risk Type F1: {risk_f1:.4f}")
    print(f"✅ Transformer Severity F1: {severity_f1:.4f}")
    print(f"✅ Transformer Risk Score RMSE: {score_rmse:.4f}")
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'le_risk': le_risk,
        'le_sev': le_sev,
        'risk_f1': risk_f1,
        'severity_f1': severity_f1,
        'score_rmse': score_rmse,
        'predictions': {
            'risk_preds': risk_preds,
            'severity_preds': severity_preds,
            'score_preds': score_preds
        },
        'test_labels': {
            'risk': y_risk_test,
            'severity': y_sev_test,
            'score': y_score_test
        }
    }

# =================================================================
# 5. Model Comparison & Analysis
# =================================================================

def compare_models(baseline_results, transformer_results):
    """Compare baseline and transformer models"""
    print("\n📊 MODEL COMPARISON ANALYSIS")
    print("="*60)
    
    comparison = {
        'Risk Type Classification': {
            'Baseline (RandomForest)': baseline_results['risk_f1'],
            'Transformer (DistilBERT)': transformer_results['risk_f1'],
            'Improvement': transformer_results['risk_f1'] - baseline_results['risk_f1']
        },
        'Severity Classification': {
            'Baseline (RandomForest)': baseline_results['sev_f1'],
            'Transformer (DistilBERT)': transformer_results['severity_f1'],
            'Improvement': transformer_results['severity_f1'] - baseline_results['sev_f1']
        },
        'Risk Score Regression': {
            'Baseline (RandomForest)': baseline_results['score_rmse'],
            'Transformer (DistilBERT)': transformer_results['score_rmse'],
            'Improvement': baseline_results['score_rmse'] - transformer_results['score_rmse']
        }
    }
    
    for task, metrics in comparison.items():
        print(f"\n{task}:")
        print(f"  Baseline: {metrics['Baseline (RandomForest)']:.4f}")
        print(f"  Transformer: {metrics['Transformer (DistilBERT)']:.4f}")
        print(f"  Improvement: {metrics['Improvement']:+.4f}")
    
    return comparison

def generate_detailed_reports(baseline_results, transformer_results, le_risk, le_sev):
    """Generate detailed classification reports"""
    print("\n📋 DETAILED CLASSIFICATION REPORTS")
    print("="*60)
    
    # Risk Type Reports
    print("\n🔍 Risk Type Classification:")
    print("\nBaseline RandomForest:")
    print(classification_report(
        baseline_results['y_risk_test'], 
        baseline_results['y_risk_pred'], 
        target_names=le_risk.classes_
    ))
    
    print("\nTransformer DistilBERT:")
    print(classification_report(
        transformer_results['test_labels']['risk'],
        transformer_results['predictions']['risk_preds'],
        target_names=le_risk.classes_
    ))
    
    # Severity Reports
    print("\n🔍 Severity Classification:")
    print("\nBaseline RandomForest:")
    print(classification_report(
        baseline_results['y_sev_test'], 
        baseline_results['y_sev_pred'], 
        target_names=le_sev.classes_
    ))
    
    print("\nTransformer DistilBERT:")
    print(classification_report(
        transformer_results['test_labels']['severity'],
        transformer_results['predictions']['severity_preds'],
        target_names=le_sev.classes_
    ))

# =================================================================
# 6. Model Deployment & Inference Pipeline
# =================================================================

class RiskAnalysisInference:
    """Deployment-ready inference pipeline"""
    
    def __init__(self, baseline_models, transformer_models):
        self.baseline_models = baseline_models
        self.transformer_models = transformer_models
        self.model = transformer_models['model']
        self.tokenizer = transformer_models['tokenizer']
        self.le_risk = transformer_models['le_risk']
        self.le_sev = transformer_models['le_sev']
    
    def predict_single(self, text, use_transformer=True):
        """Predict risk for a single text"""
        if use_transformer:
            return self._predict_transformer(text)
        else:
            return self._predict_baseline(text)
    
    def _predict_transformer(self, text):
        """Transformer prediction"""
        inputs = self.tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            risk_pred = torch.argmax(outputs['risk_logits'], dim=1).item()
            severity_pred = torch.argmax(outputs['severity_logits'], dim=1).item()
            score_pred = outputs['score_predictions'].item()
        
        return {
            'risk_type': self.le_risk.inverse_transform([risk_pred])[0],
            'severity': self.le_sev.inverse_transform([severity_pred])[0],
            'risk_score': float(score_pred)
        }
    
    def _predict_baseline(self, text):
        """Baseline prediction"""
        # Process text
        processed_text = advanced_text_preprocessing(text)
        
        # Vectorize
        text_vec = self.baseline_models['vectorizer'].transform([processed_text])
        
        # Additional features
        text_length = len(text)
        word_count = len(text.split())
        has_affected_nodes = False  # Would need to be determined from context
        
        additional_features = np.array([[text_length, word_count, has_affected_nodes]])
        
        # Combine features
        from scipy.sparse import hstack
        combined_features = hstack([text_vec, additional_features])
        
        # Predict
        risk_pred = self.baseline_models['clf_risk'].predict(combined_features)[0]
        severity_pred = self.baseline_models['clf_sev'].predict(combined_features)[0]
        score_pred = self.baseline_models['reg_score'].predict(combined_features)[0]
        
        return {
            'risk_type': self.baseline_models['le_risk'].inverse_transform([risk_pred])[0],
            'severity': self.baseline_models['le_sev'].inverse_transform([severity_pred])[0],
            'risk_score': float(score_pred)
        }
    
    def predict_batch(self, texts, use_transformer=True):
        """Predict risk for multiple texts"""
        results = []
        for text in texts:
            results.append(self.predict_single(text, use_transformer))
        return results
    
    def save_models(self, path_prefix="risk_analysis_models"):
        """Save all models for deployment"""
        print(f"💾 Saving models to {path_prefix}_*")
        
        # Save baseline models
        joblib.dump(self.baseline_models, f"{path_prefix}_baseline.joblib")
        
        # Save transformer models
        torch.save(self.model.state_dict(), f"{path_prefix}_transformer.pt")
        self.tokenizer.save_pretrained(f"{path_prefix}_tokenizer")
        
        # Save encoders
        joblib.dump(self.le_risk, f"{path_prefix}_le_risk.joblib")
        joblib.dump(self.le_sev, f"{path_prefix}_le_sev.joblib")
        
        print("✅ Models saved successfully!")

# =================================================================
# 7. Main Execution Pipeline
# =================================================================

def main():
    """Main execution pipeline"""
    print("🚀 ENHANCED RISK ANALYSIS ML PIPELINE")
    print("="*60)
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Train baseline models
    baseline_models, baseline_results = train_baseline_models(df)
    
    # Train transformer models
    transformer_results = train_transformer_models(df)
    
    # Compare models
    comparison = compare_models(baseline_results, transformer_results)
    
    # Generate detailed reports
    generate_detailed_reports(baseline_results, transformer_results, 
                           transformer_results['le_risk'], transformer_results['le_sev'])
    
    # Create inference pipeline
    inference_pipeline = RiskAnalysisInference(baseline_models, transformer_results)
    
    # Save models
    inference_pipeline.save_models()
    
    # Test inference
    print("\n🧪 TESTING INFERENCE PIPELINE")
    print("="*40)
    
    sample_texts = [
        "Tesla announces major expansion in Indian market, increasing competitive pressure",
        "Supply chain disruption due to semiconductor shortage affecting production",
        "New government policy on electric vehicles creates regulatory uncertainty"
    ]
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\nSample {i}: {text}")
        
        # Baseline prediction
        baseline_pred = inference_pipeline.predict_single(text, use_transformer=False)
        print(f"Baseline: {baseline_pred}")
        
        # Transformer prediction
        transformer_pred = inference_pipeline.predict_single(text, use_transformer=True)
        print(f"Transformer: {transformer_pred}")
    
    print("\n✅ PIPELINE COMPLETED SUCCESSFULLY!")
    return inference_pipeline, comparison

if __name__ == "__main__":
    inference_pipeline, comparison = main()
