# Enhanced Risk Analysis ML Pipeline V2 - Improved Performance
# =============================================================

import json
import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_squared_error, classification_report
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# =================================================================
# 1. Enhanced Data Loading & Preprocessing
# =================================================================

def load_and_preprocess_data(json_file="tata_motors_risk_analysis.json"):
    """Load and preprocess the risk analysis dataset with enhanced features"""
    print("Loading and preprocessing data...")
    
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
    
    # Enhanced feature engineering
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    df['has_affected_nodes'] = df['Affected_Nodes'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)
    df['affected_nodes_count'] = df['Affected_Nodes'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    # Text complexity features
    df['avg_word_length'] = df['text'].str.split().apply(lambda x: np.mean([len(word) for word in x]) if x else 0)
    df['sentence_count'] = df['text'].str.count(r'[.!?]+') + 1
    df['exclamation_count'] = df['text'].str.count(r'!')
    df['question_count'] = df['text'].str.count(r'\?')
    
    print(f"Final dataset size: {len(df)}")
    return df

# =================================================================
# 2. Enhanced Baseline Models with Class Weights
# =================================================================

def train_enhanced_baseline_models(df):
    """Train enhanced baseline models with class weights and feature engineering"""
    print("\nTraining Enhanced Baseline Models with Class Weights...")
    
    # Prepare features
    X_text = df['text']
    
    # Enhanced additional features
    feature_cols = ['text_length', 'word_count', 'has_affected_nodes', 'affected_nodes_count',
                   'avg_word_length', 'sentence_count', 'exclamation_count', 'question_count']
    X_additional = df[feature_cols].astype(float).values
    
    # Encode targets
    le_risk = LabelEncoder()
    le_sev = LabelEncoder()
    df['Risk_Type_enc'] = le_risk.fit_transform(df['Risk_Type'])
    df['Severity_enc'] = le_sev.fit_transform(df['Severity'])
    
    # Compute class weights for severity (handling imbalance)
    classes = np.unique(df['Severity_enc'])
    class_weights = compute_class_weight('balanced', classes=classes, y=df['Severity_enc'])
    class_weights_dict = dict(zip(classes, class_weights))
    print(f"Class weights for severity: {class_weights_dict}")
    
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
    
    # Convert to sparse matrix
    from scipy.sparse import csr_matrix, hstack
    X_add_train = csr_matrix(X_add_train)
    X_add_test = csr_matrix(X_add_test)
    
    # Enhanced TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3), min_df=2, max_df=0.95)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Combine features
    X_train_combined = hstack([X_train_vec, X_add_train])
    X_test_combined = hstack([X_test_vec, X_add_test])
    
    # Train models with enhanced parameters
    print("Training Risk Type classifier...")
    clf_risk = RandomForestClassifier(
        n_estimators=300, 
        max_depth=20, 
        min_samples_split=5, 
        min_samples_leaf=2,
        random_state=42, 
        n_jobs=-1
    )
    clf_risk.fit(X_train_combined, y_risk_train)
    y_risk_pred = clf_risk.predict(X_test_combined)
    risk_f1 = f1_score(y_risk_test, y_risk_pred, average='weighted')
    
    print("Training Severity classifier with class weights...")
    clf_sev = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',  # Use balanced class weights
        random_state=42,
        n_jobs=-1
    )
    clf_sev.fit(X_train_combined, y_sev_train)
    y_sev_pred = clf_sev.predict(X_test_combined)
    sev_f1 = f1_score(y_sev_test, y_sev_pred, average='weighted')
    
    print("Training Risk Score regressor...")
    reg_score = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    reg_score.fit(X_train_combined, y_score_train)
    y_score_pred = reg_score.predict(X_test_combined)
    rmse = np.sqrt(mean_squared_error(y_score_test, y_score_pred))
    
    models = {
        'vectorizer': vectorizer,
        'clf_risk': clf_risk,
        'clf_sev': clf_sev,
        'reg_score': reg_score,
        'le_risk': le_risk,
        'le_sev': le_sev,
        'feature_cols': feature_cols
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
    
    print(f"Risk Type F1: {risk_f1:.4f}")
    print(f"Severity F1: {sev_f1:.4f}")
    print(f"Risk Score RMSE: {rmse:.4f}")
    
    return models, results

# =================================================================
# 3. Enhanced Transformer Training for Severity
# =================================================================

def train_severity_transformer(df, model_name="distilbert-base-uncased"):
    """Train transformer specifically for severity classification with class weights"""
    print(f"\nTraining Severity Transformer: {model_name}")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
        from torch.utils.data import Dataset
        from torch.nn import CrossEntropyLoss
        import torch.nn as nn
        
        # Prepare data for severity classification
        le_sev = LabelEncoder()
        df['Severity_enc'] = le_sev.fit_transform(df['Severity'])
        
        # Compute class weights
        classes = np.unique(df['Severity_enc'])
        class_weights = compute_class_weight('balanced', classes=classes, y=df['Severity_enc'])
        class_weights_tensor = torch.FloatTensor(class_weights)
        
        print(f"Severity classes: {le_sev.classes_}")
        print(f"Class weights: {class_weights}")
        
        X_train, X_test, y_sev_train, y_sev_test = train_test_split(
            df['text'], df['Severity_enc'], test_size=0.2, random_state=42, stratify=df['Severity_enc']
        )
        
        # Tokenize
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=256)
        test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=256)
        
        # Dataset class with class weights
        class SeverityDataset(Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
            
            def __len__(self):
                return len(self.labels)
            
            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item
        
        train_dataset = SeverityDataset(train_encodings, list(y_sev_train))
        test_dataset = SeverityDataset(test_encodings, list(y_sev_test))
        
        # Model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(le_sev.classes_)
        )
        
        # Custom trainer with weighted loss
        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get("logits")
                
                # Use weighted cross entropy loss
                loss_fct = CrossEntropyLoss(weight=class_weights_tensor)
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
                
                return (loss, outputs) if return_outputs else loss
        
        # Enhanced training arguments
        training_args = TrainingArguments(
            output_dir=f'./results_severity_{model_name}',
            num_train_epochs=5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=100,
            weight_decay=0.01,
            learning_rate=3e-5,
            logging_dir=f'./logs_severity_{model_name}',
            logging_steps=50,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            save_total_limit=2
        )
        
        # Trainer
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset
        )
        
        print("Starting severity transformer training...")
        trainer.train()
        
        # Evaluate
        predictions = trainer.predict(test_dataset)
        y_pred = torch.argmax(torch.tensor(predictions.predictions), axis=1).numpy()
        severity_f1 = f1_score(y_sev_test, y_pred, average='weighted')
        
        print(f"Severity Transformer F1: {severity_f1:.4f}")
        
        return {
            'model': model,
            'tokenizer': tokenizer,
            'le_sev': le_sev,
            'severity_f1': severity_f1,
            'predictions': y_pred,
            'test_labels': y_sev_test
        }
        
    except ImportError:
        print("Transformers not available - skipping transformer training")
        return None
    except Exception as e:
        print(f"Severity transformer training failed: {e}")
        return None

# =================================================================
# 4. Transformer Regression for Risk Score
# =================================================================

def train_risk_score_transformer(df, model_name="distilbert-base-uncased"):
    """Train transformer for risk score regression"""
    print(f"\nTraining Risk Score Transformer: {model_name}")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
        from torch.utils.data import Dataset
        from torch.nn import MSELoss
        
        # Prepare data for risk score regression
        X_train, X_test, y_score_train, y_score_test = train_test_split(
            df['text'], df['Risk_Score'], test_size=0.2, random_state=42
        )
        
        # Tokenize
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=256)
        test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=256)
        
        # Dataset class for regression
        class RiskScoreDataset(Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
            
            def __len__(self):
                return len(self.labels)
            
            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
                return item
        
        train_dataset = RiskScoreDataset(train_encodings, list(y_score_train))
        test_dataset = RiskScoreDataset(test_encodings, list(y_score_test))
        
        # Model for regression
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=1,
            problem_type="regression"
        )
        
        # Training arguments for regression
        training_args = TrainingArguments(
            output_dir=f'./results_risk_score_{model_name}',
            num_train_epochs=5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=100,
            weight_decay=0.01,
            learning_rate=3e-5,
            logging_dir=f'./logs_risk_score_{model_name}',
            logging_steps=50,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=2
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset
        )
        
        print("Starting risk score transformer training...")
        trainer.train()
        
        # Evaluate
        predictions = trainer.predict(test_dataset)
        y_pred = predictions.predictions.flatten()
        rmse = np.sqrt(mean_squared_error(y_score_test, y_pred))
        
        print(f"Risk Score Transformer RMSE: {rmse:.4f}")
        
        return {
            'model': model,
            'tokenizer': tokenizer,
            'rmse': rmse,
            'predictions': y_pred,
            'test_labels': y_score_test
        }
        
    except ImportError:
        print("Transformers not available - skipping transformer training")
        return None
    except Exception as e:
        print(f"Risk score transformer training failed: {e}")
        return None

# =================================================================
# 5. Ensemble Methods
# =================================================================

def create_ensemble_predictions(baseline_results, severity_transformer, risk_score_transformer):
    """Create ensemble predictions combining baseline and transformer models"""
    print("\nCreating Ensemble Predictions...")
    
    ensemble_results = {}
    
    # For severity - combine baseline and transformer
    if severity_transformer:
        baseline_sev_pred = baseline_results['y_sev_pred']
        transformer_sev_pred = severity_transformer['predictions']
        
        # Simple voting ensemble
        ensemble_sev_pred = []
        for i in range(len(baseline_sev_pred)):
            # Weighted average (0.6 transformer, 0.4 baseline)
            ensemble_pred = int(0.6 * transformer_sev_pred[i] + 0.4 * baseline_sev_pred[i])
            ensemble_sev_pred.append(ensemble_pred)
        
        ensemble_sev_f1 = f1_score(baseline_results['y_sev_test'], ensemble_sev_pred, average='weighted')
        print(f"Ensemble Severity F1: {ensemble_sev_f1:.4f}")
        ensemble_results['severity_f1'] = ensemble_sev_f1
        ensemble_results['severity_predictions'] = ensemble_sev_pred
    
    # For risk score - combine baseline and transformer
    if risk_score_transformer:
        baseline_score_pred = baseline_results['y_score_pred']
        transformer_score_pred = risk_score_transformer['predictions']
        
        # Weighted average ensemble
        ensemble_score_pred = 0.6 * transformer_score_pred + 0.4 * baseline_score_pred
        ensemble_rmse = np.sqrt(mean_squared_error(baseline_results['y_score_test'], ensemble_score_pred))
        print(f"Ensemble Risk Score RMSE: {ensemble_rmse:.4f}")
        ensemble_results['score_rmse'] = ensemble_rmse
        ensemble_results['score_predictions'] = ensemble_score_pred
    
    return ensemble_results

# =================================================================
# 6. Live News API Integration
# =================================================================

def fetch_live_news(api_key, query="Tata Motors", num_articles=10):
    """Fetch live news from News API"""
    try:
        import requests
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'apiKey': api_key,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': num_articles
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if data['status'] == 'ok':
            articles = []
            for article in data['articles']:
                articles.append({
                    'Title': article['title'],
                    'Explanation': article['description'] or '',
                    'Affected_Nodes': [],
                    'Risk_Type': None,
                    'Severity': None,
                    'Risk_Score': None,
                    'publishedAt': article['publishedAt'],
                    'url': article['url']
                })
            return articles
        else:
            print(f"News API error: {data.get('message', 'Unknown error')}")
            return []
            
    except Exception as e:
        print(f"Error fetching live news: {e}")
        return []

def process_live_news(articles, analyzer):
    """Process live news articles with risk analysis"""
    print(f"\nProcessing {len(articles)} live news articles...")
    
    results = []
    for i, article in enumerate(articles):
        text = article['Title'] + ' ' + article['Explanation']
        
        # Predict risk
        prediction = analyzer.predict_risk(text, model_type="baseline")
        
        # Update article with predictions
        article['Risk_Type'] = prediction['risk_type']
        article['Severity'] = prediction['severity']
        article['Risk_Score'] = prediction['risk_score']
        
        results.append(article)
        
        print(f"Article {i+1}: {article['Title'][:50]}...")
        print(f"  Risk: {prediction['risk_type']}, Severity: {prediction['severity']}, Score: {prediction['risk_score']:.2f}")
    
    return results

# =================================================================
# 7. Enhanced Risk Analyzer
# =================================================================

class EnhancedRiskAnalyzer:
    """Enhanced risk analyzer with ensemble methods"""
    
    def __init__(self, baseline_models, severity_transformer=None, risk_score_transformer=None):
        self.baseline_models = baseline_models
        self.severity_transformer = severity_transformer
        self.risk_score_transformer = risk_score_transformer
    
    def predict_risk(self, text, model_type="ensemble"):
        """Predict risk with ensemble methods"""
        if model_type == "ensemble":
            return self._predict_ensemble(text)
        elif model_type == "transformer":
            return self._predict_transformer(text)
        else:
            return self._predict_baseline(text)
    
    def _predict_baseline(self, text):
        """Baseline prediction"""
        try:
            # Vectorize text
            text_vec = self.baseline_models['vectorizer'].transform([text])
            
            # Additional features
            text_length = len(text)
            word_count = len(text.split())
            has_affected_nodes = False
            affected_nodes_count = 0
            avg_word_length = np.mean([len(word) for word in text.split()]) if text.split() else 0
            sentence_count = text.count('.') + text.count('!') + text.count('?') + 1
            exclamation_count = text.count('!')
            question_count = text.count('?')
            
            additional_features = np.array([[
                text_length, word_count, has_affected_nodes, affected_nodes_count,
                avg_word_length, sentence_count, exclamation_count, question_count
            ]])
            
            # Combine features
            from scipy.sparse import hstack, csr_matrix
            combined_features = hstack([text_vec, csr_matrix(additional_features)])
            
            # Predict
            risk_pred = self.baseline_models['clf_risk'].predict(combined_features)[0]
            severity_pred = self.baseline_models['clf_sev'].predict(combined_features)[0]
            score_pred = self.baseline_models['reg_score'].predict(combined_features)[0]
            
            return {
                'risk_type': self.baseline_models['le_risk'].inverse_transform([risk_pred])[0],
                'severity': self.baseline_models['le_sev'].inverse_transform([severity_pred])[0],
                'risk_score': float(score_pred),
                'model_type': 'baseline'
            }
            
        except Exception as e:
            print(f"Error in baseline prediction: {e}")
            return {
                'risk_type': 'None',
                'severity': 'Low',
                'risk_score': 1.0,
                'model_type': 'baseline'
            }
    
    def _predict_transformer(self, text):
        """Transformer prediction"""
        try:
            # Risk type (baseline for now)
            baseline_pred = self._predict_baseline(text)
            
            # Severity (transformer if available)
            if self.severity_transformer:
                inputs = self.severity_transformer['tokenizer'](
                    text, truncation=True, padding=True, max_length=256, return_tensors='pt'
                )
                with torch.no_grad():
                    outputs = self.severity_transformer['model'](**inputs)
                    severity_pred = torch.argmax(outputs.logits, dim=1).item()
                severity = self.severity_transformer['le_sev'].inverse_transform([severity_pred])[0]
            else:
                severity = baseline_pred['severity']
            
            # Risk score (transformer if available)
            if self.risk_score_transformer:
                inputs = self.risk_score_transformer['tokenizer'](
                    text, truncation=True, padding=True, max_length=256, return_tensors='pt'
                )
                with torch.no_grad():
                    outputs = self.risk_score_transformer['model'](**inputs)
                    score_pred = outputs.logits.item()
            else:
                score_pred = baseline_pred['risk_score']
            
            return {
                'risk_type': baseline_pred['risk_type'],
                'severity': severity,
                'risk_score': float(score_pred),
                'model_type': 'transformer'
            }
            
        except Exception as e:
            print(f"Error in transformer prediction: {e}")
            return self._predict_baseline(text)
    
    def _predict_ensemble(self, text):
        """Ensemble prediction combining baseline and transformer"""
        baseline_pred = self._predict_baseline(text)
        transformer_pred = self._predict_transformer(text)
        
        # Ensemble logic
        risk_type = baseline_pred['risk_type']  # Use baseline for risk type
        
        # Weighted ensemble for severity and score
        if self.severity_transformer and self.risk_score_transformer:
            severity = transformer_pred['severity']  # Prefer transformer
            risk_score = 0.7 * transformer_pred['risk_score'] + 0.3 * baseline_pred['risk_score']
        else:
            severity = baseline_pred['severity']
            risk_score = baseline_pred['risk_score']
        
        return {
            'risk_type': risk_type,
            'severity': severity,
            'risk_score': float(risk_score),
            'model_type': 'ensemble'
        }
    
    def save_models(self, path_prefix="enhanced_risk_models"):
        """Save all models"""
        print(f"Saving enhanced models to {path_prefix}_*")
        
        # Save baseline models
        joblib.dump(self.baseline_models, f"{path_prefix}_baseline.joblib")
        
        # Save transformer models
        if self.severity_transformer:
            torch.save(self.severity_transformer['model'].state_dict(), f"{path_prefix}_severity.pt")
            self.severity_transformer['tokenizer'].save_pretrained(f"{path_prefix}_severity_tokenizer")
            joblib.dump(self.severity_transformer['le_sev'], f"{path_prefix}_le_sev.joblib")
        
        if self.risk_score_transformer:
            torch.save(self.risk_score_transformer['model'].state_dict(), f"{path_prefix}_risk_score.pt")
            self.risk_score_transformer['tokenizer'].save_pretrained(f"{path_prefix}_risk_score_tokenizer")
        
        print("Enhanced models saved successfully!")

# =================================================================
# 8. Main Execution Pipeline
# =================================================================

def main():
    """Enhanced main execution pipeline"""
    print("ENHANCED RISK ANALYSIS ML PIPELINE V2")
    print("="*60)
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Train enhanced baseline models
    baseline_models, baseline_results = train_enhanced_baseline_models(df)
    
    # Train transformer models
    severity_transformer = train_severity_transformer(df)
    risk_score_transformer = train_risk_score_transformer(df)
    
    # Create ensemble predictions
    ensemble_results = create_ensemble_predictions(baseline_results, severity_transformer, risk_score_transformer)
    
    # Create enhanced analyzer
    analyzer = EnhancedRiskAnalyzer(baseline_models, severity_transformer, risk_score_transformer)
    
    # Save models
    analyzer.save_models()
    
    # Test inference
    print("\nTESTING ENHANCED INFERENCE PIPELINE")
    print("="*50)
    
    sample_texts = [
        "Tesla announces major expansion in Indian market, increasing competitive pressure",
        "Supply chain disruption due to semiconductor shortage affecting production",
        "New government policy on electric vehicles creates regulatory uncertainty"
    ]
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\nSample {i}: {text}")
        
        # Baseline prediction
        baseline_pred = analyzer.predict_risk(text, model_type="baseline")
        print(f"Baseline: {baseline_pred}")
        
        # Ensemble prediction
        ensemble_pred = analyzer.predict_risk(text, model_type="ensemble")
        print(f"Ensemble: {ensemble_pred}")
    
    # Test live news integration (if API key provided)
    print("\nLIVE NEWS INTEGRATION TEST")
    print("="*40)
    print("To test live news integration, provide your News API key:")
    print("analyzer.process_live_news(articles, analyzer)")
    
    print("\nENHANCED PIPELINE COMPLETED SUCCESSFULLY!")
    return analyzer, ensemble_results

if __name__ == "__main__":
    analyzer, ensemble_results = main()
