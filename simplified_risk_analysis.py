# Simplified Risk Analysis ML Pipeline
# ====================================

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
import warnings
warnings.filterwarnings('ignore')

# =================================================================
# 1. Data Loading & Preprocessing
# =================================================================

def load_and_preprocess_data(json_file="tata_motors_risk_analysis.json"):
    """Load and preprocess the risk analysis dataset"""
    print("Loading and preprocessing data...")
    
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    print(f"Original dataset size: {len(df)}")
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['Title', 'Explanation'])
    print(f"After deduplication: {len(df)}")
    
    # Create text features
    df['text'] = df['Title'].fillna('') + ' ' + df['Explanation'].fillna('')
    df = df[df['text'].str.strip() != '']
    
    # Add additional features
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    df['has_affected_nodes'] = df['Affected_Nodes'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)
    
    print(f"Final dataset size: {len(df)}")
    return df

# =================================================================
# 2. Enhanced Baseline Models
# =================================================================

def train_baseline_models(df):
    """Train enhanced baseline models"""
    print("\nTraining Enhanced Baseline Models...")
    
    # Prepare features
    X_text = df['text']
    
    # Additional features
    X_additional = df[['text_length', 'word_count', 'has_affected_nodes']].astype(float).values
    
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
    
    # Convert to sparse matrix
    from scipy.sparse import csr_matrix
    X_add_train = csr_matrix(X_add_train)
    X_add_test = csr_matrix(X_add_test)
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Combine text and additional features
    from scipy.sparse import hstack
    X_train_combined = hstack([X_train_vec, X_add_train])
    X_test_combined = hstack([X_test_vec, X_add_test])
    
    # Train models
    print("Training Risk Type classifier...")
    clf_risk = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf_risk.fit(X_train_combined, y_risk_train)
    y_risk_pred = clf_risk.predict(X_test_combined)
    risk_f1 = f1_score(y_risk_test, y_risk_pred, average='weighted')
    
    print("Training Severity classifier...")
    clf_sev = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf_sev.fit(X_train_combined, y_sev_train)
    y_sev_pred = clf_sev.predict(X_test_combined)
    sev_f1 = f1_score(y_sev_test, y_sev_pred, average='weighted')
    
    print("Training Risk Score regressor...")
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
    
    print(f"Risk Type F1: {risk_f1:.4f}")
    print(f"Severity F1: {sev_f1:.4f}")
    print(f"Risk Score RMSE: {rmse:.4f}")
    
    return models, results

# =================================================================
# 3. Simple Transformer Training (Optional)
# =================================================================

def train_simple_transformer(df, model_name="distilbert-base-uncased"):
    """Simple transformer training with basic setup"""
    print(f"\nTraining Simple Transformer: {model_name}")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
        from torch.utils.data import Dataset
        
        # Prepare data for risk type classification only
        le_risk = LabelEncoder()
        df['Risk_Type_enc'] = le_risk.fit_transform(df['Risk_Type'])
        
        X_train, X_test, y_risk_train, y_risk_test = train_test_split(
            df['text'], df['Risk_Type_enc'], test_size=0.2, random_state=42, stratify=df['Risk_Type_enc']
        )
        
        # Tokenize
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=256)
        test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=256)
        
        # Simple dataset class
        class SimpleRiskDataset(Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
            
            def __len__(self):
                return len(self.labels)
            
            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item
        
        train_dataset = SimpleRiskDataset(train_encodings, list(y_risk_train))
        test_dataset = SimpleRiskDataset(test_encodings, list(y_risk_test))
        
        # Model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(le_risk.classes_)
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f'./results_{model_name}',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            logging_dir=f'./logs_{model_name}',
            logging_steps=50,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset
        )
        
        print("Starting transformer training...")
        trainer.train()
        
        # Evaluate
        predictions = trainer.predict(test_dataset)
        y_pred = torch.argmax(torch.tensor(predictions.predictions), axis=1).numpy()
        risk_f1 = f1_score(y_risk_test, y_pred, average='weighted')
        
        print(f"Transformer Risk Type F1: {risk_f1:.4f}")
        
        return {
            'model': model,
            'tokenizer': tokenizer,
            'le_risk': le_risk,
            'risk_f1': risk_f1,
            'predictions': y_pred,
            'test_labels': y_risk_test
        }
        
    except ImportError:
        print("Transformers not available - skipping transformer training")
        return None
    except Exception as e:
        print(f"Transformer training failed: {e}")
        return None

# =================================================================
# 4. Model Comparison
# =================================================================

def compare_models(baseline_results, transformer_results):
    """Compare baseline and transformer models"""
    print("\nMODEL COMPARISON ANALYSIS")
    print("="*60)
    
    if transformer_results is None:
        print("Transformer results not available")
        return
    
    comparison = {
        'Risk Type Classification': {
            'Baseline (RandomForest)': baseline_results['risk_f1'],
            'Transformer (DistilBERT)': transformer_results['risk_f1'],
            'Improvement': transformer_results['risk_f1'] - baseline_results['risk_f1']
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
    print("\nDETAILED CLASSIFICATION REPORTS")
    print("="*60)
    
    # Risk Type Reports
    print("\nRisk Type Classification:")
    print("\nBaseline RandomForest:")
    print(classification_report(
        baseline_results['y_risk_test'], 
        baseline_results['y_risk_pred'], 
        target_names=le_risk.classes_
    ))
    
    if transformer_results:
        print("\nTransformer DistilBERT:")
        print(classification_report(
            transformer_results['test_labels'],
            transformer_results['predictions'],
            target_names=le_risk.classes_
        ))

# =================================================================
# 5. Model Deployment
# =================================================================

class SimpleRiskAnalyzer:
    """Simple deployment-ready risk analyzer"""
    
    def __init__(self, baseline_models, transformer_models=None):
        self.baseline_models = baseline_models
        self.transformer_models = transformer_models
    
    def predict_risk(self, text, model_type="baseline"):
        """Predict risk for a single text"""
        if model_type == "transformer" and self.transformer_models:
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
        """Transformer prediction (simplified)"""
        try:
            inputs = self.transformer_models['tokenizer'](
                text, truncation=True, padding=True, max_length=256, return_tensors='pt'
            )
            
            with torch.no_grad():
                outputs = self.transformer_models['model'](**inputs)
                risk_pred = torch.argmax(outputs.logits, dim=1).item()
            
            return {
                'risk_type': self.transformer_models['le_risk'].inverse_transform([risk_pred])[0],
                'severity': 'Medium',  # Placeholder
                'risk_score': 5.0,  # Placeholder
                'model_type': 'transformer'
            }
            
        except Exception as e:
            print(f"Error in transformer prediction: {e}")
            return self._predict_baseline(text)
    
    def save_models(self, path_prefix="simple_risk_models"):
        """Save models for deployment"""
        print(f"Saving models to {path_prefix}_*")
        
        # Save baseline models
        joblib.dump(self.baseline_models, f"{path_prefix}_baseline.joblib")
        
        if self.transformer_models:
            torch.save(self.transformer_models['model'].state_dict(), f"{path_prefix}_transformer.pt")
            self.transformer_models['tokenizer'].save_pretrained(f"{path_prefix}_tokenizer")
            joblib.dump(self.transformer_models['le_risk'], f"{path_prefix}_le_risk.joblib")
        
        print("Models saved successfully!")

# =================================================================
# 6. Main Execution Pipeline
# =================================================================

def main():
    """Main execution pipeline"""
    print("SIMPLIFIED RISK ANALYSIS ML PIPELINE")
    print("="*60)
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Train baseline models
    baseline_models, baseline_results = train_baseline_models(df)
    
    # Train transformer models (optional)
    transformer_results = train_simple_transformer(df)
    
    # Compare models
    comparison = compare_models(baseline_results, transformer_results)
    
    # Generate detailed reports
    generate_detailed_reports(baseline_results, transformer_results, 
                           baseline_models['le_risk'], baseline_models['le_sev'])
    
    # Create inference pipeline
    inference_pipeline = SimpleRiskAnalyzer(baseline_models, transformer_results)
    
    # Save models
    inference_pipeline.save_models()
    
    # Test inference
    print("\nTESTING INFERENCE PIPELINE")
    print("="*40)
    
    sample_texts = [
        "Tesla announces major expansion in Indian market, increasing competitive pressure",
        "Supply chain disruption due to semiconductor shortage affecting production",
        "New government policy on electric vehicles creates regulatory uncertainty"
    ]
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\nSample {i}: {text}")
        
        # Baseline prediction
        baseline_pred = inference_pipeline.predict_risk(text, model_type="baseline")
        print(f"Baseline: {baseline_pred}")
        
        # Transformer prediction (if available)
        if transformer_results:
            transformer_pred = inference_pipeline.predict_risk(text, model_type="transformer")
            print(f"Transformer: {transformer_pred}")
    
    print("\nSIMPLIFIED PIPELINE COMPLETED SUCCESSFULLY!")
    return inference_pipeline, comparison

if __name__ == "__main__":
    inference_pipeline, comparison = main()

