# Risk Analysis ML Pipeline

This project implements a machine learning pipeline for analyzing risk data from Tata Motors, using both traditional ML methods and transformer-based models.

## Features

- **Data Processing**: Loads JSON dataset, removes duplicates, and preprocesses text data
- **Baseline Models**: TF-IDF + RandomForest for classification and regression
- **Transformer Fine-tuning**: DistilBERT model for advanced text classification
- **Multi-task Learning**: Predicts Risk_Type, Severity, and Risk_Score
- **Comprehensive Evaluation**: F1 scores, RMSE, and detailed classification reports

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the `tata_motors_risk_analysis.json` file in the same directory.

## Usage

Run the complete pipeline:
```bash
python risk_analysis_ml.py
```

## Pipeline Components

### 1. Data Loading & Preprocessing
- Loads JSON dataset with risk analysis data
- Removes duplicates based on Title and Explanation
- Creates combined text features from Title + Explanation
- Encodes categorical variables (Risk_Type, Severity)

### 2. Baseline ML Models
- **TF-IDF Vectorization**: Converts text to numerical features
- **RandomForest Classifier**: For Risk_Type and Severity prediction
- **RandomForest Regressor**: For Risk_Score prediction

### 3. Transformer Fine-tuning
- **DistilBERT**: Pre-trained transformer model
- **Fine-tuning**: Custom training on risk analysis data
- **Evaluation**: Comprehensive metrics and classification reports

### 4. Model Comparison
- Compares baseline vs transformer performance
- Detailed classification reports for each model
- Performance metrics: F1-score, RMSE

## Output

The script provides:
- Dataset statistics and preprocessing results
- Baseline model performance metrics
- Transformer training progress and evaluation
- Detailed classification reports
- Model comparison summary

## Requirements

- Python 3.8+
- PyTorch
- Transformers library
- Scikit-learn
- Pandas, NumPy
- JSON dataset file

## Notes

- The transformer training may take some time depending on your hardware
- GPU acceleration is recommended for transformer training
- Results are saved in `./results_risk/` directory
- Logs are saved in `./logs/` directory
