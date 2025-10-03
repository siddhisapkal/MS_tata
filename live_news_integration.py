# Live News Integration for Risk Analysis
# ======================================

import json
import requests
import time
from datetime import datetime, timedelta
import pandas as pd
from enhanced_risk_analysis_v2 import EnhancedRiskAnalyzer
import joblib
import torch

class LiveNewsRiskAnalyzer:
    """Live news integration with risk analysis"""
    
    def __init__(self, model_path_prefix="enhanced_risk_models"):
        """Initialize with trained models"""
        self.model_path_prefix = model_path_prefix
        self.load_models()
    
    def load_models(self):
        """Load trained models"""
        print("Loading trained models...")
        
        try:
            # Load baseline models
            self.baseline_models = joblib.load(f"{self.model_path_prefix}_baseline.joblib")
            
            # Load transformer models if available
            self.severity_transformer = None
            self.risk_score_transformer = None
            
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                
                # Load severity transformer
                severity_model = AutoModelForSequenceClassification.from_pretrained(
                    f"{self.model_path_prefix}_severity_tokenizer"
                )
                severity_tokenizer = AutoTokenizer.from_pretrained(f"{self.model_path_prefix}_severity_tokenizer")
                le_sev = joblib.load(f"{self.model_path_prefix}_le_sev.joblib")
                
                self.severity_transformer = {
                    'model': severity_model,
                    'tokenizer': severity_tokenizer,
                    'le_sev': le_sev
                }
                print("Severity transformer loaded successfully")
                
            except Exception as e:
                print(f"Severity transformer not available: {e}")
            
            try:
                # Load risk score transformer
                risk_score_model = AutoModelForSequenceClassification.from_pretrained(
                    f"{self.model_path_prefix}_risk_score_tokenizer"
                )
                risk_score_tokenizer = AutoTokenizer.from_pretrained(f"{self.model_path_prefix}_risk_score_tokenizer")
                
                self.risk_score_transformer = {
                    'model': risk_score_model,
                    'tokenizer': risk_score_tokenizer
                }
                print("Risk score transformer loaded successfully")
                
            except Exception as e:
                print(f"Risk score transformer not available: {e}")
            
            # Create analyzer
            self.analyzer = EnhancedRiskAnalyzer(
                self.baseline_models, 
                self.severity_transformer, 
                self.risk_score_transformer
            )
            
            print("All models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def fetch_news(self, api_key, query="Tata Motors", num_articles=10, hours_back=24):
        """Fetch live news from NewsAPI.ai"""
        try:
            # Calculate time range
            to_date = datetime.now()
            from_date = to_date - timedelta(hours=hours_back)
            
            url = "https://eventregistry.org/api/v1/article/getArticles"
            params = {
                'resultType': 'articles',
                'keyword': query,
                'keywordOper': 'or',
                'lang': 'eng',
                'articlesSortBy': 'date',
                'maxItems': num_articles,
                'dateStart': from_date.strftime('%Y-%m-%d'),
                'dateEnd': to_date.strftime('%Y-%m-%d'),
                'includeArticleConcepts': 'true',
                'includeArticleCategories': 'true',
                'apiKey': api_key
            }
            
            print(f"Fetching news for query: '{query}'")
            print(f"Time range: {from_date} to {to_date}")
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'articles' in data and data['articles']:
                articles = []
                for article in data['articles']['results']:
                    articles.append({
                        'Title': article.get('title', ''),
                        'Explanation': article.get('body', '') or article.get('summary', ''),
                        'Affected_Nodes': [],
                        'Risk_Type': None,
                        'Severity': None,
                        'Risk_Score': None,
                        'publishedAt': article.get('date', ''),
                        'url': article.get('url', ''),
                        'source': article.get('source', {}).get('title', 'Unknown')
                    })
                
                print(f"Fetched {len(articles)} articles")
                return articles
            else:
                print(f"News API error: {data.get('message', 'Unknown error')}")
                return []
                
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []
    
    def analyze_articles(self, articles):
        """Analyze articles for risk"""
        print(f"\nAnalyzing {len(articles)} articles...")
        
        results = []
        for i, article in enumerate(articles):
            text = article['Title'] + ' ' + article['Explanation']
            
            # Predict risk using ensemble method
            prediction = self.analyzer.predict_risk(text, model_type="ensemble")
            
            # Update article with predictions
            article['Risk_Type'] = prediction['risk_type']
            article['Severity'] = prediction['severity']
            article['Risk_Score'] = prediction['risk_score']
            article['Model_Type'] = prediction['model_type']
            
            results.append(article)
            
            print(f"Article {i+1}: {article['Title'][:60].encode('ascii', 'ignore').decode('ascii')}...")
            print(f"  Risk: {prediction['risk_type']}, Severity: {prediction['severity']}, Score: {prediction['risk_score']:.2f}")
        
        return results
    
    def save_results(self, results, filename="live_risk_analysis.json"):
        """Save analysis results to JSON"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {filename}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def create_summary_report(self, results):
        """Create summary report of risk analysis"""
        if not results:
            return "No articles analyzed"
        
        df = pd.DataFrame(results)
        
        # Risk type distribution
        risk_dist = df['Risk_Type'].value_counts()
        
        # Severity distribution
        severity_dist = df['Severity'].value_counts()
        
        # Average risk score
        avg_risk_score = df['Risk_Score'].mean()
        
        # High risk articles
        high_risk = df[df['Risk_Score'] > 7.0]
        
        report = f"""
RISK ANALYSIS SUMMARY REPORT
============================
Total Articles Analyzed: {len(results)}
Average Risk Score: {avg_risk_score:.2f}

Risk Type Distribution:
{risk_dist.to_string()}

Severity Distribution:
{severity_dist.to_string()}

High Risk Articles (Score > 7.0): {len(high_risk)}
"""
        
        if len(high_risk) > 0:
            report += "\nHigh Risk Articles:\n"
            for _, article in high_risk.iterrows():
                report += f"- {article['Title'][:50]}... (Score: {article['Risk_Score']:.2f})\n"
        
        return report
    
    def run_live_analysis(self, api_key, query="Tata Motors", num_articles=20, hours_back=24):
        """Run complete live analysis pipeline"""
        print("LIVE NEWS RISK ANALYSIS PIPELINE")
        print("="*50)
        
        # Fetch news
        articles = self.fetch_news(api_key, query, num_articles, hours_back)
        
        if not articles:
            print("No articles fetched. Check your API key and query.")
            return
        
        # Analyze articles
        results = self.analyze_articles(articles)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"live_risk_analysis_{timestamp}.json"
        self.save_results(results, filename)
        
        # Create summary report
        summary = self.create_summary_report(results)
        print(summary)
        
        # Save summary report
        summary_filename = f"risk_summary_{timestamp}.txt"
        with open(summary_filename, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"Summary report saved to {summary_filename}")
        
        return results, summary

def main():
    """Main function for live news analysis"""
    print("LIVE NEWS RISK ANALYSIS")
    print("="*30)
    
    # Initialize analyzer
    analyzer = LiveNewsRiskAnalyzer()
    
    # Get API key from user
    api_key = input("Enter your News API key (or press Enter to skip): ").strip()
    
    if not api_key:
        print("No API key provided. Skipping live news analysis.")
        print("To use live news analysis, provide a valid News API key.")
        return
    
    # Run live analysis
    try:
        results, summary = analyzer.run_live_analysis(
            api_key=api_key,
            query="Tata Motors",
            num_articles=20,
            hours_back=24
        )
        
        print("\nLive analysis completed successfully!")
        
    except Exception as e:
        print(f"Error in live analysis: {e}")

if __name__ == "__main__":
    main()
