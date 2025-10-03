# Intelligent News Filter using NewsAPI + Gemini AI
# ================================================

import json
import requests
import google.generativeai as genai
from datetime import datetime, timedelta
import re

class IntelligentNewsFilter:
    """Fetch real news from NewsAPI and use Gemini AI for intelligent filtering"""
    
    def __init__(self, newsapi_key, gemini_key):
        """Initialize with both API keys"""
        self.newsapi_key = newsapi_key
        self.gemini_key = gemini_key
        
        # Configure Gemini for filtering
        genai.configure(api_key=gemini_key)
        self.gemini_model = genai.GenerativeModel('models/gemini-2.5-flash')
        
        print("Initialized with NewsAPI.ai and Gemini AI for intelligent filtering")
    
    def fetch_raw_news(self, query="Tata Motors", num_articles=50, hours_back=24):
        """Fetch raw news from NewsAPI.ai"""
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
                'apiKey': self.newsapi_key
            }
            
            print(f"Fetching {num_articles} raw news articles about: {query}")
            print(f"Time range: {from_date} to {to_date}")
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'articles' in data and data['articles']:
                articles = []
                for article in data['articles']['results']:
                    articles.append({
                        'Title': article.get('title', ''),
                        'Explanation': article.get('body', '') or article.get('summary', ''),
                        'publishedAt': article.get('date', ''),
                        'url': article.get('url', ''),
                        'source': article.get('source', {}).get('title', 'Unknown'),
                        'raw_content': article  # Keep original for reference
                    })
                
                print(f"Fetched {len(articles)} raw articles from NewsAPI")
                return articles
            else:
                print(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                return []
                
        except Exception as e:
            print(f"Error fetching raw news: {e}")
            return []
    
    def create_filtering_prompt(self, articles):
        """Create Gemini prompt for intelligent filtering"""
        # Prepare articles for filtering
        articles_text = ""
        for i, article in enumerate(articles, 1):
            articles_text += f"\nArticle {i}:\n"
            articles_text += f"Title: {article['Title']}\n"
            articles_text += f"Content: {article['Explanation'][:500]}...\n"
            articles_text += f"Source: {article['source']}\n"
            articles_text += "---\n"
        
        prompt = f"""
You are an AI expert in risk analysis for Tata Motors. Your task is to filter news articles and identify ONLY those that are relevant to Tata Motors risk analysis.

**Risk Categories to Consider:**
- Strategic risks (competitor moves, market expansions, partnerships)
- Supply chain disruptions (semiconductors, batteries, raw materials)
- Regulatory changes (EV policies, emission norms, government regulations)
- Financial risks (market conditions, economic factors, currency fluctuations)
- Operational risks (production issues, recalls, cybersecurity)
- Technology risks (EV technology, autonomous driving, charging infrastructure)

**Instructions:**
1. Analyze each article below
2. Identify articles that could impact Tata Motors' business risks
3. For relevant articles, extract key information
4. Return ONLY relevant articles in JSON format

**Output Format:**
Return a JSON array with only the relevant articles:
[
  {{
    "Title": "Original news title",
    "Explanation": "Detailed explanation of the news and its risk implications for Tata Motors",
    "Affected_Nodes": [],
    "Risk_Type": null,
    "Severity": null,
    "Risk_Score": null,
    "source": "Original source name",
    "publishedAt": "Original publish date",
    "url": "Original URL",
    "relevance_score": 0.85,
    "risk_implications": "Specific risk implications for Tata Motors"
  }}
]

**Articles to analyze:**
{articles_text}

Return only the JSON array, no other text.
"""
        return prompt
    
    def filter_news_with_gemini(self, articles):
        """Use Gemini AI to filter and enhance news articles"""
        if not articles:
            return []
        
        try:
            print(f"Using Gemini AI to filter {len(articles)} articles for Tata Motors relevance...")
            
            # Create filtering prompt
            prompt = self.create_filtering_prompt(articles)
            
            # Get Gemini response
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip()
            
            print("Gemini response received, parsing filtered articles...")
            
            # Clean response to extract JSON
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            # Find JSON array
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            
            if json_start == -1 or json_end == 0:
                print("No JSON array found in Gemini response")
                return []
            
            json_text = response_text[json_start:json_end]
            
            # Parse JSON
            filtered_articles = json.loads(json_text)
            
            print(f"Gemini AI filtered to {len(filtered_articles)} relevant articles")
            return filtered_articles
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print("Raw Gemini response:", response_text[:200])
            return []
        except Exception as e:
            print(f"Error in Gemini filtering: {e}")
            return []
    
    def process_news_pipeline(self, query="Tata Motors", num_articles=50, hours_back=24):
        """Complete pipeline: Fetch raw news -> Gemini filter -> JSON output"""
        print("Starting intelligent news processing pipeline...")
        print("="*60)
        
        # Step 1: Fetch raw news from NewsAPI
        print("Step 1: Fetching raw news from NewsAPI.ai...")
        raw_articles = self.fetch_raw_news(query, num_articles, hours_back)
        
        if not raw_articles:
            print("No raw articles fetched. Check your NewsAPI key and query.")
            return []
        
        # Step 2: Use Gemini AI for intelligent filtering
        print("\nStep 2: Using Gemini AI for intelligent filtering...")
        filtered_articles = self.filter_news_with_gemini(raw_articles)
        
        if not filtered_articles:
            print("No relevant articles found after Gemini filtering.")
            return []
        
        # Step 3: Add metadata
        print("\nStep 3: Adding metadata and finalizing...")
        for article in filtered_articles:
            article['processed_at'] = datetime.now().isoformat()
            article['filter_method'] = 'Gemini AI'
        
        print(f"\nPipeline completed successfully!")
        print(f"Raw articles: {len(raw_articles)}")
        print(f"Filtered articles: {len(filtered_articles)}")
        print(f"Relevance rate: {len(filtered_articles)/len(raw_articles)*100:.1f}%")
        
        return filtered_articles
    
    def save_filtered_news(self, articles, filename=None):
        """Save filtered news to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"filtered_news_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(articles, f, indent=2, ensure_ascii=False)
            print(f"Filtered news saved to {filename}")
            return filename
        except Exception as e:
            print(f"Error saving filtered news: {e}")
            return None

def main():
    """Test the intelligent news filter"""
    # API keys
    newsapi_key = "74830ae7-dea2-498e-b538-344e7a149eff"
    gemini_key = "AIzaSyAQDAZEPDDWwNx3loUxWLQfUjytNASG7ac"
    
    # Initialize filter
    filter_system = IntelligentNewsFilter(newsapi_key, gemini_key)
    
    # Process news pipeline
    filtered_articles = filter_system.process_news_pipeline(
        query="Tata Motors",
        num_articles=30,
        hours_back=24
    )
    
    if filtered_articles:
        print(f"\nFiltered {len(filtered_articles)} relevant articles:")
        for i, article in enumerate(filtered_articles, 1):
            print(f"\nArticle {i}:")
            print(f"Title: {article['Title']}")
            print(f"Source: {article['source']}")
            print(f"Relevance Score: {article.get('relevance_score', 'N/A')}")
            print(f"Risk Implications: {article.get('risk_implications', 'N/A')[:100]}...")
        
        # Save results
        filename = filter_system.save_filtered_news(filtered_articles)
        print(f"\nResults saved to: {filename}")
    else:
        print("No relevant articles found")

if __name__ == "__main__":
    main()
