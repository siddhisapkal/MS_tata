# Working Gemini News Fetcher for Tata Motors Risk Analysis
# =========================================================

import json
import google.generativeai as genai
from datetime import datetime
import re

class WorkingGeminiNewsFetcher:
    """Working Gemini API news fetcher for Tata Motors risk analysis"""
    
    def __init__(self, api_key):
        """Initialize with Gemini API key"""
        self.api_key = api_key
        genai.configure(api_key=api_key)
        
        # Use the latest working model
        self.model = genai.GenerativeModel('models/gemini-2.5-flash')
        print("Initialized with Gemini 2.5 Flash model")
    
    def create_news_prompt(self, query="Tata Motors", num_articles=10):
        """Create prompt for generating relevant news articles"""
        prompt = f"""
You are an AI tasked with generating {num_articles} realistic news articles specifically relevant to Tata Motors risk analysis.

Requirements:
1. **Relevance**: Only articles that could indicate risks for Tata Motors
2. **Risk Types**: Strategic, Supply Chain, Regulatory, Financial, Operational
3. **Topics**: EV market, automotive industry, competitor moves, policy changes, supply chain disruptions
4. **Format**: JSON array with exact structure below

Generate {num_articles} articles in this JSON format:
[
  {{
    "Title": "Realistic news headline about {query}",
    "Explanation": "Detailed explanation of the news and potential risks for Tata Motors",
    "Affected_Nodes": [],
    "Risk_Type": null,
    "Severity": null,
    "Risk_Score": null
  }}
]

Focus on realistic scenarios like:
- Competitor expansions in EV market
- Government policy changes affecting automotive sector
- Supply chain disruptions (semiconductors, batteries)
- Regulatory changes in EV adoption
- Financial challenges or opportunities
- Technology disruptions
- Market competition

Make each article realistic and relevant to Tata Motors' business risks.
Return only valid JSON, no other text.
"""
        return prompt
    
    def fetch_news(self, query="Tata Motors", num_articles=10):
        """Fetch news using Gemini API"""
        try:
            print(f"Generating {num_articles} relevant news articles about {query}...")
            
            # Create prompt
            prompt = self.create_news_prompt(query, num_articles)
            
            # Generate response
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            print("Response received, parsing JSON...")
            
            # Clean response to extract JSON
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            # Find JSON array
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            
            if json_start == -1 or json_end == 0:
                print("No JSON array found in response")
                return []
            
            json_text = response_text[json_start:json_end]
            
            # Parse JSON
            articles = json.loads(json_text)
            
            # Add metadata
            for article in articles:
                article['source'] = 'Gemini AI'
                article['publishedAt'] = datetime.now().isoformat()
                article['url'] = 'https://gemini-ai-generated.com'
            
            print(f"Successfully generated {len(articles)} articles")
            return articles
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print("Raw response:", response_text[:200])
            return []
        except Exception as e:
            print(f"Error: {e}")
            return []
    
    def save_articles(self, articles, filename=None):
        """Save articles to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gemini_news_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(articles, f, indent=2, ensure_ascii=False)
            print(f"Articles saved to {filename}")
            return filename
        except Exception as e:
            print(f"Error saving: {e}")
            return None

def main():
    """Test the working Gemini news fetcher"""
    # Your Gemini API key
    api_key = "AIzaSyAQDAZEPDDWwNx3loUxWLQfUjytNASG7ac"
    
    # Initialize fetcher
    fetcher = WorkingGeminiNewsFetcher(api_key)
    
    # Fetch news
    articles = fetcher.fetch_news(
        query="Tata Motors risk analysis",
        num_articles=10
    )
    
    if articles:
        print(f"\nGenerated {len(articles)} articles:")
        for i, article in enumerate(articles, 1):
            print(f"\nArticle {i}:")
            print(f"Title: {article['Title']}")
            print(f"Explanation: {article['Explanation'][:100]}...")
        
        # Save results
        filename = fetcher.save_articles(articles)
        print(f"\nResults saved to: {filename}")
    else:
        print("No articles generated")

if __name__ == "__main__":
    main()
