# Simple Gemini News Fetcher for Tata Motors Risk Analysis
# =======================================================

import json
import google.generativeai as genai
from datetime import datetime
import re

class SimpleGeminiNewsFetcher:
    """Simple Gemini API news fetcher for Tata Motors risk analysis"""
    
    def __init__(self, api_key):
        """Initialize with Gemini API key"""
        self.api_key = api_key
        genai.configure(api_key=api_key)
        
        # Use the most basic model configuration
        try:
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        except Exception as e:
            print(f"Model initialization error: {e}")
            # Fallback to any available model
            self.model = None
    
    def create_simple_prompt(self, query="Tata Motors", num_articles=10):
        """Create a simple prompt for news generation"""
        prompt = f"""
Generate {num_articles} realistic news articles about {query} that could be relevant for risk analysis.

Format as JSON array with this structure:
[
  {{
    "Title": "News headline about {query}",
    "Explanation": "Brief explanation of the news and potential risks",
    "Affected_Nodes": [],
    "Risk_Type": null,
    "Severity": null,
    "Risk_Score": null
  }}
]

Focus on:
- Strategic risks (competitor moves, market changes)
- Supply chain issues
- Regulatory changes
- Financial risks
- Technology disruptions

Make the news realistic and relevant to automotive/EV industry.
Return only valid JSON, no other text.
"""
        return prompt
    
    def fetch_news(self, query="Tata Motors", num_articles=10):
        """Fetch news using Gemini API"""
        if not self.model:
            print("Model not available")
            return []
        
        try:
            print(f"Fetching {num_articles} news articles about {query}...")
            
            # Create prompt
            prompt = self.create_simple_prompt(query, num_articles)
            
            # Generate response
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            print("Raw response received, parsing JSON...")
            
            # Clean response to extract JSON
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            # Find JSON array
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            
            if json_start == -1 or json_end == 0:
                print("No JSON array found")
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
    """Test the simple Gemini news fetcher"""
    # Your Gemini API key
    api_key = "AIzaSyAQDAZEPDDWwNx3loUxWLQfUjytNASG7ac"
    
    # Initialize fetcher
    fetcher = SimpleGeminiNewsFetcher(api_key)
    
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
