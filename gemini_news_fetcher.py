# Gemini API News Fetcher for Tata Motors Risk Analysis
# =====================================================

import json
import requests
import google.generativeai as genai
from datetime import datetime
import re
import time

class GeminiNewsFetcher:
    """Gemini API-based news fetcher for Tata Motors risk analysis"""
    
    def __init__(self, api_key):
        """Initialize with Gemini API key"""
        self.api_key = api_key
        genai.configure(api_key=api_key)
        # Try different model names
        try:
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        except:
            try:
                self.model = genai.GenerativeModel('gemini-pro')
            except:
                # Use the default model
                self.model = genai.GenerativeModel('models/gemini-1.5-flash')
        
        # Risk-relevant keywords for filtering
        self.risk_keywords = [
            'Tata Motors', 'EV', 'electric vehicles', 'automotive', 
            'competitor', 'merger', 'acquisition', 'policy', 
            'government regulation', 'supply chain', 'semiconductor shortage',
            'product recall', 'cybersecurity', 'automotive sector',
            'Jaguar Land Rover', 'Tata Group', 'automotive industry',
            'EV policy', 'automotive policy', 'vehicle manufacturing',
            'automotive supply chain', 'electric vehicle market'
        ]
        
        # Credible news sources for automotive/business
        self.credible_sources = [
            'Reuters', 'Bloomberg', 'Financial Times', 'Wall Street Journal',
            'Economic Times', 'Business Standard', 'Mint', 'Hindu Business Line',
            'Autocar', 'Autocar India', 'Car and Driver', 'Motor Trend',
            'Automotive News', 'Just Auto', 'Auto Express', 'Top Gear',
            'CNBC', 'BBC Business', 'Forbes', 'Fortune'
        ]
    
    def create_news_prompt(self, query="Tata Motors", num_articles=20):
        """Create the Gemini prompt for fetching relevant news"""
        prompt = f"""
You are an AI coder tasked with fetching news articles **specifically relevant to Tata Motors risk analysis**.

Requirements:

1. **Source Selection**
   - Only consider credible news sources related to business, automotive industry, financial markets, and technology.
   - Ignore unrelated sources (sports, entertainment, lifestyle).

2. **Relevance Filtering**
   - Include only news that may indicate **risks for Tata Motors**, such as:
     - Strategic risks (competitor moves, market expansions)
     - Supply chain disruptions
     - Regulatory or policy changes affecting automotive sector
     - Cybersecurity incidents or product recalls
     - Financial/operational risks (loans, mergers, acquisitions)
   - **Keywords to consider for relevance:**  
     `Tata Motors, EV, electric vehicles, automotive, competitor, merger, acquisition, policy, government regulation, supply chain, semiconductor shortage, product recall, cybersecurity`  
   - Exclude general news not impacting Tata Motors risks.

3. **Output Format**
   - Return exactly {num_articles} relevant articles
   - JSON structured like the training dataset:
     ```json
     [
       {{
         "Title": "string",
         "Explanation": "string",
         "Affected_Nodes": [],
         "Risk_Type": null,
         "Severity": null,
         "Risk_Score": null
       }}
     ]
     ```

4. **Text Extraction**
   - Extract concise summaries or explanations for each article.
   - Clean HTML, advertisements, and unrelated content.

5. **Quantity**
   - Return only news articles that are relevant and actionable (no bulk irrelevant news).

Goal: Generate a **live JSON feed of Tata Motors risk-relevant news** ready to be processed by the ML pipeline for Risk_Type, Severity, and Risk_Score prediction.

Search for news related to: {query}
Return exactly {num_articles} relevant articles in the specified JSON format.
"""
        return prompt
    
    def fetch_news(self, query="Tata Motors", num_articles=20):
        """Fetch relevant news using Gemini API"""
        try:
            print(f"Fetching {num_articles} relevant news articles for: {query}")
            print("Using Gemini API for intelligent news filtering...")
            
            # Create the prompt
            prompt = self.create_news_prompt(query, num_articles)
            
            # Generate response from Gemini
            response = self.model.generate_content(prompt)
            
            # Extract JSON from response
            response_text = response.text
            
            # Clean the response to extract JSON
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            
            if json_start == -1 or json_end == 0:
                print("No JSON array found in response")
                return []
            
            json_text = response_text[json_start:json_end]
            
            # Parse JSON
            try:
                articles = json.loads(json_text)
                print(f"Successfully fetched {len(articles)} relevant articles")
                return articles
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print("Raw response:", response_text[:500])
                return []
                
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []
    
    def validate_article_relevance(self, article):
        """Validate if article is relevant to Tata Motors risk analysis"""
        title = article.get('Title', '').lower()
        explanation = article.get('Explanation', '').lower()
        
        # Check for risk-relevant keywords
        text = f"{title} {explanation}"
        relevance_score = 0
        
        for keyword in self.risk_keywords:
            if keyword.lower() in text:
                relevance_score += 1
        
        # Must have at least 2 relevant keywords
        return relevance_score >= 2
    
    def clean_article(self, article):
        """Clean and format article data"""
        # Clean title
        title = article.get('Title', '').strip()
        title = re.sub(r'[^\w\s\-.,!?]', '', title)
        
        # Clean explanation
        explanation = article.get('Explanation', '').strip()
        explanation = re.sub(r'<[^>]+>', '', explanation)  # Remove HTML tags
        explanation = re.sub(r'[^\w\s\-.,!?]', '', explanation)
        
        # Ensure minimum length
        if len(explanation) < 50:
            explanation = f"{title}. {explanation}"
        
        return {
            'Title': title,
            'Explanation': explanation,
            'Affected_Nodes': [],
            'Risk_Type': None,
            'Severity': None,
            'Risk_Score': None,
            'source': 'Gemini AI',
            'publishedAt': datetime.now().isoformat(),
            'url': 'https://gemini-ai-generated.com'
        }
    
    def fetch_and_validate_news(self, query="Tata Motors", num_articles=20):
        """Fetch news and validate relevance"""
        print(f"Fetching {num_articles} Tata Motors risk-relevant news articles...")
        
        # Fetch news from Gemini
        raw_articles = self.fetch_news(query, num_articles)
        
        if not raw_articles:
            print("No articles fetched from Gemini")
            return []
        
        # Validate and clean articles
        validated_articles = []
        for article in raw_articles:
            if self.validate_article_relevance(article):
                cleaned_article = self.clean_article(article)
                validated_articles.append(cleaned_article)
            else:
                print(f"Skipping irrelevant article: {article.get('Title', '')[:50]}...")
        
        print(f"Validated {len(validated_articles)} relevant articles")
        return validated_articles
    
    def save_results(self, articles, filename=None):
        """Save articles to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gemini_news_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(articles, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {filename}")
            return filename
        except Exception as e:
            print(f"Error saving results: {e}")
            return None

def main():
    """Test the Gemini news fetcher"""
    # Your Gemini API key
    api_key = "AIzaSyAQDAZEPDDWwNx3loUxWLQfUjytNASG7ac"
    
    # Initialize fetcher
    fetcher = GeminiNewsFetcher(api_key)
    
    # Fetch relevant news
    articles = fetcher.fetch_and_validate_news(
        query="Tata Motors risk analysis",
        num_articles=15
    )
    
    if articles:
        print(f"\nSuccessfully fetched {len(articles)} relevant articles:")
        for i, article in enumerate(articles, 1):
            print(f"\nArticle {i}:")
            print(f"Title: {article['Title']}")
            print(f"Explanation: {article['Explanation'][:100]}...")
        
        # Save results
        filename = fetcher.save_results(articles)
        print(f"\nResults saved to: {filename}")
    else:
        print("No relevant articles found")

if __name__ == "__main__":
    main()
