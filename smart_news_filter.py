# Smart News Filter using NewsAPI + Keyword-based filtering
# ========================================================

import json
import requests
from datetime import datetime, timedelta
import re

class SmartNewsFilter:
    """Fetch real news from NewsAPI and use smart keyword filtering for Tata Motors relevance"""
    
    def __init__(self, newsapi_key):
        """Initialize with NewsAPI key"""
        self.newsapi_key = newsapi_key
        
        # Tata Motors specific risk-relevant keywords - MUCH MORE PRECISE
        self.risk_keywords = {
            'tata_motors_direct': ['tata motors', 'tata group automotive', 'jaguar land rover', 'jlr', 'tata nexon', 'tata harrier', 'tata safari', 'tata punch', 'tata altroz', 'tata tiago', 'tata tigor'],
            'ev_specific': ['electric vehicle', 'ev', 'battery', 'charging', 'electric car', 'lithium', 'lithium-ion', 'ev battery', 'charging station', 'ev charging', 'electric mobility'],
            'automotive_specific': ['automotive', 'automobile', 'car', 'vehicle', 'auto industry', 'passenger vehicle', 'commercial vehicle', 'suv', 'sedan', 'hatchback'],
            'supply_chain_risks': ['supply chain', 'semiconductor', 'chip shortage', 'battery shortage', 'raw materials', 'steel price', 'aluminum price', 'rubber price', 'lithium price', 'cobalt price', 'nickel price', 'supply disruption', 'component shortage'],
            'ev_policy_regulatory': ['fame', 'fame scheme', 'ev subsidy', 'government subsidy', 'ev policy', 'emission norms', 'bs6', 'bs7', 'carbon emission', 'ev incentive', 'electric vehicle policy', 'ev charging policy'],
            'competitors_automotive': ['mahindra', 'maruti suzuki', 'hyundai', 'toyota', 'honda', 'volkswagen', 'bmw', 'mercedes', 'kia', 'mg motor', 'skoda', 'renault', 'nissan'],
            'automotive_risks': ['recall', 'safety', 'crash', 'defect', 'quality issue', 'production halt', 'manufacturing', 'assembly line', 'factory', 'plant'],
            'ev_technology': ['autonomous', 'connected car', 'iot', 'ev technology', 'battery technology', 'charging technology', 'ev infrastructure']
        }
        
        print("Initialized with NewsAPI.ai and smart keyword filtering")
    
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
                'apiKey': self.newsapi_key,
                # Add timestamp to prevent caching
                'timestamp': int(datetime.now().timestamp())
            }
            
            print(f"Fetching {num_articles} raw news articles about: {query}")
            print(f"Time range: {from_date} to {to_date}")
            print(f"Request timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
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
                        'raw_content': article
                    })
                
                print(f"Fetched {len(articles)} raw articles from NewsAPI")
                return articles
            else:
                print(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                return []
                
        except Exception as e:
            print(f"Error fetching raw news: {e}")
            return []
    
    def calculate_relevance_score(self, article):
        """Calculate relevance score for Tata Motors risk analysis - STRICT FILTERING"""
        title = article['Title'].lower()
        explanation = article['Explanation'].lower()
        text = f"{title} {explanation}"
        
        score = 0
        matched_categories = []
        
        # MUST have automotive/EV context to be relevant - STRICT CHECK
        automotive_context = False
        ev_context = False
        
        # Check for automotive context - MUST be in title for strict filtering
        for keyword in self.risk_keywords['automotive_specific']:
            if keyword.lower() in title:
                automotive_context = True
                break
        
        # Check for EV context - MUST be in title for strict filtering
        for keyword in self.risk_keywords['ev_specific']:
            if keyword.lower() in title:
                ev_context = True
                break
        
        # If no automotive or EV context, reject immediately
        if not automotive_context and not ev_context:
            return {
                'relevance_score': 0.0,
                'matched_categories': [],
                'total_matches': 0
            }
        
        # ADDITIONAL CHECK: Must have Tata Motors OR automotive industry context
        tata_context = any(keyword in text for keyword in self.risk_keywords['tata_motors_direct'])
        automotive_industry_context = any(keyword in text for keyword in ['automotive industry', 'auto industry', 'car industry', 'vehicle industry'])
        
        if not tata_context and not automotive_industry_context:
            return {
                'relevance_score': 0.0,
                'matched_categories': [],
                'total_matches': 0
            }
        
        # Check each keyword category
        for category, keywords in self.risk_keywords.items():
            category_score = 0
            for keyword in keywords:
                if keyword.lower() in text:
                    category_score += 1
                    score += 1
            
            if category_score > 0:
                matched_categories.append(category)
        
        # MASSIVE bonus for direct Tata Motors mentions
        if any(keyword in text for keyword in self.risk_keywords['tata_motors_direct']):
            score += 5
        
        # Bonus for supply chain risks (very relevant)
        if 'supply_chain_risks' in matched_categories:
            score += 3
        
        # Bonus for EV policy/regulatory changes
        if 'ev_policy_regulatory' in matched_categories:
            score += 2
        
        # Bonus for automotive risks
        if 'automotive_risks' in matched_categories:
            score += 2
        
        # Normalize score (0-1) - be more strict
        normalized_score = min(score / 15, 1.0)
        
        # Additional filtering: require minimum automotive/EV relevance
        if not automotive_context and not ev_context:
            normalized_score = 0.0
        
        return {
            'relevance_score': normalized_score,
            'matched_categories': matched_categories,
            'total_matches': score
        }
    
    def filter_news_smart(self, articles, min_relevance=0.3):
        """Filter news using smart keyword matching"""
        if not articles:
            return []
        
        print(f"Filtering {len(articles)} articles for Tata Motors relevance...")
        
        filtered_articles = []
        
        for article in articles:
            # Calculate relevance
            relevance_data = self.calculate_relevance_score(article)
            
            # Filter by minimum relevance
            if relevance_data['relevance_score'] >= min_relevance:
                # Add relevance data to article
                article['relevance_score'] = relevance_data['relevance_score']
                article['matched_categories'] = relevance_data['matched_categories']
                article['total_matches'] = relevance_data['total_matches']
                
                # Add risk analysis fields
                article['Affected_Nodes'] = []
                article['Risk_Type'] = None
                article['Severity'] = None
                article['Risk_Score'] = None
                
                # Determine risk type based on categories - AUTOMOTIVE/EV SPECIFIC
                if 'supply_chain_risks' in relevance_data['matched_categories']:
                    article['Risk_Type'] = 'Supply Chain'
                elif 'ev_policy_regulatory' in relevance_data['matched_categories']:
                    article['Risk_Type'] = 'Regulatory'
                elif 'automotive_risks' in relevance_data['matched_categories']:
                    article['Risk_Type'] = 'Operational'
                elif 'ev_technology' in relevance_data['matched_categories']:
                    article['Risk_Type'] = 'Technology'
                elif 'competitors_automotive' in relevance_data['matched_categories']:
                    article['Risk_Type'] = 'Competitive'
                else:
                    article['Risk_Type'] = 'Strategic'
                
                # Determine severity based on score
                if relevance_data['relevance_score'] >= 0.8:
                    article['Severity'] = 'High'
                elif relevance_data['relevance_score'] >= 0.5:
                    article['Severity'] = 'Medium'
                else:
                    article['Severity'] = 'Low'
                
                # Calculate risk score (0-10)
                article['Risk_Score'] = round(relevance_data['relevance_score'] * 10, 2)
                
                filtered_articles.append(article)
        
        print(f"Filtered to {len(filtered_articles)} relevant articles")
        return filtered_articles
    
    def process_news_pipeline(self, query="Tata Motors automotive electric vehicle", num_articles=50, hours_back=24, min_relevance=0.6):
        """Complete pipeline: Fetch raw news -> Smart filter -> JSON output"""
        print("Starting smart news processing pipeline...")
        print("="*60)
        
        # Step 1: Fetch raw news from NewsAPI
        print("Step 1: Fetching raw news from NewsAPI.ai...")
        raw_articles = self.fetch_raw_news(query, num_articles, hours_back)
        
        if not raw_articles:
            print("No raw articles fetched. Check your NewsAPI key and query.")
            return []
        
        # Step 2: Smart filtering
        print("\nStep 2: Using smart keyword filtering...")
        filtered_articles = self.filter_news_smart(raw_articles, min_relevance)
        
        if not filtered_articles:
            print("No relevant articles found after filtering.")
            return []
        
        # Step 3: Add metadata
        print("\nStep 3: Adding metadata and finalizing...")
        for article in filtered_articles:
            article['processed_at'] = datetime.now().isoformat()
            article['filter_method'] = 'Smart Keyword Filtering'
        
        print(f"\nPipeline completed successfully!")
        print(f"Raw articles: {len(raw_articles)}")
        print(f"Filtered articles: {len(filtered_articles)}")
        print(f"Relevance rate: {len(filtered_articles)/len(raw_articles)*100:.1f}%")
        
        return filtered_articles
    
    def save_filtered_news(self, articles, filename=None):
        """Save filtered news to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"smart_filtered_news_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(articles, f, indent=2, ensure_ascii=False)
            print(f"Filtered news saved to {filename}")
            return filename
        except Exception as e:
            print(f"Error saving filtered news: {e}")
            return None

def main():
    """Test the smart news filter"""
    # API key
    newsapi_key = "74830ae7-dea2-498e-b538-344e7a149eff"
    
    # Initialize filter
    filter_system = SmartNewsFilter(newsapi_key)
    
    # Process news pipeline
    filtered_articles = filter_system.process_news_pipeline(
        query="Tata Motors",
        num_articles=30,
        hours_back=24,
        min_relevance=0.3
    )
    
    if filtered_articles:
        print(f"\nFiltered {len(filtered_articles)} relevant articles:")
        for i, article in enumerate(filtered_articles, 1):
            print(f"\nArticle {i}:")
            print(f"Title: {article['Title'].encode('ascii', 'ignore').decode('ascii')}")
            print(f"Source: {article['source']}")
            print(f"Relevance Score: {article['relevance_score']:.2f}")
            print(f"Risk Type: {article['Risk_Type']}")
            print(f"Severity: {article['Severity']}")
            print(f"Risk Score: {article['Risk_Score']}")
            print(f"Matched Categories: {', '.join(article['matched_categories'])}")
        
        # Save results
        filename = filter_system.save_filtered_news(filtered_articles)
        print(f"\nResults saved to: {filename}")
    else:
        print("No relevant articles found")

if __name__ == "__main__":
    main()
