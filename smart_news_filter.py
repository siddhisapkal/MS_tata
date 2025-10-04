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
                    # Parse and format the date properly
                    pub_date = article.get('date', '')
                    if pub_date:
                        try:
                            # Convert to readable format
                            if isinstance(pub_date, str):
                                # Try to parse different date formats
                                for fmt in ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d', '%Y-%m-%dT%H:%M:%SZ']:
                                    try:
                                        parsed_date = datetime.strptime(pub_date.split('T')[0], '%Y-%m-%d')
                                        pub_date = parsed_date.strftime('%Y-%m-%d %H:%M')
                                        break
                                    except:
                                        continue
                        except:
                            pub_date = str(pub_date)
                    
                    articles.append({
                        'Title': article.get('title', ''),
                        'Explanation': article.get('body', '') or article.get('summary', ''),
                        'publishedAt': pub_date,
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
                
                # Determine risk type based on content analysis - AUTOMOTIVE/EV SPECIFIC
                title_lower = article['Title'].lower()
                content_lower = article.get('Explanation', '').lower()
                combined_text = f"{title_lower} {content_lower}"
                
                # More intelligent risk type classification
                if any(keyword in combined_text for keyword in ['recall', 'defect', 'safety', 'quality', 'manufacturing', 'production', 'facility', 'plant', 'assembly']):
                    article['Risk_Type'] = 'Operational'
                elif any(keyword in combined_text for keyword in ['supply chain', 'shortage', 'semiconductor', 'chip', 'lithium', 'battery', 'raw material', 'component', 'disruption']):
                    article['Risk_Type'] = 'Supply Chain'
                elif any(keyword in combined_text for keyword in ['policy', 'regulation', 'subsidy', 'fame', 'government', 'ministry', 'compliance', 'standard', 'emission', 'safety']):
                    article['Risk_Type'] = 'Regulatory'
                elif any(keyword in combined_text for keyword in ['technology', 'innovation', 'battery', 'charging', 'autonomous', 'ai', 'software', 'digital', 'connectivity']):
                    article['Risk_Type'] = 'Technology'
                elif any(keyword in combined_text for keyword in ['competitor', 'mahindra', 'maruti', 'hyundai', 'toyota', 'honda', 'market share', 'competition', 'rival']):
                    article['Risk_Type'] = 'Competitive'
                elif any(keyword in combined_text for keyword in ['financial', 'revenue', 'profit', 'loss', 'investment', 'funding', 'merger', 'acquisition', 'partnership', 'deal']):
                    article['Risk_Type'] = 'Financial'
                elif any(keyword in combined_text for keyword in ['cyber', 'security', 'hack', 'breach', 'data', 'privacy', 'digital security']):
                    article['Risk_Type'] = 'Cybersecurity'
                elif any(keyword in combined_text for keyword in ['environmental', 'sustainability', 'carbon', 'emission', 'green', 'climate', 'esg']):
                    article['Risk_Type'] = 'Environmental'
                elif any(keyword in combined_text for keyword in ['demerger', 'restructuring', 'reorganization', 'spin-off', 'divestment', 'strategic']):
                    article['Risk_Type'] = 'Strategic'
                else:
                    # Fallback based on categories
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
                
                # Determine severity based on content analysis and score
                severity_keywords_high = ['crisis', 'shortage', 'recall', 'defect', 'safety', 'emergency', 'urgent', 'critical', 'severe', 'major', 'significant', 'disruption', 'halt', 'stop', 'shutdown']
                severity_keywords_medium = ['concern', 'issue', 'challenge', 'problem', 'delay', 'slow', 'decline', 'drop', 'fall', 'reduction', 'moderate', 'some', 'partial']
                severity_keywords_low = ['growth', 'increase', 'rise', 'improvement', 'positive', 'good', 'strong', 'record', 'success', 'achievement', 'milestone']
                
                # Check for severity indicators in content
                has_high_severity = any(keyword in combined_text for keyword in severity_keywords_high)
                has_medium_severity = any(keyword in combined_text for keyword in severity_keywords_medium)
                has_low_severity = any(keyword in combined_text for keyword in severity_keywords_low)
                
                # Determine severity based on content and score
                if has_high_severity or relevance_data['relevance_score'] >= 0.8:
                    article['Severity'] = 'High'
                elif has_medium_severity or relevance_data['relevance_score'] >= 0.5:
                    article['Severity'] = 'Medium'
                elif has_low_severity or relevance_data['relevance_score'] >= 0.3:
                    article['Severity'] = 'Low'
                else:
                    article['Severity'] = 'Low'  # Default to low if no clear indicators
                
                # Calculate risk score (0-10) based on content analysis
                base_score = relevance_data['relevance_score'] * 10
                
                # Adjust score based on risk type and severity indicators
                if article['Risk_Type'] == 'Supply Chain':
                    base_score += 1.0  # Supply chain risks are critical for automotive
                elif article['Risk_Type'] == 'Regulatory':
                    base_score += 0.8  # Regulatory changes can significantly impact business
                elif article['Risk_Type'] == 'Operational':
                    base_score += 1.2  # Operational issues like recalls are very serious
                elif article['Risk_Type'] == 'Technology':
                    base_score += 0.5  # Technology risks are important but manageable
                elif article['Risk_Type'] == 'Competitive':
                    base_score += 0.3  # Competitive risks are moderate
                elif article['Risk_Type'] == 'Financial':
                    base_score += 0.7  # Financial risks are significant
                elif article['Risk_Type'] == 'Cybersecurity':
                    base_score += 1.5  # Cybersecurity risks are very serious
                elif article['Risk_Type'] == 'Environmental':
                    base_score += 0.6  # Environmental risks are important
                
                # Adjust based on severity keywords
                if has_high_severity:
                    base_score += 1.0
                elif has_medium_severity:
                    base_score += 0.5
                elif has_low_severity:
                    base_score -= 0.5
                
                # Ensure score is within 0-10 range
                article['Risk_Score'] = round(max(0, min(10, base_score)), 2)
                
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
