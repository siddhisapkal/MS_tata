# Streamlit Dashboard for Live Risk Analysis
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import requests
from live_news_integration import LiveNewsRiskAnalyzer
from smart_news_filter import SmartNewsFilter
import joblib
import warnings
import google.generativeai as genai
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Live Risk Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high {
        color: #d62728;
        font-weight: bold;
    }
    .risk-medium {
        color: #ff7f0e;
        font-weight: bold;
    }
    .risk-low {
        color: #2ca02c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'news_data' not in st.session_state:
    st.session_state.news_data = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []

def load_models():
    """Load the trained models"""
    try:
        if st.session_state.analyzer is None:
            with st.spinner("Loading trained models..."):
                st.session_state.analyzer = LiveNewsRiskAnalyzer()
            st.success("Models loaded successfully!")
        return True
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return False

def fetch_live_news(api_key, query, num_articles, hours_back):
    """Fetch live news and analyze"""
    try:
        with st.spinner("Fetching live news..."):
            articles = st.session_state.analyzer.fetch_news(
                api_key=api_key,
                query=query,
                num_articles=num_articles,
                hours_back=hours_back
            )
        
        if not articles:
            st.warning("No articles found. Try different search terms or time range.")
            return []
        
        with st.spinner("Analyzing articles for risk..."):
            results = st.session_state.analyzer.analyze_articles(articles)
        
        st.session_state.news_data = articles
        st.session_state.analysis_results = results
        
        return results
    except Exception as e:
        st.error(f"Error fetching/analyzing news: {e}")
        return []

def fetch_smart_filtered_news(api_key, query, num_articles, hours_back):
    """Fetch real news from NewsAPI and use smart filtering"""
    try:
        with st.spinner(f"Fetching fresh news from last {hours_back//24} days..."):
            # Always create a new filter instance to avoid caching
            smart_filter = SmartNewsFilter(api_key)
            articles = smart_filter.process_news_pipeline(
                query=query,
                num_articles=num_articles,
                hours_back=hours_back,
                min_relevance=0.6
            )
        
        if not articles:
            st.warning("No relevant articles found after smart filtering.")
            return []
        
        with st.spinner("Analyzing filtered articles for risk..."):
            # The smart filter already provides risk analysis, but we can enhance it
            results = st.session_state.analyzer.analyze_articles(articles)
        
        # Always update session state with fresh data
        st.session_state.news_data = articles
        st.session_state.analysis_results = results
        
        # Show fresh data info with search enhancement stats
        if 'enhanced_query' in st.session_state:
            st.info(f"✅ Fetched {len(articles)} fresh articles using AI-enhanced search from last {hours_back//24} days")
            st.success(f"🎯 Search enhanced from '{user_query}' to '{query[:100]}...'")
        else:
            st.info(f"✅ Fetched {len(articles)} fresh articles from last {hours_back//24} days")
        
        return results
    except Exception as e:
        st.error(f"Error fetching/analyzing smart filtered news: {e}")
        return []

def get_gemini_search_enhancement(gemini_key, user_query):
    """Use Gemini AI to enhance search queries with relevant keywords"""
    try:
        if not gemini_key or len(gemini_key) < 10:
            return user_query  # Return original query if no API key
        
        # Configure Gemini
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Create enhancement prompt
        prompt = f"""
You are an expert in automotive industry and Tata Motors risk analysis. 

Given the user's search query: "{user_query}"

Generate an enhanced search query that will find the most relevant news articles for Tata Motors risk analysis. Include:

1. **Core automotive/EV keywords** (electric vehicle, automotive, car, vehicle, etc.)
2. **Tata Motors specific terms** (Tata Motors, JLR, Jaguar Land Rover, Nexon, Harrier, etc.)
3. **Risk-related terms** (supply chain, lithium, semiconductor, chip shortage, battery, charging, policy, regulation, subsidy, FAME, etc.)
4. **Competitor terms** (Mahindra, Maruti, Hyundai, Toyota, Honda, etc.)
5. **Industry terms** (automotive industry, auto industry, passenger vehicle, commercial vehicle, etc.)

Return ONLY the enhanced search query as a single string, optimized for news API search. Make it comprehensive but focused on Tata Motors risk analysis.

Example: If user searches "lithium", return something like "Tata Motors lithium battery electric vehicle automotive supply chain EV charging infrastructure"
"""
        
        response = model.generate_content(prompt)
        enhanced_query = response.text.strip()
        
        # Clean up the response
        enhanced_query = enhanced_query.replace('"', '').replace("'", '')
        
        return enhanced_query
        
    except Exception as e:
        print(f"Error enhancing search query: {e}")
        return user_query  # Return original query if enhancement fails

def get_gemini_analysis(gemini_key, articles, analysis_results):
    """Get detailed analysis from Gemini AI"""
    try:
        if not gemini_key or len(gemini_key) < 10:
            return "Please provide a valid Gemini API key for AI analysis."
        
        # Configure Gemini
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Prepare news data for analysis
        news_summary = ""
        for i, article in enumerate(articles[:10], 1):  # Limit to top 10 articles
            news_summary += f"\n{i}. {article['Title']}\n"
            news_summary += f"   Source: {article['source']}\n"
            news_summary += f"   Published: {article.get('publishedAt', 'N/A')}\n"
            news_summary += f"   URL: {article.get('url', 'N/A')}\n"
            news_summary += f"   Risk Type: {article.get('Risk_Type', 'N/A')}\n"
            news_summary += f"   Severity: {article.get('Severity', 'N/A')}\n"
            news_summary += f"   Risk Score: {article.get('Risk_Score', 'N/A')}\n"
            news_summary += f"   Explanation: {article.get('Explanation', '')[:200]}...\n"
            news_summary += "   " + "="*50 + "\n"
        
        # Create analysis prompt
        prompt = f"""
You are an expert risk analyst for Tata Motors. Analyze the following news articles and provide a comprehensive risk assessment:

{news_summary}

Please provide:

1. **EXECUTIVE SUMMARY** (2-3 sentences)
2. **KEY RISKS IDENTIFIED** (Top 5 risks with severity)
3. **IMMEDIATE ACTIONS REQUIRED** (What Tata Motors should do now)
4. **STRATEGIC RECOMMENDATIONS** (Long-term strategies)
5. **COMPETITIVE LANDSCAPE** (How competitors are affecting Tata Motors)
6. **SUPPLY CHAIN IMPACT** (Any supply chain risks identified)
7. **REGULATORY CONCERNS** (Policy changes affecting the company)
8. **FINANCIAL IMPLICATIONS** (Potential financial impact)

Focus on actionable insights and specific recommendations. Be concise but comprehensive.
"""
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Error getting Gemini analysis: {str(e)}"

def create_risk_distribution_chart(results_df):
    """Create risk distribution charts"""
    if results_df.empty:
        return None
    
    # Risk Type Distribution
    risk_type_counts = results_df['Risk_Type'].value_counts()
    
    fig1 = px.pie(
        values=risk_type_counts.values,
        names=risk_type_counts.index,
        title="Risk Type Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig1.update_traces(textposition='inside', textinfo='percent+label')
    
    # Severity Distribution
    severity_counts = results_df['Severity'].value_counts()
    
    fig2 = px.bar(
        x=severity_counts.index,
        y=severity_counts.values,
        title="Severity Distribution",
        color=severity_counts.values,
        color_continuous_scale="RdYlGn_r"
    )
    fig2.update_layout(xaxis_title="Severity Level", yaxis_title="Count")
    
    return fig1, fig2

def create_risk_score_analysis(results_df):
    """Create risk score analysis charts"""
    if results_df.empty:
        return None
    
    # Risk Score Distribution
    fig1 = px.histogram(
        results_df,
        x='Risk_Score',
        nbins=20,
        title="Risk Score Distribution",
        color_discrete_sequence=['#1f77b4']
    )
    fig1.update_layout(xaxis_title="Risk Score", yaxis_title="Frequency")
    
    # Risk Score by Risk Type
    fig2 = px.box(
        results_df,
        x='Risk_Type',
        y='Risk_Score',
        title="Risk Score by Risk Type",
        color='Risk_Type',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig2.update_layout(xaxis_title="Risk Type", yaxis_title="Risk Score")
    
    # Risk Score by Severity
    fig3 = px.violin(
        results_df,
        x='Severity',
        y='Risk_Score',
        title="Risk Score by Severity",
        color='Severity',
        color_discrete_sequence=['#d62728', '#ff7f0e', '#2ca02c', '#9467bd']
    )
    fig3.update_layout(xaxis_title="Severity", yaxis_title="Risk Score")
    
    return fig1, fig2, fig3

def create_time_series_analysis(results_df):
    """Create time series analysis"""
    if results_df.empty:
        return None
    
    # Convert publishedAt to datetime
    results_df['publishedAt'] = pd.to_datetime(results_df['publishedAt'], errors='coerce')
    results_df = results_df.dropna(subset=['publishedAt'])
    
    if results_df.empty:
        return None
    
    # Group by hour and calculate average risk score
    results_df['hour'] = results_df['publishedAt'].dt.floor('H')
    hourly_risk = results_df.groupby('hour')['Risk_Score'].mean().reset_index()
    
    fig = px.line(
        hourly_risk,
        x='hour',
        y='Risk_Score',
        title="Average Risk Score Over Time",
        markers=True
    )
    fig.update_layout(xaxis_title="Time", yaxis_title="Average Risk Score")
    
    return fig

def create_heatmap_analysis(results_df):
    """Create risk heatmap"""
    if results_df.empty:
        return None
    
    # Create risk matrix
    risk_matrix = results_df.groupby(['Risk_Type', 'Severity']).size().unstack(fill_value=0)
    
    fig = px.imshow(
        risk_matrix.values,
        x=risk_matrix.columns,
        y=risk_matrix.index,
        title="Risk Type vs Severity Heatmap",
        color_continuous_scale="RdYlGn_r",
        aspect="auto"
    )
    fig.update_layout(
        xaxis_title="Severity",
        yaxis_title="Risk Type"
    )
    
    return fig

def create_source_analysis(results_df):
    """Create source analysis"""
    if results_df.empty:
        return None
    
    # Top sources by risk score
    source_risk = results_df.groupby('source')['Risk_Score'].agg(['mean', 'count']).reset_index()
    source_risk = source_risk[source_risk['count'] >= 2]  # Only sources with 2+ articles
    
    if source_risk.empty:
        return None
    
    fig = px.scatter(
        source_risk,
        x='count',
        y='mean',
        size='count',
        hover_name='source',
        title="Source Analysis: Article Count vs Average Risk Score",
        color='mean',
        color_continuous_scale="RdYlGn_r"
    )
    fig.update_layout(
        xaxis_title="Number of Articles",
        yaxis_title="Average Risk Score"
    )
    
    return fig

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Live Risk Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    
    # API Configuration
    st.sidebar.subheader("API Configuration")
    api_type = st.sidebar.selectbox(
        "News Source",
        ["Smart Filter (NewsAPI + AI)", "NewsAPI.ai Only"],
        index=0
    )
    
    api_key = st.sidebar.text_input(
        "NewsAPI.ai Key",
        value="74830ae7-dea2-498e-b538-344e7a149eff",
        type="password",
        help="Your NewsAPI.ai API key for fetching live news"
    )
    
    gemini_key = st.sidebar.text_input(
        "Gemini API Key",
        value="AIzaSyAQDAZEPDDWwNx3loUxWLQfUjytNASG7ac",
        type="password",
        help="Your Gemini API key for AI-powered analysis and chatbot"
    )
    
    # Search Configuration
    st.sidebar.subheader("Search Configuration")
    user_query = st.sidebar.text_input("Search Query", value="Tata Motors")
    
    # AI-Enhanced Search
    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        if st.button("🤖 Enhance Search with AI"):
            if gemini_key and len(gemini_key) > 10:
                with st.spinner("Enhancing search query with AI..."):
                    enhanced_query = get_gemini_search_enhancement(gemini_key, user_query)
                    st.session_state.enhanced_query = enhanced_query
                    st.success("Search enhanced!")
            else:
                st.error("Please provide a valid Gemini API key for search enhancement.")
    
    with col2:
        if st.button("🔄 Reset"):
            if 'enhanced_query' in st.session_state:
                del st.session_state.enhanced_query
            st.success("Search reset!")
    
    # Use enhanced query if available, otherwise use original
    query = st.session_state.get('enhanced_query', user_query)
    
    # Show the actual query being used
    if 'enhanced_query' in st.session_state:
        st.sidebar.info(f"🔍 Enhanced Query: {query[:50]}...")
    else:
        st.sidebar.info(f"🔍 Using Query: {query}")
    
    # Quick search suggestions
    st.sidebar.markdown("**💡 Quick Search Suggestions:**")
    suggestions = [
        "lithium", "semiconductor", "EV policy", "supply chain", 
        "competition", "battery", "charging", "FAME scheme"
    ]
    
    cols = st.sidebar.columns(2)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(f"🔍 {suggestion}", key=f"suggest_{i}"):
                st.session_state.user_query = suggestion
                st.rerun()
    
    num_articles = st.sidebar.slider("Number of Articles", 5, 50, 20)
    days_back = st.sidebar.slider("Days Back", 1, 30, 3)
    
    # Show time range
    from datetime import datetime, timedelta
    to_time = datetime.now()
    from_time = to_time - timedelta(days=days_back)
    st.sidebar.info(f"Searching from: {from_time.strftime('%Y-%m-%d %H:%M')} to {to_time.strftime('%Y-%m-%d %H:%M')}")
    
    # Convert days to hours for API
    hours_back = days_back * 24
    
    # Load Models
    if st.sidebar.button("🔄 Load Models"):
        load_models()
    
    # Fetch News
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("📰 Fetch & Analyze News"):
            if st.session_state.analyzer is None:
                st.error("Please load models first!")
            else:
                if api_type == "Smart Filter (NewsAPI + AI)":
                    results = fetch_smart_filtered_news(api_key, query, num_articles, hours_back)
                else:
                    results = fetch_live_news(api_key, query, num_articles, hours_back)
                if results:
                    st.success(f"Successfully analyzed {len(results)} articles!")
                    # Store timestamp
                    st.session_state.last_fetch_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with col2:
        if st.button("🔄 Refresh Data"):
            if st.session_state.analyzer is None:
                st.error("Please load models first!")
            else:
                if api_type == "Smart Filter (NewsAPI + AI)":
                    results = fetch_smart_filtered_news(api_key, query, num_articles, hours_back)
                else:
                    results = fetch_live_news(api_key, query, num_articles, hours_back)
                if results:
                    st.success(f"Successfully refreshed {len(results)} articles!")
                    st.session_state.last_fetch_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Show last fetch time and clear data button
    if hasattr(st.session_state, 'last_fetch_time'):
        st.sidebar.info(f"Last fetched: {st.session_state.last_fetch_time}")
    
    if st.sidebar.button("🗑️ Clear Data"):
        st.session_state.analysis_results = None
        st.session_state.news_data = None
        st.session_state.last_fetch_time = None
        st.success("Data cleared! Fetch new data to see results.")
    
    # Main Content
    if st.session_state.analysis_results:
        results_df = pd.DataFrame(st.session_state.analysis_results)
        
        # Display All Extracted News at the Top
        st.markdown("---")
        st.markdown("## 📰 All Extracted News Articles")
        
        # Show search context and summary
        search_context = f"**Topic:** {query} | **Time Period:** Last {days_back} days | **Articles Found:** {len(results_df)}"
        st.info(search_context)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📰 Total Articles", len(results_df))
        
        with col2:
            high_count = len(results_df[results_df['Severity'] == 'High']) if 'Severity' in results_df.columns else 0
            st.metric("🔴 High Risk", high_count)
        
        with col3:
            medium_count = len(results_df[results_df['Severity'] == 'Medium']) if 'Severity' in results_df.columns else 0
            st.metric("🟡 Medium Risk", medium_count)
        
        with col4:
            low_count = len(results_df[results_df['Severity'] == 'Low']) if 'Severity' in results_df.columns else 0
            st.metric("🟢 Low Risk", low_count)
        
        # Risk type distribution
        if 'Risk_Type' in results_df.columns:
            risk_types = results_df['Risk_Type'].value_counts()
            if not risk_types.empty:
                st.markdown("**📊 Risk Type Distribution:**")
                risk_cols = st.columns(len(risk_types))
                for i, (risk_type, count) in enumerate(risk_types.items()):
                    with risk_cols[i]:
                        st.metric(risk_type, count)
        
        st.markdown("---")
        
        # Create expandable sections for different risk levels
        high_risk = results_df[results_df['Severity'] == 'High'] if 'Severity' in results_df.columns else pd.DataFrame()
        medium_risk = results_df[results_df['Severity'] == 'Medium'] if 'Severity' in results_df.columns else pd.DataFrame()
        low_risk = results_df[results_df['Severity'] == 'Low'] if 'Severity' in results_df.columns else pd.DataFrame()
        no_severity = results_df[results_df['Severity'].isna()] if 'Severity' in results_df.columns else pd.DataFrame()
        
        # High Risk Articles
        if not high_risk.empty:
            with st.expander(f"🔴 High Risk Articles ({len(high_risk)})", expanded=True):
                for idx, article in high_risk.iterrows():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        # Make title clickable
                        if pd.notna(article.get('url', '')) and article['url'] != 'N/A' and article['url'] != '':
                            st.markdown(f"**[{article['Title']}]({article['url']})**")
                        else:
                            st.markdown(f"**{article['Title']}**")
                        
                        # Show source and date
                        source_date = f"📰 {article.get('source', 'Unknown')}"
                        if pd.notna(article.get('publishedAt', '')) and article['publishedAt'] != 'N/A':
                            source_date += f" | 📅 {article['publishedAt']}"
                        st.caption(source_date)
                        
                        # Show risk details
                        risk_info = f"🎯 Risk: {article.get('Risk_Type', 'N/A')} | ⚠️ Severity: {article.get('Severity', 'N/A')} | 📊 Score: {article.get('Risk_Score', 'N/A')}"
                        st.caption(risk_info)
                        
                        # Show explanation snippet
                        if pd.notna(article.get('Explanation', '')):
                            explanation = article['Explanation'][:200] + "..." if len(str(article['Explanation'])) > 200 else article['Explanation']
                            st.text(explanation)
                    
                    with col2:
                        # Risk score indicator
                        score = article.get('Risk_Score', 0)
                        if score >= 8:
                            st.markdown("🔴 **HIGH**")
                        elif score >= 5:
                            st.markdown("🟡 **MED**")
                        else:
                            st.markdown("🟢 **LOW**")
                    
                    st.markdown("---")
        
        # Medium Risk Articles
        if not medium_risk.empty:
            with st.expander(f"🟡 Medium Risk Articles ({len(medium_risk)})"):
                for idx, article in medium_risk.iterrows():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        if pd.notna(article.get('url', '')) and article['url'] != 'N/A' and article['url'] != '':
                            st.markdown(f"**[{article['Title']}]({article['url']})**")
                        else:
                            st.markdown(f"**{article['Title']}**")
                        
                        source_date = f"📰 {article.get('source', 'Unknown')}"
                        if pd.notna(article.get('publishedAt', '')) and article['publishedAt'] != 'N/A':
                            source_date += f" | 📅 {article['publishedAt']}"
                        st.caption(source_date)
                        
                        risk_info = f"🎯 Risk: {article.get('Risk_Type', 'N/A')} | ⚠️ Severity: {article.get('Severity', 'N/A')} | 📊 Score: {article.get('Risk_Score', 'N/A')}"
                        st.caption(risk_info)
                        
                        if pd.notna(article.get('Explanation', '')):
                            explanation = article['Explanation'][:150] + "..." if len(str(article['Explanation'])) > 150 else article['Explanation']
                            st.text(explanation)
                    
                    with col2:
                        score = article.get('Risk_Score', 0)
                        if score >= 8:
                            st.markdown("🔴 **HIGH**")
                        elif score >= 5:
                            st.markdown("🟡 **MED**")
                        else:
                            st.markdown("🟢 **LOW**")
                    
                    st.markdown("---")
        
        # Low Risk Articles
        if not low_risk.empty:
            with st.expander(f"🟢 Low Risk Articles ({len(low_risk)})"):
                for idx, article in low_risk.iterrows():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        if pd.notna(article.get('url', '')) and article['url'] != 'N/A' and article['url'] != '':
                            st.markdown(f"**[{article['Title']}]({article['url']})**")
                        else:
                            st.markdown(f"**{article['Title']}**")
                        
                        source_date = f"📰 {article.get('source', 'Unknown')}"
                        if pd.notna(article.get('publishedAt', '')) and article['publishedAt'] != 'N/A':
                            source_date += f" | 📅 {article['publishedAt']}"
                        st.caption(source_date)
                        
                        risk_info = f"🎯 Risk: {article.get('Risk_Type', 'N/A')} | ⚠️ Severity: {article.get('Severity', 'N/A')} | 📊 Score: {article.get('Risk_Score', 'N/A')}"
                        st.caption(risk_info)
                    
                    with col2:
                        score = article.get('Risk_Score', 0)
                        if score >= 8:
                            st.markdown("🔴 **HIGH**")
                        elif score >= 5:
                            st.markdown("🟡 **MED**")
                        else:
                            st.markdown("🟢 **LOW**")
                    
                    st.markdown("---")
        
        # No Severity Articles
        if not no_severity.empty:
            with st.expander(f"⚪ Other Articles ({len(no_severity)})"):
                for idx, article in no_severity.iterrows():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        if pd.notna(article.get('url', '')) and article['url'] != 'N/A' and article['url'] != '':
                            st.markdown(f"**[{article['Title']}]({article['url']})**")
                        else:
                            st.markdown(f"**{article['Title']}**")
                        
                        source_date = f"📰 {article.get('source', 'Unknown')}"
                        if pd.notna(article.get('publishedAt', '')) and article['publishedAt'] != 'N/A':
                            source_date += f" | 📅 {article['publishedAt']}"
                        st.caption(source_date)
                        
                        risk_info = f"🎯 Risk: {article.get('Risk_Type', 'N/A')} | ⚠️ Severity: {article.get('Severity', 'N/A')} | 📊 Score: {article.get('Risk_Score', 'N/A')}"
                        st.caption(risk_info)
                    
                    with col2:
                        score = article.get('Risk_Score', 0)
                        if score >= 8:
                            st.markdown("🔴 **HIGH**")
                        elif score >= 5:
                            st.markdown("🟡 **MED**")
                        else:
                            st.markdown("🟢 **LOW**")
                    
                    st.markdown("---")
        
        # Add Gemini AI Analysis Section
        st.markdown("---")
        st.markdown("## 🤖 AI-Powered Risk Analysis")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🧠 Get Detailed AI Analysis", type="primary"):
                if gemini_key and len(gemini_key) > 10:
                    with st.spinner("🤖 Gemini AI is analyzing the news and generating detailed insights..."):
                        analysis = get_gemini_analysis(gemini_key, st.session_state.news_data, st.session_state.analysis_results)
                        st.session_state.gemini_analysis = analysis
                else:
                    st.error("Please provide a valid Gemini API key for AI analysis.")
        
        with col2:
            if st.button("🗑️ Clear Analysis"):
                if 'gemini_analysis' in st.session_state:
                    del st.session_state.gemini_analysis
                st.success("Analysis cleared!")
        
        # Display Gemini Analysis
        if 'gemini_analysis' in st.session_state:
            st.markdown("### 📊 AI Analysis Results")
            st.markdown(st.session_state.gemini_analysis)
        
        st.markdown("---")
        
        # Key Metrics
        st.subheader("📈 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Articles", len(results_df))
        
        with col2:
            avg_risk = results_df['Risk_Score'].mean()
            st.metric("Average Risk Score", f"{avg_risk:.2f}")
        
        with col3:
            high_risk = len(results_df[results_df['Risk_Score'] > 7.0])
            st.metric("High Risk Articles", high_risk)
        
        with col4:
            most_common_risk = results_df['Risk_Type'].mode().iloc[0] if not results_df.empty else "N/A"
            st.metric("Most Common Risk", most_common_risk)
        
        # Risk Distribution Charts
        st.subheader("📊 Risk Distribution Analysis")
        
        fig1, fig2 = create_risk_distribution_chart(results_df)
        if fig1 and fig2:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                st.plotly_chart(fig2, use_container_width=True)
        
        # Risk Score Analysis
        st.subheader("🎯 Risk Score Analysis")
        
        fig1, fig2, fig3 = create_risk_score_analysis(results_df)
        if fig1 and fig2 and fig3:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                st.plotly_chart(fig2, use_container_width=True)
            
            st.plotly_chart(fig3, use_container_width=True)
        
        # Time Series Analysis
        st.subheader("⏰ Time Series Analysis")
        time_fig = create_time_series_analysis(results_df)
        if time_fig:
            st.plotly_chart(time_fig, use_container_width=True)
        
        # Heatmap Analysis
        st.subheader("🔥 Risk Heatmap")
        heatmap_fig = create_heatmap_analysis(results_df)
        if heatmap_fig:
            st.plotly_chart(heatmap_fig, use_container_width=True)
        
        # Source Analysis
        st.subheader("📰 Source Analysis")
        source_fig = create_source_analysis(results_df)
        if source_fig:
            st.plotly_chart(source_fig, use_container_width=True)
        
        # Detailed Results Table
        st.subheader("📋 Detailed Results")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            risk_type_filter = st.selectbox("Filter by Risk Type", ["All"] + list(results_df['Risk_Type'].unique()))
        with col2:
            severity_filter = st.selectbox("Filter by Severity", ["All"] + list(results_df['Severity'].unique()))
        with col3:
            min_risk_score = st.slider("Minimum Risk Score", 0.0, 10.0, 0.0)
        
        # Apply filters
        filtered_df = results_df.copy()
        if risk_type_filter != "All":
            filtered_df = filtered_df[filtered_df['Risk_Type'] == risk_type_filter]
        if severity_filter != "All":
            filtered_df = filtered_df[filtered_df['Severity'] == severity_filter]
        filtered_df = filtered_df[filtered_df['Risk_Score'] >= min_risk_score]
        
        # Display filtered results with clickable links
        display_df = filtered_df[['Title', 'Risk_Type', 'Severity', 'Risk_Score', 'source', 'publishedAt', 'url']].copy()
        display_df['Risk_Score'] = display_df['Risk_Score'].round(2)
        
        # Add clickable links
        def make_clickable(url, title):
            if pd.notna(url) and url != 'N/A' and url != '':
                return f'<a href="{url}" target="_blank" style="color: #1f77b4; text-decoration: none;">{title[:60]}...</a>'
            return title[:60] + "..."
        
        display_df['Title_Link'] = display_df.apply(lambda x: make_clickable(x['url'], x['Title']), axis=1)
        display_df = display_df[['Title_Link', 'Risk_Type', 'Severity', 'Risk_Score', 'source', 'publishedAt']]
        display_df.columns = ['Title (Click to Read)', 'Risk Type', 'Severity', 'Risk Score', 'Source', 'Published']
        
        # Color code risk scores
        def color_risk_score(val):
            if val >= 7.0:
                return 'background-color: #ffebee'
            elif val >= 4.0:
                return 'background-color: #fff3e0'
            else:
                return 'background-color: #e8f5e8'
        
        styled_df = display_df.style.applymap(color_risk_score, subset=['Risk Score'])
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"risk_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Summary Report
        st.subheader("📄 Summary Report")
        
        # Create summary statistics
        summary_stats = {
            "Total Articles": len(results_df),
            "Average Risk Score": f"{results_df['Risk_Score'].mean():.2f}",
            "Highest Risk Score": f"{results_df['Risk_Score'].max():.2f}",
            "Lowest Risk Score": f"{results_df['Risk_Score'].min():.2f}",
            "High Risk Articles (>7.0)": len(results_df[results_df['Risk_Score'] > 7.0]),
            "Medium Risk Articles (4.0-7.0)": len(results_df[(results_df['Risk_Score'] >= 4.0) & (results_df['Risk_Score'] <= 7.0)]),
            "Low Risk Articles (<4.0)": len(results_df[results_df['Risk_Score'] < 4.0])
        }
        
        for key, value in summary_stats.items():
            st.write(f"**{key}**: {value}")
    
    else:
        st.info("👆 Please load models and fetch news to see the dashboard!")
        
        # Show sample data structure
        st.subheader("📋 Expected Data Structure")
        sample_data = {
            "Title": "Tesla announces major expansion in Indian market",
            "Explanation": "Tesla plans to invest $2 billion in Indian manufacturing facilities",
            "Affected_Nodes": [],
            "Risk_Type": "Strategic",
            "Severity": "High",
            "Risk_Score": 7.5,
            "source": "Reuters",
            "publishedAt": "2024-01-15T10:30:00Z"
        }
        st.json(sample_data)

if __name__ == "__main__":
    main()
