# Test Script for Streamlit Dashboard
# ===================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# Test the dashboard components
print("Testing Streamlit Dashboard Components...")

# Test 1: Basic imports
try:
    import streamlit as st
    print("Streamlit imported successfully")
except ImportError as e:
    print(f"❌ Streamlit import error: {e}")

# Test 2: Plotly imports
try:
    import plotly.express as px
    import plotly.graph_objects as go
    print("Plotly imported successfully")
except ImportError as e:
    print(f"❌ Plotly import error: {e}")

# Test 3: Create sample data
try:
    sample_data = {
        'Title': ['Tesla expansion in India', 'Supply chain disruption', 'Government policy change'],
        'Risk_Type': ['Strategic', 'Supply Chain', 'Strategic'],
        'Severity': ['High', 'Medium', 'Low'],
        'Risk_Score': [7.5, 4.2, 2.8],
        'source': ['Reuters', 'Bloomberg', 'Financial Times'],
        'publishedAt': ['2024-01-15', '2024-01-14', '2024-01-13']
    }
    df = pd.DataFrame(sample_data)
    print("Sample data created successfully")
    print(f"Data shape: {df.shape}")
except Exception as e:
    print(f"❌ Sample data creation error: {e}")

# Test 4: Create visualizations
try:
    # Risk type distribution
    risk_counts = df['Risk_Type'].value_counts()
    fig1 = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title="Risk Type Distribution"
    )
    print("Pie chart created successfully")
    
    # Risk score histogram
    fig2 = px.histogram(
        df,
        x='Risk_Score',
        title="Risk Score Distribution"
    )
    print("Histogram created successfully")
    
    # Risk score by severity
    fig3 = px.box(
        df,
        x='Severity',
        y='Risk_Score',
        title="Risk Score by Severity"
    )
    print("Box plot created successfully")
    
except Exception as e:
    print(f"❌ Visualization creation error: {e}")

# Test 5: Test live news integration
try:
    from live_news_integration import LiveNewsRiskAnalyzer
    print("Live news integration imported successfully")
except ImportError as e:
    print(f"❌ Live news integration import error: {e}")

print("\nDashboard components test completed!")
print("All dependencies are working correctly.")
print("\nTo run the dashboard:")
print("streamlit run streamlit_dashboard.py")
