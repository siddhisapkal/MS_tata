# Start Dashboard Script
# ====================

import subprocess
import sys
import os

def start_dashboard():
    """Start the Streamlit dashboard"""
    print("Starting Live Risk Analysis Dashboard...")
    print("="*50)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("Streamlit is installed")
    except ImportError:
        print("Installing Streamlit...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
    
    # Check if plotly is installed
    try:
        import plotly
        print("Plotly is installed")
    except ImportError:
        print("Installing Plotly...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
    
    # Start the dashboard
    print("\nStarting Streamlit dashboard...")
    print("Dashboard will open in your browser at: http://localhost:8501")
    print("Press Ctrl+C to stop the dashboard")
    print("="*50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_dashboard.py",
            "--server.headless", "false",
            "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Error starting dashboard: {e}")

if __name__ == "__main__":
    start_dashboard()
