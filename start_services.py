#!/usr/bin/env python3
"""
Start both API and Dashboard services for local testing
"""
import subprocess
import time
import requests
import sys
import os

def start_api():
    """Start the API server in a separate process"""
    print("Starting API server...")
    api_process = subprocess.Popen([
        "powershell", "-Command", 
        f"cd '{os.getcwd()}'; uv run uvicorn simple_api:app --host 0.0.0.0 --port 8081"
    ], creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0)
    
    # Wait for API to start
    print("Waiting for API to start...")
    for i in range(30):  # Wait up to 30 seconds
        try:
            response = requests.get("http://localhost:8081/health", timeout=1)
            if response.status_code == 200:
                print("API server is running at http://localhost:8081")
                print(f"   Health: {response.json()}")
                return api_process
        except Exception:
            time.sleep(1)
    
    print("API server failed to start")
    return None

def start_dashboard():
    """Start the dashboard in a separate process"""
    print("Starting Dashboard...")
    dashboard_process = subprocess.Popen([
        "powershell", "-Command", 
        f"cd '{os.getcwd()}'; uv run streamlit run simple_dashboard.py --server.port 8501 --server.headless true --browser.gatherUsageStats false"
    ], creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0)
    
    # Wait for dashboard to start
    print("Waiting for dashboard to start...")
    for i in range(30):  # Wait up to 30 seconds
        try:
            response = requests.get("http://localhost:8501", timeout=1)
            if response.status_code == 200:
                print("Dashboard is running at http://localhost:8501")
                return dashboard_process
        except Exception:
            time.sleep(1)
    
    print("Dashboard failed to start")
    return None

def test_services():
    """Test both services"""
    print("\nTesting Services:")
    print("-" * 30)
    
    # Test API
    try:
        health = requests.get("http://localhost:8081/health", timeout=5)
        print(f"API Health: {health.json()['status']}")
    except Exception as e:
        print(f"API Health: Error - {e}")
    
    try:
        prediction = requests.post(
            "http://localhost:8081/predict", 
            json={"symbol": "AAPL", "days_ahead": 7},
            timeout=5
        )
        pred_data = prediction.json()
        print(f"API Prediction: {pred_data['symbol']} -> ${pred_data['predicted_price']:.2f}")
    except Exception as e:
        print(f"API Prediction: Error - {e}")
    
    # Test Dashboard
    try:
        dashboard = requests.get("http://localhost:8501", timeout=5)
        print(f"Dashboard: Accessible ({len(dashboard.text)} bytes)")
    except Exception as e:
        print(f"Dashboard: Error - {e}")

def main():
    print("Stock Prediction Local Services Launcher")
    print("=" * 50)
    
    # Start API
    api_process = start_api()
    if not api_process:
        print("Failed to start API. Exiting.")
        return
    
    # Start Dashboard 
    dashboard_process = start_dashboard()
    if not dashboard_process:
        print("Failed to start Dashboard. Exiting.")
        return
    
    # Test services
    test_services()
    
    # Show URLs
    print("\nService URLs:")
    print("-" * 30)
    print("• API Documentation: http://localhost:8081/docs")
    print("• API Health: http://localhost:8081/health")
    print("• Dashboard: http://localhost:8501")
    
    print("\nServices are running!")
    print("Both services are running in separate PowerShell windows.")
    print("Close those windows to stop the services.")
    
    # Open browser
    try:
        import webbrowser
        print("\nOpening dashboard in browser...")
        webbrowser.open("http://localhost:8501")
    except Exception:
        print("\nManual: Open http://localhost:8501 in your browser")

if __name__ == "__main__":
    main()