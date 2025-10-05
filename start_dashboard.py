#!/usr/bin/env python3
"""
Complete System Startup Script - Starts API and Dashboard with config.yaml ports
"""

import os
import sys
import subprocess
import time
import threading
from pathlib import Path
import yaml
import signal

def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent / "config" / "config.yaml"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load config.yaml: {e}")
        return {}

def get_ports():
    """Get API and dashboard ports from config or environment variables"""
    config = load_config()
    
    # For cloud deployment, both services might use the same PORT env var
    if os.getenv("PORT"):
        # In cloud, we typically run one service per container
        port = int(os.getenv("PORT"))
        return port, port  # Same port, but only one service runs
    
    # Local development - read from config.yaml
    api_port = config.get('api', {}).get('port', 8080)
    dashboard_port = config.get('dashboard', {}).get('port', 8501)
    
    return api_port, dashboard_port

def start_api(api_port):
    """Start the API server in a subprocess"""
    print(f"🚀 Starting API server on port {api_port}...")
    
    api_path = Path(__file__).parent / "simple_api.py"
    
    # Set environment for API
    env = os.environ.copy()
    if not os.getenv("PORT"):  # Only set PORT if not already set (for local dev)
        env["PORT"] = str(api_port)
    
    try:
        # Start API as subprocess
        api_process = subprocess.Popen([
            sys.executable, str(api_path)
        ], env=env)
        
        print(f"✅ API server started (PID: {api_process.pid})")
        return api_process
        
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        return None

def start_dashboard(dashboard_port):
    """Start the Streamlit dashboard"""
    print(f"🎨 Starting Streamlit Dashboard on port {dashboard_port}...")
    
    dashboard_path = Path(__file__).parent / "src" / "dashboard" / "app.py"
    
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(dashboard_path),
        "--server.port", str(dashboard_port),
        "--server.address", "0.0.0.0" if os.getenv("PORT") else "127.0.0.1",
        "--server.headless", "true" if os.getenv("PORT") else "false"
    ]
    
    try:
        # Run streamlit (this blocks)
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("Dashboard stopped by user")
    except Exception as e:
        print(f"Dashboard error: {e}")
        sys.exit(1)

def main():
    """Start both API and Dashboard with proper port configuration"""
    api_port, dashboard_port = get_ports()
    
    if os.getenv("PORT"):
        print("🌥️  Cloud deployment mode (PORT env var detected)")
        print("ℹ️  In cloud mode, typically only one service runs per container")
    else:
        print("🏠 Local development mode")
        print(f"📊 API will run on: http://127.0.0.1:{api_port}")
        print(f"🎨 Dashboard will run on: http://127.0.0.1:{dashboard_port}")
    
    api_process = None
    
    try:
        # Start API first
        if not os.getenv("PORT") or os.getenv("SERVICE_TYPE") == "full":
            api_process = start_api(api_port)
            if api_process:
                # Give API time to start
                import time
                time.sleep(3)
        
        # Start dashboard (this blocks)
        start_dashboard(dashboard_port)
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        if api_process:
            api_process.terminate()
            api_process.wait()
        print("✅ All services stopped")
    except Exception as e:
        print(f"❌ System error: {e}")
        if api_process:
            api_process.terminate()
        sys.exit(1)

if __name__ == "__main__":
    main()