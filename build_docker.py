#!/usr/bin/env python3
"""
Centralized Docker build script that reads port configuration from config.yaml
This ensures ports are only defined in one place!
"""

import yaml
import subprocess
import sys
from pathlib import Path

def load_config():
    """Load port configuration from config.yaml"""
    config_path = Path("config/config.yaml")
    
    if not config_path.exists():
        print("❌ Config file not found: config/config.yaml")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    api_port = config.get('api', {}).get('port', 8080)
    dashboard_port = config.get('dashboard', {}).get('port', 8501)
    
    return api_port, dashboard_port

def build_api(api_port):
    """Build API container with configured port"""
    print(f"🔨 Building API container with port {api_port}")
    
    cmd = [
        "docker", "build",
        "--build-arg", f"API_PORT={api_port}",
        "-t", "stock-prediction-api",
        "-f", "Dockerfile",
        "."
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ API container built successfully")
        return True
    else:
        print(f"❌ API build failed: {result.stderr}")
        return False

def build_dashboard(dashboard_port):
    """Build dashboard container with configured port"""
    print(f"🔨 Building Dashboard container with port {dashboard_port}")
    
    cmd = [
        "docker", "build", 
        "--build-arg", f"DASHBOARD_PORT={dashboard_port}",
        "-t", "stock-prediction-dashboard",
        "-f", "Dockerfile.dashboard", 
        "."
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Dashboard container built successfully")
        return True
    else:
        print(f"❌ Dashboard build failed: {result.stderr}")
        return False

def main():
    """Main build function"""
    print("📦 Stock Prediction Docker Build")
    print("Reading port configuration from config/config.yaml...")
    
    try:
        api_port, dashboard_port = load_config()
        print("📋 Configuration loaded:")
        print(f"   API Port: {api_port}")
        print(f"   Dashboard Port: {dashboard_port}")
        
        # Build containers
        success = True
        
        if len(sys.argv) == 1 or "api" in sys.argv:
            success &= build_api(api_port)
        
        if len(sys.argv) == 1 or "dashboard" in sys.argv:
            success &= build_dashboard(dashboard_port)
        
        if success:
            print("\n🎉 All containers built successfully!")
            print("\n🚀 Run commands:")
            print(f"   API: docker run -p {api_port}:{api_port} stock-prediction-api")
            print(f"   Dashboard: docker run -p {dashboard_port}:{dashboard_port} stock-prediction-dashboard")
        else:
            print("\n❌ Some builds failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Build failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()