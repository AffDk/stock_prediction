#!/usr/bin/env python3
"""
Comprehensive Service Manager for Stock Prediction Pipeline
Manages both API server and Streamlit dashboard with proper error handling
"""

import os
import sys
import time
import signal
import subprocess
import threading
from pathlib import Path
import psutil

class ServiceManager:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.api_process = None
        self.dashboard_process = None
        self.running = True
        
    def is_port_in_use(self, port: int) -> bool:
        """Check if a port is already in use"""
        for conn in psutil.net_connections():
            if conn.laddr.port == port:
                return True
        return False
    
    def kill_port(self, port: int):
        """Kill any process using the specified port"""
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                for conn in proc.info['connections'] or []:
                    if conn.laddr.port == port:
                        print(f"🔄 Killing process {proc.info['pid']} ({proc.info['name']}) using port {port}")
                        proc.kill()
                        time.sleep(1)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    
    def start_api_server(self):
        """Start the FastAPI server"""
        print("🚀 Starting API Server...")
        
        # Kill any existing processes on port 8081
        if self.is_port_in_use(8081):
            print("⚠️ Port 8081 is in use, cleaning up...")
            self.kill_port(8081)
            time.sleep(2)
        
        # Start API server
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "simple_api:app", 
            "--host", "0.0.0.0", 
            "--port", "8081",
            "--log-level", "info"
        ]
        
        try:
            self.api_process = subprocess.Popen(
                cmd, 
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Monitor API startup
            def monitor_api():
                for line in iter(self.api_process.stdout.readline, ''):
                    if not self.running:
                        break
                    print(f"[API] {line.strip()}")
                    if "Uvicorn running on" in line:
                        print("✅ API Server started successfully!")
            
            threading.Thread(target=monitor_api, daemon=True).start()
            
            # Wait for startup
            time.sleep(5)
            
            if self.api_process.poll() is None:
                print("✅ API Server is running on http://localhost:8081")
                return True
            else:
                print("❌ API Server failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start API server: {e}")
            return False
    
    def start_dashboard(self):
        """Start the Streamlit dashboard"""
        print("🎯 Starting Dashboard...")
        
        # Kill any existing processes on port 8501
        if self.is_port_in_use(8501):
            print("⚠️ Port 8501 is in use, cleaning up...")
            self.kill_port(8501)
            time.sleep(2)
        
        # Start dashboard
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "simple_dashboard.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        try:
            self.dashboard_process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Monitor dashboard startup
            def monitor_dashboard():
                for line in iter(self.dashboard_process.stdout.readline, ''):
                    if not self.running:
                        break
                    print(f"[DASH] {line.strip()}")
                    if "You can now view your Streamlit app" in line or "Network URL:" in line:
                        print("✅ Dashboard started successfully!")
            
            threading.Thread(target=monitor_dashboard, daemon=True).start()
            
            # Wait for startup
            time.sleep(8)
            
            if self.dashboard_process.poll() is None:
                print("✅ Dashboard is running on http://localhost:8501")
                return True
            else:
                print("❌ Dashboard failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start dashboard: {e}")
            return False
    
    def check_health(self):
        """Check if services are healthy"""
        try:
            import requests
            
            # Check API health
            api_healthy = False
            try:
                response = requests.get("http://localhost:8081/health", timeout=5)
                api_healthy = response.status_code == 200
            except:
                pass
            
            # Check dashboard health
            dashboard_healthy = False
            try:
                response = requests.get("http://localhost:8501", timeout=5)
                dashboard_healthy = response.status_code == 200
            except:
                pass
            
            return api_healthy, dashboard_healthy
            
        except ImportError:
            print("⚠️ requests module not available for health checks")
            return True, True  # Assume healthy if we can't check
    
    def stop_services(self):
        """Stop all services"""
        print("\n🛑 Stopping services...")
        self.running = False
        
        if self.api_process and self.api_process.poll() is None:
            print("🔄 Stopping API server...")
            self.api_process.terminate()
            try:
                self.api_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.api_process.kill()
        
        if self.dashboard_process and self.dashboard_process.poll() is None:
            print("🔄 Stopping dashboard...")
            self.dashboard_process.terminate()
            try:
                self.dashboard_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.dashboard_process.kill()
        
        # Clean up ports
        self.kill_port(8081)
        self.kill_port(8501)
        print("✅ All services stopped")
    
    def run(self):
        """Main service runner"""
        print("🎯 Stock Prediction Pipeline Service Manager")
        print("=" * 50)
        
        # Setup signal handlers
        def signal_handler(signum, frame):
            self.stop_services()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start services
        api_started = self.start_api_server()
        if not api_started:
            print("❌ Cannot continue without API server")
            return False
        
        dashboard_started = self.start_dashboard()
        if not dashboard_started:
            print("❌ Cannot continue without dashboard")
            self.stop_services()
            return False
        
        print("\n🎉 All services started successfully!")
        print("📊 Dashboard: http://localhost:8501")
        print("🔗 API: http://localhost:8081")
        print("📖 API Docs: http://localhost:8081/docs")
        print("\nPress Ctrl+C to stop all services...")
        
        # Health monitoring loop
        try:
            while self.running:
                time.sleep(30)  # Check every 30 seconds
                
                api_healthy, dashboard_healthy = self.check_health()
                
                if not api_healthy:
                    print("⚠️ API server appears unhealthy")
                
                if not dashboard_healthy:
                    print("⚠️ Dashboard appears unhealthy")
                
                # Check if processes are still running
                if self.api_process and self.api_process.poll() is not None:
                    print("❌ API process died, restarting...")
                    self.start_api_server()
                
                if self.dashboard_process and self.dashboard_process.poll() is not None:
                    print("❌ Dashboard process died, restarting...")
                    self.start_dashboard()
                    
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_services()
        
        return True

if __name__ == "__main__":
    manager = ServiceManager()
    success = manager.run()
    sys.exit(0 if success else 1)