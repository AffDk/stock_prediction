"""
Simplified GCP deployment script without Unicode issues
"""
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run command with error handling"""
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"SUCCESS: {description}")
        if result.stdout.strip():
            print(f"Output: {result.stdout.strip()}")
        return True
    else:
        print(f"ERROR: {description} failed")
        if result.stderr:
            print(f"Error: {result.stderr}")
        return False

def deploy_api():
    """Deploy API service to Cloud Run"""
    print("\n=== DEPLOYING API SERVICE ===")
    
    commands = [
        ("docker build -t gcr.io/proud-curve-473118-h6/stock-prediction-api:latest .", 
         "Building API Docker image"),
        ("docker push gcr.io/proud-curve-473118-h6/stock-prediction-api:latest", 
         "Pushing API image to registry"),
        ("gcloud run deploy stock-prediction-api --image gcr.io/proud-curve-473118-h6/stock-prediction-api:latest --platform managed --region us-central1 --allow-unauthenticated --port 8000", 
         "Deploying API to Cloud Run")
    ]
    
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            return False
    
    print("\nAPI SERVICE DEPLOYED SUCCESSFULLY!")
    return True

def deploy_dashboard():
    """Deploy dashboard service to Cloud Run"""
    print("\n=== DEPLOYING DASHBOARD SERVICE ===")
    
    commands = [
        ("docker build -f Dockerfile.dashboard -t gcr.io/proud-curve-473118-h6/stock-prediction-dashboard:latest .", 
         "Building Dashboard Docker image"),
        ("docker push gcr.io/proud-curve-473118-h6/stock-prediction-dashboard:latest", 
         "Pushing Dashboard image to registry"),
        ("gcloud run deploy stock-prediction-dashboard --image gcr.io/proud-curve-473118-h6/stock-prediction-dashboard:latest --platform managed --region us-central1 --allow-unauthenticated --port 8501", 
         "Deploying Dashboard to Cloud Run")
    ]
    
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            return False
    
    print("\nDASHBOARD SERVICE DEPLOYED SUCCESSFULLY!")
    return True

def main():
    """Main deployment function"""
    print("GCP DEPLOYMENT TOOL")
    print("==================")
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python deploy_simple.py api       # Deploy API only")
        print("  python deploy_simple.py dashboard # Deploy Dashboard only") 
        print("  python deploy_simple.py both      # Deploy both services")
        return
    
    action = sys.argv[1].lower()
    
    if action == "api":
        deploy_api()
    elif action == "dashboard":
        deploy_dashboard()
    elif action == "both":
        if deploy_api():
            deploy_dashboard()
    else:
        print(f"Unknown action: {action}")
        return

if __name__ == "__main__":
    main()