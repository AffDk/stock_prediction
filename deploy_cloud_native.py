"""
Cloud-native deployment script using Google Cloud Build (no local Docker required)
"""
import subprocess
import sys

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

def deploy_with_cloud_build():
    """Deploy using Google Cloud Build (no local Docker needed)"""
    print("\n=== CLOUD-NATIVE DEPLOYMENT ===")
    print("Using Google Cloud Build to build Docker images in the cloud")
    
    commands = [
        # Build API image in cloud
        ("gcloud builds submit --tag gcr.io/proud-curve-473118-h6/stock-prediction-api:latest .", 
         "Building API Docker image in Google Cloud"),
        
        # Deploy API to Cloud Run
        ("gcloud run deploy stock-prediction-api --image gcr.io/proud-curve-473118-h6/stock-prediction-api:latest --platform managed --region us-central1 --allow-unauthenticated --port 8000 --memory 4Gi --cpu 2", 
         "Deploying API to Cloud Run"),
        
        # Build Dashboard image in cloud  
        ("gcloud builds submit --tag gcr.io/proud-curve-473118-h6/stock-prediction-dashboard:latest --file Dockerfile.dashboard .", 
         "Building Dashboard Docker image in Google Cloud"),
        
        # Deploy Dashboard to Cloud Run
        ("gcloud run deploy stock-prediction-dashboard --image gcr.io/proud-curve-473118-h6/stock-prediction-dashboard:latest --platform managed --region us-central1 --allow-unauthenticated --port 8501 --memory 2Gi --cpu 1", 
         "Deploying Dashboard to Cloud Run")
    ]
    
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            return False
    
    print("\n✅ DEPLOYMENT SUCCESSFUL!")
    print("\n🎉 Your services are now live:")
    print("📊 API: Check Cloud Console for API URL")  
    print("📈 Dashboard: Check Cloud Console for Dashboard URL")
    return True

def main():
    """Main deployment function"""
    print("CLOUD-NATIVE GCP DEPLOYMENT")
    print("==========================")
    print("No Docker installation required!")
    
    deploy_with_cloud_build()

if __name__ == "__main__":
    main()