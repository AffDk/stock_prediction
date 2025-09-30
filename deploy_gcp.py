#!/usr/bin/env python3
"""
Deployment script for Google Cloud Platform.
Sets up GCS bucket and prepares for cloud deployment.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from src.utils.config import config
from src.utils.logging_config import get_logger


def run_command(command, description):
    """Run a shell command and handle errors."""
    logger = get_logger(__name__)
    logger.info(f"Starting: {description}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"Completed: {description}")
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {description}")
        logger.error(f"Command: {command}")
        logger.error(f"Error: {e.stderr}")
        return None


def setup_gcp_bucket():
    """Set up GCS bucket for data storage."""
    logger = get_logger(__name__)
    
    project_id = config.get('gcp.project_id')
    bucket_name = config.get('gcp.bucket_name')
    region = config.get('gcp.region')
    
    if not project_id or not bucket_name:
        logger.error("❌ GCP project ID or bucket name not configured")
        return False
    
    logger.info(f"🪣 Setting up GCS bucket: {bucket_name}")
    
    # Set project
    run_command(f"gcloud config set project {project_id}", "Setting GCP project")
    
    # Create bucket
    result = run_command(
        f"gsutil mb -p {project_id} -c STANDARD -l {region} gs://{bucket_name}",
        f"Creating GCS bucket {bucket_name}"
    )
    
    if result is None:
        # Bucket might already exist
        logger.info("Bucket might already exist, checking...")
        result = run_command(f"gsutil ls gs://{bucket_name}", "Checking bucket existence")
        if result is None:
            return False
    
    # Set up bucket structure
    bucket_dirs = [
        config.get('gcp.raw_data_prefix', 'raw_data'),
        config.get('gcp.processed_data_prefix', 'processed_data'),
        config.get('gcp.models_prefix', 'models'),
        config.get('gcp.logs_prefix', 'logs')
    ]
    
    for dir_name in bucket_dirs:
        placeholder_file = f"{dir_name}/.gitkeep"
        
        # Create local placeholder
        Path(placeholder_file).parent.mkdir(parents=True, exist_ok=True)
        Path(placeholder_file).touch()
        
        # Upload to bucket
        run_command(
            f"gsutil cp {placeholder_file} gs://{bucket_name}/{placeholder_file}",
            f"Creating bucket directory {dir_name}"
        )
        
        # Clean up local file
        Path(placeholder_file).parent.rmdir()
    
    logger.info("✅ GCS bucket setup completed")
    return True


def deploy_to_cloud_run():
    """Deploy API to Google Cloud Run."""
    logger = get_logger(__name__)
    
    project_id = config.get('gcp.project_id')
    region = config.get('gcp.region')
    
    logger.info("🚀 Deploying to Cloud Run")
    
    # Build container
    result = run_command(
        f"gcloud builds submit --tag gcr.io/{project_id}/stock-prediction-api",
        "Building container image"
    )
    
    if result is None:
        return False
    
    # Deploy to Cloud Run
    result = run_command(
        f"gcloud run deploy stock-prediction-api "
        f"--image gcr.io/{project_id}/stock-prediction-api "
        f"--platform managed "
        f"--region {region} "
        f"--allow-unauthenticated "
        f"--memory 2Gi "
        f"--cpu 2 "
        f"--timeout 300 "
        f"--concurrency 10",
        "Deploying to Cloud Run"
    )
    
    if result is None:
        return False
    
    logger.info("✅ Cloud Run deployment completed")
    return True


def setup_cost_controls():
    """Set up billing alerts and budget controls."""
    logger = get_logger(__name__)
    
    project_id = config.get('gcp.project_id')
    
    logger.info("💰 Setting up cost controls")
    
    # Create budget
    budget_config = {
        "displayName": "Stock Prediction Budget",
        "budgetFilter": {
            "projects": [f"projects/{project_id}"]
        },
        "amount": {
            "specifiedAmount": {
                "currencyCode": "USD",
                "units": "50"  # $50 monthly budget
            }
        },
        "thresholdRules": [
            {
                "thresholdPercent": 0.5,
                "spendBasis": "CURRENT_SPEND"
            },
            {
                "thresholdPercent": 0.8,
                "spendBasis": "CURRENT_SPEND"
            },
            {
                "thresholdPercent": 1.0,
                "spendBasis": "CURRENT_SPEND"
            }
        ]
    }
    
    # Save budget config
    budget_file = "budget-config.json"
    with open(budget_file, 'w') as f:
        json.dump(budget_config, f, indent=2)
    
    # Create budget (requires billing account ID - user needs to set this up manually)
    logger.info("Budget configuration saved to budget-config.json")
    logger.info("Please set up billing alerts manually in GCP Console")
    
    # Clean up
    Path(budget_file).unlink()
    
    return True


def create_dockerfile():
    """Create Dockerfile for container deployment."""
    dockerfile_content = """FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml .
COPY uv.lock .

# Install uv
RUN pip install uv

# Install dependencies
RUN uv sync --frozen

# Copy application code
COPY src/ src/
COPY config/ config/
COPY models/ models/

# Expose port
EXPOSE 8000

# Run the application
CMD ["uv", "run", "python", "-m", "src.api.main"]
"""
    
    with open("Dockerfile", 'w') as f:
        f.write(dockerfile_content)
    
    logger = get_logger(__name__)
    logger.info("✅ Dockerfile created")


def main():
    """Main deployment function."""
    logger = get_logger(__name__)
    logger.info("🚀 Starting GCP Deployment Setup")
    
    try:
        # Check if gcloud is installed
        result = run_command("gcloud version", "Checking Google Cloud SDK")
        if result is None:
            logger.error("❌ Google Cloud SDK not found. Please install it first.")
            logger.info("Visit: https://cloud.google.com/sdk/docs/install")
            return False
        
        # Step 1: Set up GCS bucket
        if not setup_gcp_bucket():
            logger.error("❌ Failed to set up GCS bucket")
            return False
        
        # Step 2: Create Dockerfile
        create_dockerfile()
        
        # Step 3: Set up cost controls
        setup_cost_controls()
        
        # Step 4: Deploy to Cloud Run (optional)
        deploy_cloud_run = input("Deploy to Cloud Run now? (y/N): ").lower().strip() == 'y'
        
        if deploy_cloud_run:
            if not deploy_to_cloud_run():
                logger.error("❌ Cloud Run deployment failed")
                return False
        
        logger.info("🎉 GCP setup completed successfully!")
        
        # Print next steps
        logger.info("\n📋 Next Steps:")
        logger.info("1. Upload your trained model to GCS:")
        logger.info(f"   gsutil -m cp -r models/ gs://{config.get('gcp.bucket_name')}/models/")
        logger.info("2. Set up billing alerts in GCP Console")
        logger.info("3. Monitor usage and costs regularly")
        logger.info("4. Consider using preemptible instances for training")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Deployment setup failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)