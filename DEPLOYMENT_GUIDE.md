# 🚀 Stock Prediction - Cloud Deployment Guide

## 📋 Overview
Deploy your AI-powered stock prediction system to Google Cloud Run for global access and scalability.

## ✅ Prerequisites Checklist

- [x] **Trained Model**: ✅ Ready (96.5% accuracy)
- [x] **API Server**: ✅ Ready (FastAPI with health checks)  
- [x] **Dashboard**: ✅ Ready (Streamlit with real-time predictions)
- [ ] **Docker**: Install Docker Desktop
- [ ] **GCP CLI**: Install Google Cloud SDK
- [ ] **Authentication**: Setup GCP credentials

## 🔧 Step 1: Install Prerequisites

### Install Docker Desktop
```bash
# Windows: Download from https://docs.docker.com/desktop/install/windows/
# After installation, verify:
docker --version
```

### Install Google Cloud SDK
```bash
# Windows: Download from https://cloud.google.com/sdk/docs/install-sdk#windows
# After installation, verify:
gcloud version
```

### Authenticate with Google Cloud
```bash
# Login to your Google account
gcloud auth login

# Set your project
gcloud config set project proud-curve-473118-h6

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com  
gcloud services enable containerregistry.googleapis.com
```

## 🐳 Step 2: Build Docker Images

### Build API Image
```bash
cd C:\local\my_project_attempts\stock_prediction

# Build API container
docker build -t gcr.io/proud-curve-473118-h6/stock-prediction-api:latest -f Dockerfile .

# Test locally (optional)
docker run -p 8081:8000 gcr.io/proud-curve-473118-h6/stock-prediction-api:latest
```

### Build Dashboard Image  
```bash
# Build Dashboard container
docker build -t gcr.io/proud-curve-473118-h6/stock-prediction-dashboard:latest -f Dockerfile.dashboard .

# Test locally (optional)
docker run -p 8501:8501 gcr.io/proud-curve-473118-h6/stock-prediction-dashboard:latest
```

## ☁️ Step 3: Push to Google Container Registry

```bash
# Configure Docker for GCP
gcloud auth configure-docker

# Push API image
docker push gcr.io/proud-curve-473118-h6/stock-prediction-api:latest

# Push Dashboard image  
docker push gcr.io/proud-curve-473118-h6/stock-prediction-dashboard:latest
```

## 🚀 Step 4: Deploy to Cloud Run

### Deploy API Service
```bash
gcloud run deploy stock-prediction-api \
    --image gcr.io/proud-curve-473118-h6/stock-prediction-api:latest \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 1 \
    --timeout 300 \
    --max-instances 10 \
    --set-env-vars NEWS_API_KEY=2fc7702fb7eb4488919a2adcba42ea45
```

### Deploy Dashboard Service
```bash
# First, get the API URL
API_URL=$(gcloud run services describe stock-prediction-api --region us-central1 --format 'value(status.url)')

# Deploy Dashboard
gcloud run deploy stock-prediction-dashboard \
    --image gcr.io/proud-curve-473118-h6/stock-prediction-dashboard:latest \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --timeout 300 \
    --max-instances 5 \
    --set-env-vars API_BASE_URL=$API_URL
```

## 🧪 Step 5: Test Deployment

### Test API
```bash
# Get API URL
API_URL=$(gcloud run services describe stock-prediction-api --region us-central1 --format 'value(status.url)')

# Test health endpoint
curl "$API_URL/health"

# Test prediction
curl -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "days_ahead": 7}'
```

### Test Dashboard
```bash
# Get Dashboard URL
DASHBOARD_URL=$(gcloud run services describe stock-prediction-dashboard --region us-central1 --format 'value(status.url)')

echo "Dashboard available at: $DASHBOARD_URL"
```

## 📊 Step 6: Monitor & Manage

### View Logs
```bash
# API logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=stock-prediction-api"

# Dashboard logs  
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=stock-prediction-dashboard"
```

### Monitor Costs
```bash
# View current billing
gcloud billing accounts list
gcloud billing projects describe proud-curve-473118-h6

# Set billing alerts in GCP Console
```

### Update Services
```bash
# Rebuild and redeploy
docker build -t gcr.io/proud-curve-473118-h6/stock-prediction-api:v2 .
docker push gcr.io/proud-curve-473118-h6/stock-prediction-api:v2

gcloud run deploy stock-prediction-api \
    --image gcr.io/proud-curve-473118-h6/stock-prediction-api:v2 \
    --region us-central1
```

## 💰 Expected Costs

### Monthly Estimates (Light Usage)
- **API Server**: $5-15/month
- **Dashboard**: $3-10/month  
- **Container Registry**: $1-3/month
- **Load Balancer**: $18/month (if needed)
- **Total**: ~$30-50/month

### Cost Optimization Tips
1. **Enable auto-scaling to zero**: Saves costs when not in use
2. **Use Cloud Run** (not GKE): Pay per request model
3. **Monitor usage**: Set billing alerts
4. **Optimize images**: Smaller containers = faster cold starts

## 🔒 Security Considerations

### Environment Variables
```bash
# Store sensitive data securely
gcloud secrets create news-api-key --data-file=-
echo "2fc7702fb7eb4488919a2adcba42ea45" | gcloud secrets create news-api-key --data-file=-

# Use in deployment
gcloud run deploy stock-prediction-api \
    --set-env-vars NEWS_API_KEY=$(gcloud secrets versions access latest --secret=news-api-key)
```

### IAM & Permissions
```bash
# Least privilege principle
gcloud projects add-iam-policy-binding proud-curve-473118-h6 \
    --member="serviceAccount:service-account@proud-curve-473118-h6.iam.gserviceaccount.com" \
    --role="roles/run.invoker"
```

## 🎯 Success Metrics

### Performance Targets
- **API Response Time**: < 200ms
- **Dashboard Load Time**: < 3 seconds
- **Uptime**: > 99.5%
- **Accuracy**: > 95% (current: 96.5%)

### Monitoring Dashboard URLs
After deployment, you'll have:
- **API Health**: `https://your-api-url/health`
- **API Docs**: `https://your-api-url/docs`
- **Dashboard**: `https://your-dashboard-url`
- **GCP Console**: `https://console.cloud.google.com/run`

## 🆘 Troubleshooting

### Common Issues
1. **Build Fails**: Check Dockerfile syntax and dependencies
2. **Deploy Fails**: Verify GCP permissions and quotas
3. **503 Errors**: Check memory/CPU limits
4. **API Timeout**: Increase timeout or optimize code

### Quick Fixes
```bash
# View service details
gcloud run services describe stock-prediction-api --region us-central1

# Check quotas
gcloud compute project-info describe --project=proud-curve-473118-h6

# Reset service
gcloud run services delete stock-prediction-api --region us-central1
# Then redeploy
```

## 🎉 Final Steps

1. **Bookmark URLs**: Save your API and Dashboard URLs
2. **Setup Monitoring**: Configure uptime checks
3. **Document APIs**: Share the `/docs` endpoint
4. **Schedule Backups**: Regular model retraining
5. **Scale as Needed**: Monitor usage and adjust resources

---

**🚀 Ready to Deploy?**  
Run the automated deployment script: `python deploy_cloud.py`

**📞 Need Help?**  
- GCP Console: https://console.cloud.google.com
- Cloud Run Docs: https://cloud.google.com/run/docs
- Troubleshooting: Check service logs in GCP Console