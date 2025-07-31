#!/bin/bash

echo "🚀 FastAPI News Service - Google Cloud Run Deployment"
echo "=================================================="

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Google Cloud CLI is not installed!"
    echo "📥 Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed!"
    echo "📥 Install from: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if user is logged in to gcloud
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Not logged in to Google Cloud!"
    echo "🔐 Run: gcloud auth login"
    exit 1
fi

echo "✅ Prerequisites check passed!"

# Configuration
PROJECT_ID="our-highway-467609-h8"
SERVICE_NAME="fastapi-news-service"
REGION="us-central1"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo ""
echo "📋 Configuration:"
echo "Project ID: $PROJECT_ID"
echo "Service Name: $SERVICE_NAME"
echo "Region: $REGION"
echo ""

# Check if environment variables are set
echo "🔍 Checking environment variables..."

# Required environment variables
REQUIRED_VARS=(
    "MONGO_URI"
    "FIREBASE_SERVICE_ACCOUNT_BASE64"
    "ELEVENLABS_API_KEY"
    "API_KEY"
)

MISSING_VARS=()

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        MISSING_VARS+=("$var")
    fi
done

if [ ${#MISSING_VARS[@]} -ne 0 ]; then
    echo "❌ Missing required environment variables:"
    for var in "${MISSING_VARS[@]}"; do
        echo "   - $var"
    done
    echo ""
    echo "📝 Please set these variables:"
    echo "export MONGO_URI=\"your-mongodb-atlas-uri\""
    echo "export FIREBASE_SERVICE_ACCOUNT_BASE64=\"your-base64-encoded-firebase-key\""
    echo "export ELEVENLABS_API_KEY=\"your-elevenlabs-api-key\""
    echo "export API_KEY=\"your-fastapi-auth-key\""
    echo ""
    echo "💡 Or create a .env file and source it:"
    echo "source .env"
    exit 1
fi

echo "✅ All required environment variables are set!"

# Set default values for optional variables
export DATABASE_NAME=${DATABASE_NAME:-"news_service"}
export FIREBASE_STORAGE_BUCKET=${FIREBASE_STORAGE_BUCKET:-"gs://varthajanapadanewsapp.firebasestorage.app"}
export CORS_ORIGINS=${CORS_ORIGINS:-"https://diprwebapp.gully2global.in,https://dipradmin.gully2global.in"}
export AI4BHARAT_MODELS_PATH=${AI4BHARAT_MODELS_PATH:-"/app/models"}
export LOG_LEVEL=${LOG_LEVEL:-"INFO"}

echo ""
echo "🚀 Starting deployment..."

# 1. Set project
echo "📦 Setting GCP project..."
gcloud config set project $PROJECT_ID

# 2. Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# 3. Configure Docker for GCR
echo "🐳 Configuring Docker for Google Container Registry..."
gcloud auth configure-docker

# 4. Build and push Docker image
echo "🏗️ Building and pushing Docker image..."
echo "This may take 5-10 minutes for the first build..."
docker build -t $IMAGE_NAME .
docker push $IMAGE_NAME

# 5. Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
echo "This may take 5-10 minutes for the first deployment (model loading)..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 8Gi \
  --cpu 2 \
  --timeout 3600 \
  --concurrency 80 \
  --max-instances 10 \
  --set-env-vars="MONGO_URI=$MONGO_URI" \
  --set-env-vars="DATABASE_NAME=$DATABASE_NAME" \
  --set-env-vars="FIREBASE_SERVICE_ACCOUNT_BASE64=$FIREBASE_SERVICE_ACCOUNT_BASE64" \
  --set-env-vars="FIREBASE_STORAGE_BUCKET=$FIREBASE_STORAGE_BUCKET" \
  --set-env-vars="ELEVENLABS_API_KEY=$ELEVENLABS_API_KEY" \
  --set-env-vars="API_KEY=$API_KEY" \
  --set-env-vars="CORS_ORIGINS=$CORS_ORIGINS" \
  --set-env-vars="AI4BHARAT_MODELS_PATH=$AI4BHARAT_MODELS_PATH" \
  --set-env-vars="CLOUD_PROVIDER=gcp" \
  --set-env-vars="REGION=$REGION" \
  --set-env-vars="CLUSTER_NAME=cloud-run" \
  --set-env-vars="SERVICE_NAME=$SERVICE_NAME" \
  --set-env-vars="LOG_LEVEL=$LOG_LEVEL"

# 6. Get service URL
echo "🌐 Getting service URL..."
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")

echo ""
echo "🎉 Deployment complete!"
echo "=================================================="
echo "🌐 Your service is available at: $SERVICE_URL"
echo "📚 API Documentation: $SERVICE_URL/docs"
echo "❤️ Health Check: $SERVICE_URL/health"
echo "🔧 Service Management: https://console.cloud.google.com/run/detail/$REGION/$SERVICE_NAME"
echo ""
echo "📋 Useful commands:"
echo "View logs: gcloud logs tail --service=$SERVICE_NAME --region=$REGION"
echo "Update service: gcloud run services update $SERVICE_NAME --region=$REGION --image=$IMAGE_NAME"
echo "Delete service: gcloud run services delete $SERVICE_NAME --region=$REGION"
echo ""
echo "⚠️  Note: First request may take 5-10 minutes (model loading)"
echo "✅ Subsequent requests will be fast!" 