#!/bin/bash

echo "🚀 Setting up environment variables..."

# Set environment variables (replace with your actual values)
export MONGO_URI="your-mongodb-atlas-uri"
export DATABASE_NAME="news_service"
export FIREBASE_SERVICE_ACCOUNT_BASE64="your-base64-encoded-firebase-key"
export FIREBASE_STORAGE_BUCKET="gs://varthajanapadanewsapp.firebasestorage.app"
export ELEVENLABS_API_KEY="your-elevenlabs-api-key"
export API_KEY="your-fastapi-auth-key"
export CORS_ORIGINS="https://diprwebapp.gully2global.in,https://dipradmin.gully2global.in"
export AI4BHARAT_MODELS_PATH="/app/models"
export LOG_LEVEL="INFO"

echo "📋 Environment variables set!"
echo ""
echo "⚠️  IMPORTANT: Please update the values above with your actual API keys and credentials"
echo ""
echo "🔧 Next steps:"
echo "1. Edit this file and replace the placeholder values with your actual credentials"
echo "2. Run: chmod +x deploy.sh"
echo "3. Run: ./deploy.sh"
echo ""
echo "📝 Required values to update:"
echo "- MONGO_URI: Your MongoDB Atlas connection string"
echo "- FIREBASE_SERVICE_ACCOUNT_BASE64: Your base64-encoded Firebase service account"
echo "- ELEVENLABS_API_KEY: Your ElevenLabs API key"
echo "- API_KEY: Any secret key for your FastAPI service" 