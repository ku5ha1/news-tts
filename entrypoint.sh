#!/bin/bash
echo "🔥 Warming up AI4Bharat translation models..."
python -c "from app.services.translation_service import TranslationService; TranslationService(); print('✅ Models loaded successfully!')"

# Start the FastAPI server
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port ${PORT:-8080} \
    --workers 1 \
    --timeout-keep-alive 300 \
    --no-access-log
