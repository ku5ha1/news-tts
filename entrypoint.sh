set -e

echo "🚀 Starting FastAPI server without preload (lazy model loading)..."
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port ${PORT:-8080} \
    --workers 1 \
    --timeout-keep-alive 300 \
    --no-access-log
