# News TTS Service

FastAPI service for news translation (IndicTrans2) and text-to-speech (ElevenLabs)

## Quick start
```bash
python -m venv .venv
source venv/Scripts/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## Docker
```bash
docker build -t news-tts .
docker run --rm -p 8080:8080 news-tts
```

## Endpoints
- GET `/` – service info
- GET `/health` – liveness
- GET `/ready` – readiness
- GET `/docs` – Swagger UI
- POST `/api/translate`
- POST `/api/create`
- POST `/api/tts/generate`