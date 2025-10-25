# Eunoia ML Service

Short description: Flask-based REST API for multi-label emotion detection (28 GoEmotions categories), tag generation, and batch mood analytics used by the Eunoia app.

## Tech Stack
- Python 3.x, Flask
- Hugging Face Inference API (SamLowe/roberta-base-go_emotions)
- Requests, Flask-CORS

## Endpoints
- `GET /health` – service status
- `POST /analyze` – analyze single entry `{ title, content }`
- `POST /batch-analyze` – analyze multiple entries `[{ title, content }]`

## Quick Start
1. Prerequisites: Python 3.10+
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Environment variables:
   - HF_API_TOKEN=your_hf_token
   - PORT=8000
4. Run locally:
   ```bash
   python application.py
   ```
```json
{
  "primary_emotion": "joy",
  "confidence": 0.87,
  "top_emotions": [{"emotion":"joy","score":0.87}, ...],
  "tags": ["joy","gratitude","friends"],
  "emotional_state_summary": "Happy and positive"
}
```

## Deployment
- Containerized via `Dockerfile`. Deploy on Hugging Face Spaces or any Python-friendly host.
