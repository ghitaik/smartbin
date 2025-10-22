# SmartBin ♻️
YOLOv8-based waste classifier aligned with German separation rules.

## Structure
- `smartbin-api/` – FastAPI backend (predict + suggest)
- `smartbin-web/` – Frontend (upload and view results)

## Local Quickstart
```bash
# API
cd smartbin-api
pip install -r requirements.txt
SMARTBIN_MODEL=/absolute/path/to/best.pt uvicorn app:app --reload

# Web
cd smartbin-web
# start your local web as you currently do


