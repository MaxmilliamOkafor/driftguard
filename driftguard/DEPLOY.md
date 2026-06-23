# Deploying DriftGuard

## Streamlit Community Cloud (dashboard)
1. Push to GitHub.
2. On https://share.streamlit.io: repo `MaxmilliamOkafor/driftguard`,
   main file `dashboard/streamlit_app.py`.
3. No secrets needed — the demo is fully synthetic and deterministic.

## FastAPI serving (optional, Render/Railway)
Start command: `uvicorn driftguard.serve:app --host 0.0.0.0 --port $PORT`
(Run `python -m driftguard.monitor` once first to populate the model registry.)

## Pretty URL — maxmilliamlabs-ai.web.app/driftguard
Add a redirect in your Firebase `firebase.json`:
```json
{ "source": "/driftguard", "destination": "https://<your-app>.streamlit.app", "type": 302 }
```
