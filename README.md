# Cocrystal Screener API

FastAPI backend for the Cocrystal Screener demo.
Runs RDKit-based coformer ranking, HEX fingerprinting, and patent reference generation.

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | /api/health | Health check |
| POST | /api/screen | Rank coformers against one API |
| POST | /api/fingerprint | Generate patent record for one API+coformer pair |
| POST | /api/decode | Decode a HEX fingerprint to descriptor ranges |
| POST | /api/upload-library | Screen a PubChem CSV against one API |

Full interactive docs available at `/docs` once deployed.

## Deploy to Railway

1. Push this repo to GitHub
2. Go to railway.app and create a new project
3. Select "Deploy from GitHub repo"
4. Select this repository
5. Railway detects the Dockerfile automatically and builds
6. Once deployed, copy the public URL (e.g. https://cocrystal-api.railway.app)
7. Use that URL as VITE_API_URL in the Lovable frontend

## Local development

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open http://localhost:8000/docs for the interactive API explorer.

## Project structure

```
main.py          FastAPI app and all endpoints
scoring.py       All science: RDKit descriptors, encoding, scoring
requirements.txt Python dependencies
Dockerfile       Container definition for Railway
railway.json     Railway deployment config
```

## Scoring logic

The likelihood score is a weighted sum of normalised descriptors:

| Feature | Weight | Direction |
|---|---|---|
| HB complementarity | +3.0 | Higher is better |
| Coformer HBD | +0.3 | Higher is better |
| Coformer HBA | +0.3 | Higher is better |
| delta LogP | -0.8 | Lower is better |
| delta TPSA | -0.6 | Lower is better |
| delta MW | -0.5 | Lower is better |

## HEX fingerprint format

4-character code where each digit encodes one descriptor (4 bits, 16 levels):

```
[HBC][dPKA][dLGP][dTPS]
```

- HBC: HB complementarity (0-1)
- dPKA: |delta pKa| (0-20)
- dLGP: |delta LogP| (0-10)
- dTPS: |delta TPSA| (0-200)
