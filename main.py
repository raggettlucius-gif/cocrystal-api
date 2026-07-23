"""
main.py
FastAPI backend for the Cocrystal Screener.
"""

import io
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from scoring import (
    screen_coformers,
    screen_library_df,
    build_patent_record,
    decode_pair_fingerprint,
    mol_props,
)


# ── App setup ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Cocrystal Screener API",
    description="Physicochemical coformer ranking, HEX fingerprinting, and patent reference generation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / response models ──────────────────────────────────────────────

class CoformerInput(BaseModel):
    name: str
    smiles: str
    pka: Optional[float] = None


class ScreenRequest(BaseModel):
    api_name: str
    api_smiles: str
    api_pka: Optional[float] = None
    coformers: list[CoformerInput]


class FingerprintRequest(BaseModel):
    api_name: str
    api_smiles: str
    api_pka: Optional[float] = None
    coformer_name: str
    coformer_smiles: str
    coformer_pka: Optional[float] = None


class DecodeRequest(BaseModel):
    hex_fingerprint: str


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/api/screen")
def screen(req: ScreenRequest):
    """
    Rank a list of coformers against one API by cocrystal likelihood.
    Returns results sorted by likelihood_score descending with HEX fingerprints.
    """
    try:
        mol_props(req.api_smiles)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid API SMILES: {req.api_smiles}")

    if not req.coformers:
        raise HTTPException(status_code=400, detail="At least one coformer is required.")

    coformer_dicts = [
        {"name": c.name, "smiles": c.smiles, "pka": c.pka}
        for c in req.coformers
    ]

    results = screen_coformers(
        api_smiles=req.api_smiles,
        coformers=coformer_dicts,
        api_pka=req.api_pka,
    )

    return {
        "api_name":   req.api_name,
        "api_smiles": req.api_smiles,
        "api_pka":    req.api_pka,
        "count":      len(results),
        "results":    results,
    }


@app.post("/api/fingerprint")
def fingerprint(req: FingerprintRequest):
    """
    Generate a full patent reference record for a single API+coformer pair.
    """
    try:
        record = build_patent_record(
            api_name=req.api_name,
            api_smiles=req.api_smiles,
            api_pka=req.api_pka,
            cf_name=req.coformer_name,
            cf_smiles=req.coformer_smiles,
            cf_pka=req.coformer_pka,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return record


@app.post("/api/decode")
def decode(req: DecodeRequest):
    """
    Decode a 4-character HEX fingerprint back to approximate descriptor ranges.
    """
    hex_str = req.hex_fingerprint.strip().upper()
    if len(hex_str) != 4:
        raise HTTPException(
            status_code=400,
            detail="HEX fingerprint must be exactly 4 characters (e.g. A3F2)."
        )
    try:
        return decode_pair_fingerprint(hex_str)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/upload-library")
async def upload_library(
    file: UploadFile = File(...),
    api_smiles: str = Query(..., description="SMILES string of the API to screen against"),
    api_name: str = Query("API", description="Name of the API"),
    api_pka: Optional[float] = Query(None, description="pKa of the API (optional)"),
):
    """
    Upload a PubChem-format CSV and screen all coformers against one API.
    Expected columns: Coformer Name, CID, SMILES, MW, LogP, HBD, HBA, TPSA, source_seed
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    required_cols = {"SMILES", "MW", "LogP", "HBD", "HBA", "TPSA"}
    missing = required_cols - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"CSV is missing required columns: {missing}"
        )

    # Normalise the name column (trailing space variant from PubChem export)
    if "Coformer Name " in df.columns:
        df.rename(columns={"Coformer Name ": "Coformer Name"}, inplace=True)

    try:
        mol_props(api_smiles)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid API SMILES: {api_smiles}")

    try:
        results = screen_library_df(
            api_smiles=api_smiles,
            df=df,
            api_pka=api_pka,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Screening failed: {e}")

    return {
        "api_name":   api_name,
        "api_smiles": api_smiles,
        "api_pka":    api_pka,
        "count":      len(results),
        "results":    results,
    }
