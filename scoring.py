"""
scoring.py
All cocrystal scoring, encoding, and fingerprint logic.
Isolated from FastAPI so it can be tested independently.
"""

import math
import hashlib
from datetime import date
from typing import Optional

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


# ── Encoding schemas ───────────────────────────────────────────────────────

PAIR_SCHEMA = {
    # field: (min_val, max_val, n_bits, description, hex_label)
    "hb_complementarity": (0.0,   1.0,  4, "HB donor/acceptor complementarity", "HBC"),
    "delta_pka":          (0.0,  20.0,  4, "|delta pKa| between components",     "dPKA"),
    "delta_logp":         (0.0,  10.0,  4, "|delta LogP| between components",    "dLGP"),
    "delta_tpsa":         (0.0, 200.0,  4, "|delta TPSA| between components",    "dTPS"),
}

LIBRARY_SCHEMA = {
    "hbd":  (0.0,    6.0,  4, "H-bond donor count",    "HBD"),
    "hba":  (0.0,   14.0,  4, "H-bond acceptor count", "HBA"),
    "logp": (-16.0,  6.0,  4, "LogP (lipophilicity)",  "LGP"),
    "tpsa": (0.0,  300.0,  4, "TPSA in Angstrom^2",    "TPS"),
}

SCORING_WEIGHTS = {
    "hb_complementarity": 3.0,
    "cf_hbd":             0.3,
    "cf_hba":             0.3,
    "delta_logp":        -0.8,
    "delta_tpsa":        -0.6,
    "delta_mw":          -0.5,
}

SCORING_NORM = {
    "hb_complementarity": 1.0,
    "cf_hbd":             5.0,
    "cf_hba":            10.0,
    "delta_logp":        10.0,
    "delta_tpsa":       200.0,
    "delta_mw":         400.0,
}


# ── RDKit helpers ──────────────────────────────────────────────────────────

def mol_props(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return {
        "hbd":  rdMolDescriptors.CalcNumHBD(mol),
        "hba":  rdMolDescriptors.CalcNumHBA(mol),
        "logp": round(Descriptors.MolLogP(mol), 4),
        "tpsa": round(Descriptors.TPSA(mol), 4),
        "mw":   round(Descriptors.MolWt(mol), 4),
    }


def molecular_formula(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    counts = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        counts[sym] = counts.get(sym, 0) + 1
    result = ""
    for sym in ["C", "H"]:
        if sym in counts:
            n = counts.pop(sym)
            result += sym + (str(n) if n > 1 else "")
    for sym in sorted(counts):
        n = counts[sym]
        result += sym + (str(n) if n > 1 else "")
    return result


def cocrystal_formula(api_smiles: str, cf_smiles: str) -> str:
    mol1 = Chem.MolFromSmiles(api_smiles)
    mol2 = Chem.MolFromSmiles(cf_smiles)
    if mol1 is None or mol2 is None:
        raise ValueError("Invalid SMILES in cocrystal formula computation")
    combo = Chem.CombineMols(mol1, mol2)
    combo = Chem.AddHs(combo)
    counts = {}
    for atom in combo.GetAtoms():
        sym = atom.GetSymbol()
        counts[sym] = counts.get(sym, 0) + 1
    result = ""
    for sym in ["C", "H"]:
        if sym in counts:
            n = counts.pop(sym)
            result += sym + (str(n) if n > 1 else "")
    for sym in sorted(counts):
        n = counts[sym]
        result += sym + (str(n) if n > 1 else "")
    return result


# ── Encoding helpers ───────────────────────────────────────────────────────

def encode_field(value: float, min_val: float, max_val: float, n_bits: int) -> str:
    levels  = (2 ** n_bits) - 1
    clamped = max(min_val, min(max_val, float(value)))
    idx     = round((clamped - min_val) / (max_val - min_val) * levels)
    return format(int(idx), f"0{n_bits}b")


def decode_field(bits: str, min_val: float, max_val: float) -> float:
    levels = (2 ** len(bits)) - 1
    return round(min_val + (int(bits, 2) / levels) * (max_val - min_val), 4)


def bits_to_hex(bitstring: str) -> str:
    return format(int(bitstring, 2), f"0{len(bitstring) // 4}X")


# ── HB complementarity ────────────────────────────────────────────────────

def hb_complementarity(api_hbd: int, api_hba: int,
                        cf_hbd: int, cf_hba: int) -> float:
    def ratio(a, b):
        return min(a, b) / max(a, b, 1)
    return round(0.5 * ratio(api_hbd, cf_hba) + 0.5 * ratio(cf_hbd, api_hba), 4)


# ── Likelihood score ──────────────────────────────────────────────────────

def likelihood_score(hbc: float, d_logp: float, d_tpsa: float,
                     d_mw: float, cf_hbd: int, cf_hba: int) -> float:
    feats = {
        "hb_complementarity": hbc,
        "cf_hbd":   cf_hbd,
        "cf_hba":   cf_hba,
        "delta_logp": d_logp,
        "delta_tpsa": d_tpsa,
        "delta_mw":   d_mw,
    }
    score = 0.0
    for f, v in feats.items():
        normed = min(abs(v), SCORING_NORM[f]) / SCORING_NORM[f]
        score += SCORING_WEIGHTS[f] * normed
    return round(score, 4)


# ── Pair fingerprint ──────────────────────────────────────────────────────

def pair_fingerprint(hbc: float, d_pka: float,
                     d_logp: float, d_tpsa: float) -> tuple[str, str]:
    bits  = encode_field(hbc,    *PAIR_SCHEMA["hb_complementarity"][:2], 4)
    bits += encode_field(d_pka,  *PAIR_SCHEMA["delta_pka"][:2],          4)
    bits += encode_field(d_logp, *PAIR_SCHEMA["delta_logp"][:2],         4)
    bits += encode_field(d_tpsa, *PAIR_SCHEMA["delta_tpsa"][:2],         4)
    return bits, bits_to_hex(bits)


def decode_pair_fingerprint(hex_str: str) -> dict:
    hex_str  = hex_str.strip().upper()
    bits_str = format(int(hex_str, 16), "016b")
    result   = {"hex_fingerprint": hex_str, "bitstring": bits_str, "fields": {}}
    pos = 0
    for field, (mn, mx, nb, desc, label) in PAIR_SCHEMA.items():
        chunk = bits_str[pos:pos+nb]
        val   = decode_field(chunk, mn, mx)
        step  = (mx - mn) / ((2 ** nb) - 1)
        result["fields"][field] = {
            "label":        label,
            "hex_digit":    hex_str[pos // 4],
            "bits":         chunk,
            "level":        int(chunk, 2),
            "approx_value": val,
            "range":        f"{round(val - step/2, 2)} - {round(val + step/2, 2)}",
            "description":  desc,
        }
        pos += nb
    return result


# ── Library fingerprint ───────────────────────────────────────────────────

def library_fingerprint(hbd: float, hba: float,
                         logp: float, tpsa: float) -> tuple[str, str]:
    bits  = encode_field(hbd,  *LIBRARY_SCHEMA["hbd"][:2],  4)
    bits += encode_field(hba,  *LIBRARY_SCHEMA["hba"][:2],  4)
    bits += encode_field(logp, *LIBRARY_SCHEMA["logp"][:2], 4)
    bits += encode_field(tpsa, *LIBRARY_SCHEMA["tpsa"][:2], 4)
    return bits, bits_to_hex(bits)


# ── Patent reference record ───────────────────────────────────────────────

def patent_reference_id(hex_fp: str, formula: str) -> str:
    import re
    atom_marker = 0
    temp = formula
    for sym in ["C", "H"]:
        temp = re.sub(rf"{sym}\d*", "", temp)
    for n in re.findall(r"\d+", temp):
        atom_marker += int(n)
    non_ch_atoms = len(re.findall(r"[A-Z]", temp))
    atom_marker += non_ch_atoms
    atom_hex  = format(min(atom_marker, 255), "02X")
    checksum  = hashlib.sha256((hex_fp + formula).encode()).hexdigest()[:2].upper()
    return f"CCX-{hex_fp}-{atom_hex}-{checksum}"


def build_patent_record(
    api_name: str, api_smiles: str, api_pka: Optional[float],
    cf_name: str, cf_smiles: str, cf_pka: Optional[float],
) -> dict:
    api = mol_props(api_smiles)
    cf  = mol_props(cf_smiles)

    hbc    = hb_complementarity(api["hbd"], api["hba"], cf["hbd"], cf["hba"])
    d_pka  = abs(api_pka - cf_pka) if (api_pka is not None and cf_pka is not None) else 0.0
    d_logp = abs(api["logp"] - cf["logp"])
    d_tpsa = abs(api["tpsa"] - cf["tpsa"])
    d_mw   = abs(api["mw"]   - cf["mw"])

    bits, hex_fp = pair_fingerprint(hbc, d_pka, d_logp, d_tpsa)
    score        = likelihood_score(hbc, d_logp, d_tpsa, d_mw, cf["hbd"], cf["hba"])
    formula      = cocrystal_formula(api_smiles, cf_smiles)
    ref_id       = patent_reference_id(hex_fp, formula)

    groups = [bits[i:i+4] for i in range(0, 16, 4)]
    labels = list(PAIR_SCHEMA.keys())

    return {
        "reference_id":       ref_id,
        "hex_fingerprint":    hex_fp,
        "bitstring":          bits,
        "cocrystal_formula":  formula,
        "api_formula":        molecular_formula(api_smiles),
        "coformer_formula":   molecular_formula(cf_smiles),
        "api_name":           api_name,
        "coformer_name":      cf_name,
        "likelihood_score":   score,
        "generation_date":    str(date.today()),
        "field_breakdown": {
            labels[i]: {
                "label":       PAIR_SCHEMA[labels[i]][4],
                "hex_digit":   hex_fp[i],
                "bits":        groups[i],
                "level":       int(groups[i], 2),
                "description": PAIR_SCHEMA[labels[i]][3],
            }
            for i in range(4)
        },
        "descriptors": {
            "api":      {**api, "pka": api_pka},
            "coformer": {**cf,  "pka": cf_pka},
            "deltas": {
                "hb_complementarity": hbc,
                "delta_pka":   round(d_pka,  4),
                "delta_logp":  round(d_logp, 4),
                "delta_tpsa":  round(d_tpsa, 4),
                "delta_mw":    round(d_mw,   4),
            },
        },
    }


# ── Screen a list of coformers against one API ────────────────────────────

def screen_coformers(
    api_smiles: str,
    coformers: list[dict],
    api_pka: Optional[float] = None,
) -> list[dict]:
    """
    coformers: list of dicts with keys: name, smiles, pka (optional)
    Returns list of result dicts sorted by likelihood_score descending.
    """
    api = mol_props(api_smiles)
    results = []

    for cf_input in coformers:
        try:
            cf = mol_props(cf_input["smiles"])
        except ValueError as e:
            results.append({
                "name":   cf_input.get("name", "Unknown"),
                "smiles": cf_input.get("smiles", ""),
                "error":  str(e),
            })
            continue

        cf_pka = cf_input.get("pka")
        hbc    = hb_complementarity(api["hbd"], api["hba"], cf["hbd"], cf["hba"])
        d_pka  = abs(api_pka - cf_pka) if (api_pka is not None and cf_pka is not None) else 0.0
        d_logp = abs(api["logp"] - cf["logp"])
        d_tpsa = abs(api["tpsa"] - cf["tpsa"])
        d_mw   = abs(api["mw"]   - cf["mw"])

        score        = likelihood_score(hbc, d_logp, d_tpsa, d_mw, cf["hbd"], cf["hba"])
        bits, hex_fp = pair_fingerprint(hbc, d_pka, d_logp, d_tpsa)
        groups       = [bits[i:i+4] for i in range(0, 16, 4)]
        schema_keys  = list(PAIR_SCHEMA.keys())

        results.append({
            "name":               cf_input.get("name", "Unknown"),
            "smiles":             cf_input["smiles"],
            "pka":                cf_pka,
            "mw":                 cf["mw"],
            "logp":               cf["logp"],
            "hbd":                cf["hbd"],
            "hba":                cf["hba"],
            "tpsa":               cf["tpsa"],
            "likelihood_score":   score,
            "hb_complementarity": hbc,
            "delta_pka":          round(d_pka,  4) if d_pka else None,
            "delta_logp":         round(d_logp, 4),
            "delta_tpsa":         round(d_tpsa, 4),
            "delta_mw":           round(d_mw,   4),
            "hex_fingerprint":    hex_fp,
            "bitstring":          bits,
            "hex_digit_hbc":      hex_fp[0],
            "hex_digit_dpka":     hex_fp[1],
            "hex_digit_dlogp":    hex_fp[2],
            "hex_digit_dtpsa":    hex_fp[3],
            "bit_groups": {
                schema_keys[i]: {
                    "label": PAIR_SCHEMA[schema_keys[i]][4],
                    "bits":  groups[i],
                    "level": int(groups[i], 2),
                }
                for i in range(4)
            },
        })

    results.sort(key=lambda x: x.get("likelihood_score", -99), reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results


# ── Screen a PubChem-style library DataFrame ──────────────────────────────

def screen_library_df(api_smiles: str, df, api_pka: Optional[float] = None) -> list[dict]:
    """
    df: pandas DataFrame with columns: Coformer Name, CID, SMILES, MW, LogP, HBD, HBA, TPSA, source_seed
    Uses pre-computed PubChem properties directly — no RDKit recomputation on coformers.
    """
    api = mol_props(api_smiles)
    results = []

    for _, row in df.iterrows():
        name = str(row.get("Coformer Name", row.get("name", ""))).strip()
        if not name or name in ("nan", "NaN"):
            name = f"CID_{int(row['CID'])}"

        cf_hbd  = float(row["HBD"])
        cf_hba  = float(row["HBA"])
        cf_logp = float(row["LogP"])
        cf_tpsa = float(row["TPSA"])
        cf_mw   = float(row["MW"])

        hbc    = hb_complementarity(api["hbd"], api["hba"], int(cf_hbd), int(cf_hba))
        d_logp = abs(api["logp"] - cf_logp)
        d_tpsa = abs(api["tpsa"] - cf_tpsa)
        d_mw   = abs(api["mw"]   - cf_mw)

        score            = likelihood_score(hbc, d_logp, d_tpsa, d_mw, int(cf_hbd), int(cf_hba))
        lib_bits, lib_hex = library_fingerprint(cf_hbd, cf_hba, cf_logp, cf_tpsa)
        pair_bits, pair_hex = pair_fingerprint(hbc, 0.0, d_logp, d_tpsa)

        results.append({
            "name":               name,
            "cid":                int(row["CID"]),
            "smiles":             str(row["SMILES"]),
            "source_seed":        str(row.get("source_seed", "")),
            "mw":                 cf_mw,
            "logp":               cf_logp,
            "hbd":                int(cf_hbd),
            "hba":                int(cf_hba),
            "tpsa":               cf_tpsa,
            "likelihood_score":   score,
            "hb_complementarity": hbc,
            "delta_logp":         round(d_logp, 4),
            "delta_tpsa":         round(d_tpsa, 4),
            "delta_mw":           round(d_mw,   4),
            "lib_hex":            lib_hex,
            "lib_bitstring":      lib_bits,
            "pair_hex":           pair_hex,
            "pair_bitstring":     pair_bits,
        })

    results.sort(key=lambda x: x["likelihood_score"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results
