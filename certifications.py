import json
import hashlib
import pandas as pd
from typing import Any, Dict, List, Tuple
from normalize import normalize_text, normalize_case_policy, parse_date_to_iso
from utils import now_iso

INVALID_CERT_TOKENS = {"", "NAN", "NONE", "NULL", "-", "N/A"}


def _make_certification_id(supplier_id: str, cert_name_raw: str, idx: int) -> str:
    payload = f"{supplier_id}|{cert_name_raw}|{idx}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12].upper()
    return f"CERT_{digest}"


def split_certifications(df_sup: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]], int]:
    cert_rows: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    cert_tokens_dropped_nan_like = 0

    for _, s in df_sup.iterrows():
        supplier_id = str(s.get("supplier_id", "")).strip()
        if not supplier_id:
            warnings.append({
                "timestamp": now_iso(),
                "warning_type": "missing_supplier_id_for_certifications",
                "supplier_id": "",
                "supplier_name_normalized": s.get("supplier_name_normalized", ""),
                "certifications_raw": normalize_text(s.get("certifications_raw")),
                "expiry_date_raw": normalize_text(s.get("cert_expiry_raw")),
                "source_file": s.get("source_file", ""),
                "source_sheet": "",
                "source_row": "",
                "source_column": s.get("cert_expiry_source_column", ""),
            })
            continue

        certs_raw = normalize_text(s.get("certifications_raw"))
        expiry_raw = normalize_text(s.get("cert_expiry_raw"))
        expiry_iso = parse_date_to_iso(expiry_raw)
        prov = []
        try:
            prov = json.loads(s.get("provenance", "[]"))
        except Exception:
            prov = []
        src_sheet = prov[0].get("sheet", "") if prov else ""
        src_row = prov[0].get("row", "") if prov else ""

        if not certs_raw:
            continue

        raw_parts = []
        for chunk in certs_raw.replace("\n", ";").split(";"):
            chunk = chunk.strip()
            if chunk:
                raw_parts.extend([c.strip() for c in chunk.split(",")])

        parts = []
        for p in raw_parts:
            p_norm = normalize_case_policy(p)
            if p_norm in INVALID_CERT_TOKENS:
                cert_tokens_dropped_nan_like += 1
                continue
            parts.append(p)

        if not parts:
            continue

        if expiry_raw and not expiry_iso:
            warnings.append({
                "timestamp": now_iso(),
                "warning_type": "cert_expiry_unparseable",
                "supplier_id": s.get("supplier_id", ""),
                "supplier_name_normalized": s.get("supplier_name_normalized", ""),
                "certifications_raw": certs_raw,
                "expiry_date_raw": expiry_raw,
                "source_file": s.get("source_file", ""),
                "source_sheet": src_sheet,
                "source_row": src_row,
                "source_column": s.get("cert_expiry_source_column", ""),
            })
        if not expiry_raw:
            warnings.append({
                "timestamp": now_iso(),
                "warning_type": "cert_expiry_missing",
                "supplier_id": s.get("supplier_id", ""),
                "supplier_name_normalized": s.get("supplier_name_normalized", ""),
                "certifications_raw": certs_raw,
                "expiry_date_raw": "",
                "source_file": s.get("source_file", ""),
                "source_sheet": src_sheet,
                "source_row": src_row,
                "source_column": s.get("cert_expiry_source_column", ""),
            })

        for idx, p in enumerate(parts, start=1):
            cert_rows.append({
                "certification_id": _make_certification_id(supplier_id, p, idx),
                "supplier_id": supplier_id,
                "supplier_name_normalized": s["supplier_name_normalized"],
                "cert_name_raw": p,
                "cert_name_normalized": normalize_case_policy(p),
                "expiry_date_raw": expiry_raw,
                "expiry_date": expiry_iso,
                "source_file": s.get("source_file", ""),
                "source_sheet": src_sheet,
                "source_row": src_row,
                "source_column": s.get("cert_expiry_source_column", "CERTIFICAZIONE/CERTIFICAZIONI"),
            })

    return pd.DataFrame(cert_rows), warnings, cert_tokens_dropped_nan_like
