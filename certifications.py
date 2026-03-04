import json
import pandas as pd
from typing import Any, Dict, List, Tuple
from normalize import normalize_text, normalize_case_policy, parse_date_to_iso
from utils import now_iso


def split_certifications(df_sup: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    cert_rows: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    for _, s in df_sup.iterrows():
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

        parts = []
        for chunk in certs_raw.replace("\n", ";").split(";"):
            chunk = chunk.strip()
            if chunk:
                parts.extend([c.strip() for c in chunk.split(",") if c.strip()])

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

        for p in parts:
            cert_rows.append({
                "certification_id": f"CERT_{abs(hash((s['supplier_id'], p)))}",
                "supplier_id": s["supplier_id"],
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

    return pd.DataFrame(cert_rows), warnings
