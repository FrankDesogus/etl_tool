import pandas as pd
from typing import Any, Dict, List, Tuple
from normalize import normalize_text, normalize_case_policy

def split_certifications(df_sup: pd.DataFrame) -> pd.DataFrame:
    cert_rows = []
    for _, s in df_sup.iterrows():
        certs_raw = normalize_text(s.get("certifications_raw"))
        if not certs_raw:
            continue

        parts = []
        for chunk in certs_raw.replace("\n", ";").split(";"):
            chunk = chunk.strip()
            if chunk:
                parts.extend([c.strip() for c in chunk.split(",") if c.strip()])

        for p in parts:
            cert_rows.append({
                "certification_id": f"CERT_{abs(hash((s['supplier_id'], p)))}",
                "supplier_id": s["supplier_id"],
                "supplier_name_normalized": s["supplier_name_normalized"],
                "cert_name_raw": p,
                "cert_name_normalized": normalize_case_policy(p),
                "expiry_date_raw": "",
                "expiry_date": "",
                "source_file": s.get("source_file", ""),
                "source_sheet": "",
                "source_row": "",
                "source_column": "CERTIFICAZIONE/CERTIFICAZIONI",
            })
    return pd.DataFrame(cert_rows)