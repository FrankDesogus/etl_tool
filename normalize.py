import pandas as pd
from typing import Any, Dict, List, Optional
from utils import now_iso

INVISIBLE = {"\u00A0": " ", "\u200B": "", "\u200C": "", "\u200D": "", "\ufeff": ""}

def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    for k, v in INVISIBLE.items():
        x = x.replace(k, v)
    x = x.strip()
    while "  " in x:
        x = x.replace("  ", " ")
    return x

def normalize_case_policy(x: Any) -> str:
    return normalize_text(x).upper()

def normalize_amount(x: Any) -> float:
    txt = normalize_text(x)
    if txt in {"", "-", "€", "-€", "- €"}:
        return 0.0
    txt = txt.replace("€", "").strip()
    txt = txt.replace(".", "").replace(",", ".")
    try:
        return float(txt)
    except Exception:
        return 0.0

def parse_date_to_iso(x: Any) -> str:
    if x is None:
        return ""
    if hasattr(x, "date"):  # datetime/date
        try:
            return x.date().isoformat()
        except Exception:
            pass
    txt = normalize_text(x)
    if not txt:
        return ""
    dt = pd.to_datetime(txt, dayfirst=True, errors="coerce")
    if pd.isna(dt):
        return ""
    return dt.date().isoformat()

def add_norm_log(norm_logs: List[Dict[str, Any]], *, entity: str, record_id: str,
                 field_name: str, original_value: Any, normalized_value: Any,
                 reason: str, source_file: str, source_sheet: str, source_row: int,
                 source_column: Optional[str] = None) -> None:
    norm_logs.append({
        "timestamp": now_iso(),
        "entity": entity,
        "record_id": record_id,
        "field_name": field_name,
        "original_value": "" if original_value is None else str(original_value),
        "normalized_value": "" if normalized_value is None else str(normalized_value),
        "reason_rule_applied": reason,
        "source_file": source_file,
        "source_sheet": source_sheet,
        "source_row": source_row,
        "source_column": source_column or "",
    })