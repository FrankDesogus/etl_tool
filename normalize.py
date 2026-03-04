import re
from typing import Any, Dict, List, Optional

import pandas as pd

from utils import now_iso

INVISIBLE = {"\u00A0": " ", "\u200B": "", "\u200C": "", "\u200D": "", "\ufeff": ""}

BUSINESS_MULTI_PATTERNS = [
    (re.compile(r"\bS\s*\.?\s*R\s*\.?\s*L\s*\.?\s*S\b", re.IGNORECASE), " SRLS "),
    (re.compile(r"\bS\s*\.?\s*R\s*\.?\s*L\b", re.IGNORECASE), " SRL "),
    (re.compile(r"\bS\s*\.?\s*P\s*\.?\s*A\b", re.IGNORECASE), " SPA "),
]
BUSINESS_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

LEGAL_NOISE_TOKENS = {
    "SRL", "SRLS", "SPA", "SNC", "SAS", "SCPA", "SOCIETA", "SOCIETÀ", "SOC", "UNIPERSONALE", "COOP", "CONSORZIO",
}
BUSINESS_STOPWORDS = LEGAL_NOISE_TOKENS | {
    "DI", "DE", "DEL", "DELL", "DELLA", "DELLE", "DEGLI", "E", "AND", "THE",
}


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    if pd.isna(x):
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


def normalize_business_name(x: Any) -> str:
    txt = normalize_case_policy(x)
    if not txt:
        return ""

    for pattern, replacement in BUSINESS_MULTI_PATTERNS:
        txt = pattern.sub(replacement, txt)

    txt = BUSINESS_PUNCT_RE.sub(" ", txt)
    tokens = [t for t in txt.split() if t]
    kept = [t for t in tokens if t not in LEGAL_NOISE_TOKENS]
    return " ".join(kept).strip()


def business_tokens_for_matching(x: Any, *, min_len: int = 1, remove_stopwords: bool = False) -> List[str]:
    key = normalize_business_name(x)
    if not key:
        return []
    out: List[str] = []
    for t in key.split():
        if len(t) < min_len:
            continue
        if remove_stopwords and t in BUSINESS_STOPWORDS:
            continue
        out.append(t)
    return out


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
    if hasattr(x, "date"):
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
