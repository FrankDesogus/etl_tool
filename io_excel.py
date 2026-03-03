import json
import pandas as pd
from typing import Dict, Optional, Set, Tuple
from normalize import normalize_case_policy

def sheet_is_orders(df_preview: pd.DataFrame, order_keywords: Set[str]) -> bool:
    for _, row in df_preview.iterrows():
        cells = {normalize_case_policy(c) for c in row.tolist() if c is not None}
        if len(order_keywords.intersection(cells)) >= 3:
            return True
    return False

def find_header_row(df: pd.DataFrame, keywords: Set[str], min_hits: int) -> Optional[int]:
    for i in range(min(60, len(df))):
        cells = {normalize_case_policy(c) for c in df.iloc[i].tolist() if c is not None}
        if len(keywords.intersection(cells)) >= min_hits:
            return i
    return None

def extract_table_with_provenance(xlsx_path: str, sheet_name: str, kind: str,
                                  order_keywords: Set[str], supplier_keywords: Set[str]
                                  ) -> Tuple[pd.DataFrame, int]:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None, engine="openpyxl")
    if kind == "orders":
        hdr = find_header_row(df, order_keywords, min_hits=3)
    else:
        hdr = find_header_row(df, supplier_keywords, min_hits=1)

    if hdr is None:
        return pd.DataFrame(), -1

    header_values = [normalize_case_policy(x) for x in df.iloc[hdr].tolist()]
    df_data = df.iloc[hdr+1:].copy()
    df_data.columns = header_values

    excel_row_start = hdr + 2  # 1-based, first data row after header
    df_data["source_file"] = xlsx_path.split("/")[-1]
    df_data["source_sheet"] = sheet_name
    df_data["source_row"] = list(range(excel_row_start, excel_row_start + len(df_data)))
    return df_data, hdr

def canonicalize_columns(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    out = pd.DataFrame()
    for p in ["source_file", "source_sheet", "source_row"]:
        if p in df.columns:
            out[p] = df[p]

    reverse = {}
    for src_col in df.columns:
        src_norm = normalize_case_policy(src_col)
        if src_norm in mapping:
            reverse.setdefault(mapping[src_norm], []).append(src_col)

    for tgt, src_cols in reverse.items():
        if len(src_cols) == 1:
            out[tgt] = df[src_cols[0]]
        else:
            def concat_row(r):
                parts = []
                for c in src_cols:
                    v = r.get(c)
                    if v is not None and str(v).strip() != "":
                        parts.append(str(v).strip())
                return "; ".join(parts)
            out[tgt] = df.apply(concat_row, axis=1)

    used_cols = sorted([c for cols in reverse.values() for c in cols])
    out["source_columns_used"] = json.dumps(used_cols, ensure_ascii=False)
    return out