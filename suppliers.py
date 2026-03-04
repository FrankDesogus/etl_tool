import json
import hashlib
import pandas as pd
from typing import Any, Dict, List, Tuple
from rapidfuzz import fuzz, process
from normalize import normalize_business_name, normalize_case_policy, normalize_text, add_norm_log


# Contract: supplier_id is generated once during supplier dedup and reused across downstream datasets.
def _stable_supplier_id(supplier_name_normalized: str) -> str:
    digest = hashlib.sha1(supplier_name_normalized.encode("utf-8")).hexdigest()[:12].upper()
    return f"SUP_{digest}"

def build_suppliers_clean(df_sup: pd.DataFrame, supplier_category: str,
                          norm_logs: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cleaned, bad = [], []
    for _, r in df_sup.iterrows():
        sf, sh, sr = r["source_file"], r["source_sheet"], int(r["source_row"])
        temp_key = f"{sh}:{sr}"

        name_raw = r.get("supplier_name_raw")
        name_norm = normalize_case_policy(name_raw)
        name_key = normalize_business_name(name_raw)
        add_norm_log(norm_logs, entity="supplier", record_id=temp_key,
                     field_name="supplier_name_raw", original_value=name_raw,
                     normalized_value=name_norm,
                     reason="text_trim_invisible_collapse + UPPER",
                     source_file=sf, source_sheet=sh, source_row=sr,
                     source_column="RAGIONE SOCIALE")

        add_norm_log(norm_logs, entity="supplier", record_id=temp_key,
                     field_name="supplier_name_key", original_value=name_raw,
                     normalized_value=name_key,
                     reason="business_name_normalization",
                     source_file=sf, source_sheet=sh, source_row=sr,
                     source_column="RAGIONE SOCIALE")

        if name_norm == "":
            bad.append({
                "temp_supplier_key": temp_key,
                "supplier_name_raw": "" if name_raw is None else str(name_raw),
                "reason": "empty_name",
                "source_file": sf,
                "source_sheet": sh,
                "source_row": sr,
            })
            continue

        def norm_field(field: str) -> str:
            orig = r.get(field)
            norm = normalize_text(orig)
            if ("" if orig is None else str(orig)) != norm:
                add_norm_log(norm_logs, entity="supplier", record_id=temp_key,
                             field_name=field, original_value=orig, normalized_value=norm,
                             reason="text_trim_invisible_collapse",
                             source_file=sf, source_sheet=sh, source_row=sr,
                             source_column=normalize_text(r.get(f"{field}__source_column")))
            return norm

        cleaned.append({
            "supplier_id": "",  # assegnato dopo dedup
            "supplier_name_raw": normalize_text(name_raw),
            "supplier_name_normalized": name_norm,
            "supplier_name_key": name_key,
            "supplier_category": json.dumps([supplier_category], ensure_ascii=False),
            "supplier_type": norm_field("supplier_type"),
            "supply_scope": norm_field("supply_scope"),
            "certifications_raw": norm_field("certifications_raw"),
            "cert_expiry_raw": norm_field("cert_expiry_raw"),
            "cert_expiry_source_column": normalize_text(r.get("cert_expiry_raw__source_column")),
            "kpi_quality": norm_field("kpi_quality"),
            "kpi_time": norm_field("kpi_time"),
            "distributor_or_broker": norm_field("distributor_or_broker"),
            "notes": norm_field("notes"),
            "source_file": sf,
            "provenance": json.dumps([{
                "sheet": sh,
                "row": sr,
                "cols_used": json.loads(r.get("source_columns_used", "[]")),
            }], ensure_ascii=False),
            "extra_fields": json.dumps({}, ensure_ascii=False),
        })

    return pd.DataFrame(cleaned), pd.DataFrame(bad)

def deduplicate_suppliers(df_sup_clean: pd.DataFrame,
                          match_warnings: List[Dict[str, Any]],
                          low_threshold: int) -> pd.DataFrame:
    grouped = []
    for name_norm, grp in df_sup_clean.groupby("supplier_name_normalized"):
        rec = grp.iloc[0].to_dict()
        if len(grp) > 1:
            # merge provenance + category
            cats, prov = [], []
            for _, row in grp.iterrows():
                cats += json.loads(row["supplier_category"])
                prov += json.loads(row["provenance"])
            rec["supplier_category"] = json.dumps(sorted(set(cats)), ensure_ascii=False)
            rec["provenance"] = json.dumps(prov, ensure_ascii=False)
        grouped.append(rec)

    df_dedup = pd.DataFrame(grouped)
    df_dedup["supplier_id"] = df_dedup["supplier_name_normalized"].apply(_stable_supplier_id)

    # possible duplicates warning (no merge)
    names = df_dedup["supplier_name_normalized"].tolist()
    for i, name in enumerate(names):
        matches = process.extract(name, names, scorer=fuzz.token_sort_ratio, limit=4)
        for mname, score, _ in matches[1:]:
            if low_threshold <= score < 100:
                match_warnings.append({
                    "timestamp": "",
                    "warning_type": "possible_supplier_duplicate",
                    "decision": "manual_review",
                    "order_id": "",
                    "order_number": "",
                    "order_date": "",
                    "order_supplier_raw": "",
                    "order_supplier_normalized": "",
                    "candidate_supplier_id": "",
                    "candidate_supplier_name_normalized": f"{name} <> {mname}",
                    "similarity_score": score,
                    "candidates_topn": "[]",
                    "source_file": df_dedup.iloc[i].get("source_file", ""),
                    "source_sheet": "",
                    "source_row": "",
                })

    return df_dedup
