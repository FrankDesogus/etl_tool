import json
import re
from typing import Any, Dict, List

import pandas as pd
from rapidfuzz import fuzz, process

from normalize import (
    normalize_text,
    normalize_case_policy,
    normalize_amount,
    parse_date_to_iso,
    add_norm_log,
)
from utils import now_iso

# CDC if starts with "CDC" possibly preceded by "(" and/or spaces.
# Examples matched: "CDC-7/34", "(CDC-05/20)", " CDC 5/21 ..."
CDC_RE = re.compile(r"^\(?\s*CDC\b", re.IGNORECASE)


def clean_and_match_orders(
    df_orders: pd.DataFrame,
    df_sup: pd.DataFrame,
    low: int,
    high: int,
    norm_logs: List[Dict[str, Any]],
    match_warnings: List[Dict[str, Any]],
    unmatched_orders: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Cleans orders + matches supplier text -> suppliers list.
    Adds CDC classification columns:
      - job_cdc_normalized
      - job_cdc_is_cdc (bool)
    Logs normalization for supplier_text_raw and job_cdc (audit).
    """
    supplier_by_name = {
        r["supplier_name_normalized"]: r["supplier_id"] for _, r in df_sup.iterrows()
    }
    supplier_names = list(supplier_by_name.keys())

    out: List[Dict[str, Any]] = []

    for _, r in df_orders.iterrows():
        sf, sh, sr = r["source_file"], r["source_sheet"], int(r["source_row"])
        order_id = f"ORD_{sh}_{sr}"

        # --- Supplier normalization + log ---
        supplier_raw = r.get("supplier_text_raw")
        supplier_norm = normalize_case_policy(supplier_raw)
        add_norm_log(
            norm_logs,
            entity="order",
            record_id=order_id,
            field_name="supplier_text_raw",
            original_value=supplier_raw,
            normalized_value=supplier_norm,
            reason="text_trim_invisible_collapse + UPPER",
            source_file=sf,
            source_sheet=sh,
            source_row=sr,
            source_column="FORNITORE",
        )

        # --- JOB/CDC normalization + CDC flag + log ---
        job_cdc_raw = r.get("job_cdc")
        job_cdc_norm = normalize_case_policy(job_cdc_raw)
        job_cdc_is_cdc = bool(CDC_RE.match(job_cdc_norm))

        add_norm_log(
            norm_logs,
            entity="order",
            record_id=order_id,
            field_name="job_cdc",
            original_value=job_cdc_raw,
            normalized_value=job_cdc_norm,
            reason="text_trim_invisible_collapse + UPPER",
            source_file=sf,
            source_sheet=sh,
            source_row=sr,
            source_column="JOB / CDC",
        )

        # --- Date parsing (optionally log if you want; leaving as-is like your current file) ---
        order_date_iso = parse_date_to_iso(r.get("order_date_raw"))

        # --- Matching ---
        matched_id, method, score, conf = "", "none", "", "none"

        if not supplier_norm:
            unmatched_orders.append(
                {
                    "order_id": order_id,
                    "order_number": r.get("order_number", ""),
                    "order_date": order_date_iso,
                    "supplier_text_raw": normalize_text(supplier_raw),
                    "supplier_text_normalized": supplier_norm,
                    "reason": "missing_supplier_text",
                    "candidates_topn": "[]",
                    "source_file": sf,
                    "source_sheet": sh,
                    "source_row": sr,
                }
            )

        elif supplier_norm in supplier_by_name:
            matched_id = supplier_by_name[supplier_norm]
            method, score, conf = "exact", 100.0, "high"

        else:
            best = process.extract(
                supplier_norm,
                supplier_names,
                scorer=fuzz.token_sort_ratio,
                limit=5,
            )
            top = [
                {"supplier_id": supplier_by_name[n], "name": n, "score": sc}
                for n, sc, _ in best
            ]
            best_name, best_score, _ = best[0]

            if best_score >= high:
                matched_id = supplier_by_name[best_name]
                method, score, conf = "fuzzy", float(best_score), "high"
                match_warnings.append(
                    {
                        "timestamp": now_iso(),
                        "warning_type": "fuzzy_accepted",
                        "decision": "accepted",
                        "order_id": order_id,
                        "order_number": r.get("order_number", ""),
                        "order_date": order_date_iso,
                        "order_supplier_raw": normalize_text(supplier_raw),
                        "order_supplier_normalized": supplier_norm,
                        "candidate_supplier_id": matched_id,
                        "candidate_supplier_name_normalized": best_name,
                        "similarity_score": best_score,
                        "candidates_topn": json.dumps(top, ensure_ascii=False),
                        "source_file": sf,
                        "source_sheet": sh,
                        "source_row": sr,
                    }
                )

            elif low <= best_score < high:
                match_warnings.append(
                    {
                        "timestamp": now_iso(),
                        "warning_type": "fuzzy_gray_zone",
                        "decision": "manual_review",
                        "order_id": order_id,
                        "order_number": r.get("order_number", ""),
                        "order_date": order_date_iso,
                        "order_supplier_raw": normalize_text(supplier_raw),
                        "order_supplier_normalized": supplier_norm,
                        "candidate_supplier_id": supplier_by_name[best_name],
                        "candidate_supplier_name_normalized": best_name,
                        "similarity_score": best_score,
                        "candidates_topn": json.dumps(top, ensure_ascii=False),
                        "source_file": sf,
                        "source_sheet": sh,
                        "source_row": sr,
                    }
                )
                unmatched_orders.append(
                    {
                        "order_id": order_id,
                        "order_number": r.get("order_number", ""),
                        "order_date": order_date_iso,
                        "supplier_text_raw": normalize_text(supplier_raw),
                        "supplier_text_normalized": supplier_norm,
                        "reason": "gray_zone_ambiguous",
                        "candidates_topn": json.dumps(top, ensure_ascii=False),
                        "source_file": sf,
                        "source_sheet": sh,
                        "source_row": sr,
                    }
                )

            else:
                unmatched_orders.append(
                    {
                        "order_id": order_id,
                        "order_number": r.get("order_number", ""),
                        "order_date": order_date_iso,
                        "supplier_text_raw": normalize_text(supplier_raw),
                        "supplier_text_normalized": supplier_norm,
                        "reason": "no_candidate_above_low",
                        "candidates_topn": json.dumps(top, ensure_ascii=False),
                        "source_file": sf,
                        "source_sheet": sh,
                        "source_row": sr,
                    }
                )

        # --- Output row ---
        out.append(
            {
                "order_id": order_id,
                "order_number": r.get("order_number", ""),
                "order_date_raw": normalize_text(r.get("order_date_raw")),
                "order_date": order_date_iso,
                "operator": normalize_text(r.get("operator")),
                "supplier_text_raw": normalize_text(supplier_raw),
                "supplier_text_normalized": supplier_norm,
                "amount_total_raw": normalize_text(r.get("amount_total_raw")),
                "amount_total": normalize_amount(r.get("amount_total_raw")),
                "amount_open_raw": normalize_text(r.get("amount_open_raw")),
                "amount_open": normalize_amount(r.get("amount_open_raw")),
                "amount_fulfilled_raw": normalize_text(r.get("amount_fulfilled_raw")),
                "amount_fulfilled": normalize_amount(r.get("amount_fulfilled_raw")),
                "status": normalize_text(r.get("status")),
                "job_cdc": normalize_text(job_cdc_raw),
                "job_cdc_normalized": job_cdc_norm,
                "job_cdc_is_cdc": job_cdc_is_cdc,
                "requested_date": parse_date_to_iso(r.get("requested_date_raw")),
                "confirmed_date": parse_date_to_iso(r.get("confirmed_date_raw")),
                "effective_date": parse_date_to_iso(r.get("effective_date_raw")),
                "matched_supplier_id": matched_id,
                "match_method": method,
                "match_score": score,
                "match_confidence": conf,
                "source_file": sf,
                "source_sheet": sh,
                "source_row": sr,
                "source_columns_used": r.get("source_columns_used", "[]"),
            }
        )

    return pd.DataFrame(out)