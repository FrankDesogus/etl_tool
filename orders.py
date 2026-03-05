import json
import re
from typing import Any, Dict, List, Tuple

import pandas as pd
from rapidfuzz import fuzz, process

from normalize import (
    business_tokens_for_matching,
    normalize_amount,
    normalize_business_name,
    normalize_case_policy,
    normalize_text,
    parse_date_to_iso,
    add_norm_log,
)
from utils import now_iso

CDC_RE = re.compile(r"^\(?\s*CDC\b", re.IGNORECASE)


def _job_cdc_starts_with_cdc(job_cdc_value: Any) -> bool:
    """
    Classify CDC/non-CDC from JOB/CDC.

    Rule: CDC if text starts with "CDC" (case-insensitive), allowing optional
    leading parenthesis/spaces, e.g. "CDC-07/40", "CDC 5/21", "(CDC-05/20)".
    """
    job_cdc_norm = normalize_text(job_cdc_value)
    if not job_cdc_norm:
        return False
    return bool(CDC_RE.match(job_cdc_norm))


def _is_query_short_or_weak(order_supplier_key: str) -> bool:
    strong_tokens = [t for t in business_tokens_for_matching(order_supplier_key, min_len=5, remove_stopwords=True)]
    return len(order_supplier_key) < 4 or len(strong_tokens) == 0


def _controlled_token_containment(
    order_supplier_key: str,
    supplier_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    query_tokens = set(business_tokens_for_matching(order_supplier_key, min_len=3, remove_stopwords=True))
    strong_tokens = [t for t in query_tokens if len(t) >= 5]
    if not query_tokens or not strong_tokens:
        return []

    matches = []
    for rec in supplier_records:
        candidate_tokens = set(business_tokens_for_matching(rec["supplier_name_key"], min_len=3, remove_stopwords=True))
        if query_tokens.issubset(candidate_tokens):
            matches.append(rec)
    return matches


def _combined_scorer(order_key: str, supplier_key: str, allow_partial: bool) -> float:
    token_set = fuzz.token_set_ratio(order_key, supplier_key)
    if not allow_partial:
        return float(token_set)
    partial = fuzz.partial_ratio(order_key, supplier_key)
    return float(max(token_set, partial))


def clean_and_match_orders(
    df_orders: pd.DataFrame,
    df_sup: pd.DataFrame,
    low: int,
    high: int,
    norm_logs: List[Dict[str, Any]],
    match_warnings: List[Dict[str, Any]],
    unmatched_orders: List[Dict[str, Any]],
) -> pd.DataFrame:
    supplier_by_name = {
        r["supplier_name_normalized"]: r["supplier_id"] for _, r in df_sup.iterrows()
    }
    supplier_records = [
        {
            "supplier_id": r["supplier_id"],
            "supplier_name_normalized": r["supplier_name_normalized"],
            "supplier_name_key": r.get("supplier_name_key", normalize_business_name(r["supplier_name_normalized"])),
        }
        for _, r in df_sup.iterrows()
    ]

    supplier_ids_by_key: Dict[str, List[Dict[str, Any]]] = {}
    for rec in supplier_records:
        key = rec["supplier_name_key"]
        if key:
            supplier_ids_by_key.setdefault(key, []).append(rec)

    unique_supplier_keys = list(supplier_ids_by_key.keys())

    out: List[Dict[str, Any]] = []

    for _, r in df_orders.iterrows():
        sf, sh, sr = r["source_file"], r["source_sheet"], int(r["source_row"])
        order_id = f"ORD_{sh}_{sr}"

        supplier_raw = r.get("supplier_text_raw")
        supplier_norm = normalize_case_policy(supplier_raw)
        order_supplier_key = normalize_business_name(supplier_raw)
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
        add_norm_log(
            norm_logs,
            entity="order",
            record_id=order_id,
            field_name="supplier_key",
            original_value=supplier_raw,
            normalized_value=order_supplier_key,
            reason="business_name_normalization",
            source_file=sf,
            source_sheet=sh,
            source_row=sr,
            source_column="FORNITORE",
        )

        job_cdc_raw = r.get("job_cdc")
        job_cdc_norm = normalize_case_policy(job_cdc_raw)
        job_cdc_is_cdc = _job_cdc_starts_with_cdc(job_cdc_raw)

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

        order_date_iso = parse_date_to_iso(r.get("order_date_raw"))

        matched_id, method, score, conf = "", "none", "", "none"

        if not supplier_norm:
            unmatched_orders.append(
                {
                    "order_id": order_id,
                    "order_number": r.get("order_number", ""),
                    "order_date": order_date_iso,
                    "supplier_text_raw": normalize_text(supplier_raw),
                    "supplier_text_normalized": supplier_norm,
                    "order_supplier_key": order_supplier_key,
                    "reason": "missing_supplier_text",
                    "candidates_topn": "[]",
                    "source_file": sf,
                    "source_sheet": sh,
                    "source_row": sr,
                }
            )

        elif order_supplier_key in supplier_ids_by_key and len(supplier_ids_by_key[order_supplier_key]) == 1:
            matched_id = supplier_ids_by_key[order_supplier_key][0]["supplier_id"]
            method, score, conf = "exact_key", 100.0, "high"

        elif supplier_norm in supplier_by_name:
            matched_id = supplier_by_name[supplier_norm]
            method, score, conf = "exact", 100.0, "high"

        else:
            weak_query = _is_query_short_or_weak(order_supplier_key)
            best_raw: List[Tuple[str, float, int]] = process.extract(
                order_supplier_key,
                unique_supplier_keys,
                scorer=lambda q, c, **_: _combined_scorer(q, c, allow_partial=not weak_query),
                limit=5,
            ) if order_supplier_key and unique_supplier_keys else []

            top = []
            for name_key, sc, _ in best_raw:
                first_rec = supplier_ids_by_key[name_key][0]
                top.append(
                    {
                        "supplier_id": first_rec["supplier_id"],
                        "name": first_rec["supplier_name_normalized"],
                        "supplier_key": name_key,
                        "score": sc,
                    }
                )

            best_name_key = ""
            best_score = 0.0
            best_rec = None
            if best_raw:
                best_name_key, best_score, _ = best_raw[0]
                best_rec = supplier_ids_by_key[best_name_key][0]

            containment_matches = _controlled_token_containment(order_supplier_key, supplier_records)
            containment_unique = None
            if len(containment_matches) == 1:
                containment_unique = containment_matches[0]

            auto_accepted = False
            if containment_unique is not None and best_score >= (high - 2) and not weak_query:
                matched_id = containment_unique["supplier_id"]
                method, score, conf = "token_containment", float(best_score), "high"
                auto_accepted = True
                match_warnings.append(
                    {
                        "timestamp": now_iso(),
                        "warning_type": "token_containment_accepted",
                        "decision": "accepted",
                        "order_id": order_id,
                        "order_number": r.get("order_number", ""),
                        "order_date": order_date_iso,
                        "order_supplier_raw": normalize_text(supplier_raw),
                        "order_supplier_normalized": supplier_norm,
                        "order_supplier_key": order_supplier_key,
                        "candidate_supplier_id": containment_unique["supplier_id"],
                        "candidate_supplier_name_normalized": containment_unique["supplier_name_normalized"],
                        "candidate_supplier_key": containment_unique["supplier_name_key"],
                        "similarity_score": best_score,
                        "candidates_topn": json.dumps(top, ensure_ascii=False),
                        "source_file": sf,
                        "source_sheet": sh,
                        "source_row": sr,
                    }
                )

            if not auto_accepted:
                best_rec = best_rec or {"supplier_id": "", "supplier_name_normalized": "", "supplier_name_key": ""}
                if best_score >= high and not weak_query:
                    matched_id = best_rec["supplier_id"]
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
                            "order_supplier_key": order_supplier_key,
                            "candidate_supplier_id": matched_id,
                            "candidate_supplier_name_normalized": best_rec["supplier_name_normalized"],
                            "candidate_supplier_key": best_rec["supplier_name_key"],
                            "similarity_score": best_score,
                            "candidates_topn": json.dumps(top, ensure_ascii=False),
                            "source_file": sf,
                            "source_sheet": sh,
                            "source_row": sr,
                        }
                    )
                elif low <= best_score < high or weak_query:
                    reason = "gray_zone_ambiguous" if not weak_query else "short_or_weak_supplier_key"
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
                            "order_supplier_key": order_supplier_key,
                            "candidate_supplier_id": best_rec["supplier_id"],
                            "candidate_supplier_name_normalized": best_rec["supplier_name_normalized"],
                            "candidate_supplier_key": best_rec["supplier_name_key"],
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
                            "order_supplier_key": order_supplier_key,
                            "reason": reason,
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
                            "order_supplier_key": order_supplier_key,
                            "reason": "no_candidate_above_low",
                            "candidates_topn": json.dumps(top, ensure_ascii=False),
                            "source_file": sf,
                            "source_sheet": sh,
                            "source_row": sr,
                        }
                    )

        out.append(
            {
                "order_id": order_id,
                "order_number": r.get("order_number", ""),
                "order_date_raw": normalize_text(r.get("order_date_raw")),
                "order_date": order_date_iso,
                "operator": normalize_text(r.get("operator")),
                "supplier_text_raw": normalize_text(supplier_raw),
                "supplier_text_normalized": supplier_norm,
                "order_supplier_key": order_supplier_key,
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
