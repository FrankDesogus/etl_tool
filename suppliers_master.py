import json
import re
from typing import Any, Dict, List, Tuple

import pandas as pd
from rapidfuzz import fuzz, process

from normalize import business_tokens_for_matching
from utils import now_iso


def _sanitize_uid_token(value: Any) -> str:
    text = "" if value is None else str(value).strip().upper()
    if not text:
        return "UNKNOWN"
    cleaned = re.sub(r"[^A-Z0-9]+", "_", text)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "UNKNOWN"


def _is_query_short_or_weak(order_supplier_key: str) -> bool:
    strong_tokens = [t for t in business_tokens_for_matching(order_supplier_key, min_len=5, remove_stopwords=True)]
    return len(order_supplier_key) < 4 or len(strong_tokens) == 0


def _controlled_token_containment(order_supplier_key: str, supplier_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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


def _aggregate_certifications(df_certs: pd.DataFrame) -> pd.DataFrame:
    if df_certs is None or df_certs.empty:
        return pd.DataFrame(columns=[
            "matched_registry_supplier_id",
            "cert_count",
            "cert_with_expiry_count",
            "earliest_expiry",
            "latest_expiry",
        ])

    work = df_certs.copy()
    work["supplier_id"] = work.get("supplier_id", "").fillna("").astype(str).str.strip()
    work["expiry_date"] = work.get("expiry_date", "").fillna("").astype(str).str.strip()
    work["expiry_dt"] = pd.to_datetime(work["expiry_date"], errors="coerce")

    agg = (
        work.groupby("supplier_id", dropna=False)
        .agg(
            cert_count=("supplier_id", "size"),
            cert_with_expiry_count=("expiry_dt", lambda s: int(s.notna().sum())),
            earliest_expiry=("expiry_dt", "min"),
            latest_expiry=("expiry_dt", "max"),
        )
        .reset_index()
        .rename(columns={"supplier_id": "matched_registry_supplier_id"})
    )
    agg["earliest_expiry"] = agg["earliest_expiry"].dt.strftime("%Y-%m-%d").fillna("")
    agg["latest_expiry"] = agg["latest_expiry"].dt.strftime("%Y-%m-%d").fillna("")
    return agg


def build_suppliers_master(
    df_orders_clean: pd.DataFrame,
    df_registry_suppliers: pd.DataFrame,
    df_certs: pd.DataFrame,
    low: int,
    high: int,
    match_warnings: List[Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    orders = df_orders_clean.copy() if df_orders_clean is not None else pd.DataFrame()
    registry = df_registry_suppliers.copy() if df_registry_suppliers is not None else pd.DataFrame()

    if orders.empty:
        empty = pd.DataFrame()
        return empty, empty, registry, pd.DataFrame(columns=["registry_supplier_id", "supplier_external_uid"])

    for col in [
        "order_supplier_key", "supplier_text_raw", "order_id", "source_sheet", "source_row", "source_file",
        "matched_supplier_id", "match_method", "match_score", "match_confidence"
    ]:
        if col not in orders.columns:
            orders[col] = ""

    orders["order_supplier_key"] = orders["order_supplier_key"].fillna("").astype(str).str.strip()
    grouped = []
    for key, grp in orders.groupby("order_supplier_key", dropna=False):
        grp = grp.sort_values(by=["source_file", "source_sheet", "source_row"], kind="stable")
        first = grp.iloc[0]
        raw_examples = [x for x in grp["supplier_text_raw"].fillna("").astype(str).str.strip().tolist() if x]
        uniq_examples = list(dict.fromkeys(raw_examples))[:3]

        grouped.append({
            "supplier_external_uid": f"SUPORD::{_sanitize_uid_token(key)}",
            "supplier_name_raw": uniq_examples[0] if uniq_examples else "",
            "supplier_name_key": key,
            "supplier_name_raw_examples": json.dumps(uniq_examples, ensure_ascii=False),
            "orders_count": int(len(grp)),
            "first_seen_order_id": str(first.get("order_id", "")),
            "first_seen_source_sheet": str(first.get("source_sheet", "")),
            "first_seen_source_row": str(first.get("source_row", "")),
            "first_seen_source_file": str(first.get("source_file", "")),
            "matched_registry_supplier_id": "",
            "match_method": "none",
            "match_score": "",
            "match_confidence": "none",
        })

    master = pd.DataFrame(grouped)

    registry_records = []
    for _, r in registry.iterrows():
        registry_records.append({
            "supplier_id": r.get("supplier_id", ""),
            "supplier_name_normalized": r.get("supplier_name_normalized", ""),
            "supplier_name_key": r.get("supplier_name_key", ""),
        })

    by_key: Dict[str, List[Dict[str, Any]]] = {}
    for rec in registry_records:
        key = str(rec.get("supplier_name_key", "")).strip()
        if key:
            by_key.setdefault(key, []).append(rec)
    unique_registry_keys = list(by_key.keys())

    for idx, row in master.iterrows():
        key = str(row.get("supplier_name_key", "")).strip()
        matched_id, method, score, conf = "", "none", "", "none"

        # Prefer consensus from order-level matches if available
        order_subset = orders[orders["order_supplier_key"] == key]
        matched_ids = [x for x in order_subset["matched_supplier_id"].fillna("").astype(str).str.strip().tolist() if x]
        unique_matched_ids = list(dict.fromkeys(matched_ids))
        if len(unique_matched_ids) == 1:
            matched_id = unique_matched_ids[0]
            first_match = order_subset[order_subset["matched_supplier_id"].astype(str).str.strip() == matched_id].iloc[0]
            method = str(first_match.get("match_method", "order_match") or "order_match")
            score = first_match.get("match_score", "")
            conf = str(first_match.get("match_confidence", "high") or "high")
        elif key in by_key and len(by_key[key]) == 1:
            matched_id = by_key[key][0]["supplier_id"]
            method, score, conf = "exact_key", 100.0, "high"
        else:
            weak_query = _is_query_short_or_weak(key)
            best_raw = process.extract(
                key,
                unique_registry_keys,
                scorer=lambda q, c, **_: _combined_scorer(q, c, allow_partial=not weak_query),
                limit=5,
            ) if key and unique_registry_keys else []

            top = []
            for name_key, sc, _ in best_raw:
                rec = by_key[name_key][0]
                top.append({
                    "supplier_id": rec["supplier_id"],
                    "name": rec["supplier_name_normalized"],
                    "supplier_key": name_key,
                    "score": sc,
                })

            best_rec = None
            best_score = 0.0
            if best_raw:
                best_name_key, best_score, _ = best_raw[0]
                best_rec = by_key[best_name_key][0]

            containment_matches = _controlled_token_containment(key, registry_records)
            containment_unique = containment_matches[0] if len(containment_matches) == 1 else None

            if containment_unique is not None and best_score >= (high - 2) and not weak_query:
                matched_id = containment_unique["supplier_id"]
                method, score, conf = "token_containment", float(best_score), "high"
            elif best_rec is not None and best_score >= high and not weak_query:
                matched_id = best_rec["supplier_id"]
                method, score, conf = "fuzzy", float(best_score), "high"
            elif (best_rec is not None and low <= best_score < high) or weak_query:
                match_warnings.append({
                    "timestamp": now_iso(),
                    "warning_type": "supplier_master_fuzzy_gray_zone",
                    "decision": "manual_review",
                    "order_id": "",
                    "order_number": "",
                    "order_date": "",
                    "order_supplier_raw": row.get("supplier_name_raw", ""),
                    "order_supplier_normalized": "",
                    "order_supplier_key": key,
                    "candidate_supplier_id": "" if best_rec is None else best_rec.get("supplier_id", ""),
                    "candidate_supplier_name_normalized": "" if best_rec is None else best_rec.get("supplier_name_normalized", ""),
                    "candidate_supplier_key": "" if best_rec is None else best_rec.get("supplier_name_key", ""),
                    "similarity_score": best_score,
                    "candidates_topn": json.dumps(top, ensure_ascii=False),
                    "source_file": row.get("first_seen_source_file", ""),
                    "source_sheet": row.get("first_seen_source_sheet", ""),
                    "source_row": row.get("first_seen_source_row", ""),
                })

        master.loc[idx, "matched_registry_supplier_id"] = matched_id
        master.loc[idx, "match_method"] = method
        master.loc[idx, "match_score"] = score
        master.loc[idx, "match_confidence"] = conf

    # Bring registry fields under registry_* namespace
    registry_prefixed = registry.rename(columns={c: f"registry_{c}" for c in registry.columns if c != "supplier_id"})
    registry_prefixed = registry_prefixed.rename(columns={"supplier_id": "matched_registry_supplier_id"})
    master = master.merge(registry_prefixed, how="left", on="matched_registry_supplier_id")

    cert_agg = _aggregate_certifications(df_certs)
    master = master.merge(cert_agg, how="left", on="matched_registry_supplier_id")
    master["cert_count"] = master["cert_count"].fillna(0).astype(int)
    master["cert_with_expiry_count"] = master["cert_with_expiry_count"].fillna(0).astype(int)
    master["earliest_expiry"] = master["earliest_expiry"].fillna("")
    master["latest_expiry"] = master["latest_expiry"].fillna("")
    master["has_certifications"] = master["cert_count"] > 0

    def _quality(row: pd.Series) -> str:
        in_registry = bool(str(row.get("matched_registry_supplier_id", "")).strip())
        if not in_registry:
            return "MISSING_IN_REGISTRY"
        if int(row.get("cert_count", 0)) == 0:
            return "IN_REGISTRY_NO_CERTS"
        if int(row.get("cert_with_expiry_count", 0)) < int(row.get("cert_count", 0)):
            return "CERTS_MISSING_EXPIRY"
        return "IN_REGISTRY"

    master["data_quality_status"] = master.apply(_quality, axis=1)

    unmatched_master = master[master["matched_registry_supplier_id"].fillna("").astype(str).str.strip() == ""].copy()

    used_registry_ids = set(master["matched_registry_supplier_id"].fillna("").astype(str).str.strip())
    used_registry_ids.discard("")
    registry_not_used = registry[~registry["supplier_id"].astype(str).isin(used_registry_ids)].copy()

    registry_to_master = master[["matched_registry_supplier_id", "supplier_external_uid"]].copy()
    registry_to_master = registry_to_master[registry_to_master["matched_registry_supplier_id"].fillna("").astype(str).str.strip() != ""]
    registry_to_master = registry_to_master.drop_duplicates(subset=["matched_registry_supplier_id"], keep="first")
    registry_to_master = registry_to_master.rename(columns={"matched_registry_supplier_id": "registry_supplier_id"})

    return master, unmatched_master, registry_not_used, registry_to_master
