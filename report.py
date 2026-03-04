import json
import os
import pandas as pd
from typing import Any, Dict, List
from normalize import normalize_business_name
from utils import now_iso


def build_orders_supplier_cert_report(
    orders: pd.DataFrame,
    suppliers: pd.DataFrame,
    certs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build an audit-friendly orders -> suppliers -> certifications report.
    Uses matched_supplier_id when available, otherwise exact-by-normalized-name fallback.
    """
    order_cols = [
        "order_id",
        "order_number",
        "order_date",
        "job_cdc",
        "job_cdc_is_cdc",
        "supplier_text_raw",
        "supplier_text_normalized",
        "order_supplier_key",
        "matched_supplier_id",
        "match_method",
        "match_score",
        "match_confidence",
        "source_file",
        "source_sheet",
        "source_row",
    ]
    supplier_cols = ["supplier_id", "supplier_name_normalized", "supplier_name_key", "supplier_category"]

    if orders is None or orders.empty:
        return pd.DataFrame(
            columns=order_cols
            + [
                "supplier_id_effective",
                "supplier_exists",
                "supplier_name_normalized",
                "supplier_category",
                "cert_count",
                "cert_names",
                "cert_with_expiry_count",
                "earliest_expiry",
                "latest_expiry",
            ]
        )

    orders_work = orders.copy()
    for c in order_cols:
        if c not in orders_work.columns:
            orders_work[c] = ""

    suppliers_work = suppliers.copy() if suppliers is not None else pd.DataFrame()
    for c in supplier_cols:
        if c not in suppliers_work.columns:
            suppliers_work[c] = ""

    if "order_supplier_key" not in orders_work.columns:
        orders_work["order_supplier_key"] = orders_work["supplier_text_normalized"].apply(normalize_business_name)
    if "supplier_name_key" not in suppliers_work.columns:
        suppliers_work["supplier_name_key"] = suppliers_work["supplier_name_normalized"].apply(normalize_business_name)

    fallback_lookup = (
        suppliers_work[["supplier_name_key", "supplier_id"]]
        .dropna(subset=["supplier_name_key", "supplier_id"])
        .drop_duplicates(subset=["supplier_name_key"], keep="first")
        .rename(columns={"supplier_id": "fallback_supplier_id"})
    )

    report_df = orders_work.merge(
        fallback_lookup,
        how="left",
        left_on="order_supplier_key",
        right_on="supplier_name_key",
    ).drop(columns=["supplier_name_key"], errors="ignore")

    report_df["supplier_id_effective"] = report_df["matched_supplier_id"].where(
        report_df["matched_supplier_id"].astype(str).str.strip() != "",
        report_df["fallback_supplier_id"],
    )
    report_df["supplier_exists"] = report_df["supplier_id_effective"].notna() & (
        report_df["supplier_id_effective"].astype(str).str.strip() != ""
    )

    supplier_info = suppliers_work[supplier_cols].drop_duplicates(subset=["supplier_id"], keep="first")
    supplier_info = supplier_info.rename(columns={"supplier_id": "supplier_id_effective"})
    report_df = report_df.merge(supplier_info, how="left", on="supplier_id_effective")

    certs_work = certs.copy() if certs is not None else pd.DataFrame()
    if certs_work.empty:
        cert_agg = pd.DataFrame(
            columns=[
                "supplier_id_effective",
                "cert_count",
                "cert_names",
                "cert_with_expiry_count",
                "earliest_expiry",
                "latest_expiry",
            ]
        )
    else:
        if "supplier_id" not in certs_work.columns:
            certs_work["supplier_id"] = ""
        if "cert_name_normalized" not in certs_work.columns:
            certs_work["cert_name_normalized"] = ""
        if "expiry_date" not in certs_work.columns:
            certs_work["expiry_date"] = ""

        certs_work["cert_name_normalized"] = certs_work["cert_name_normalized"].fillna("").astype(str).str.strip()
        certs_work["expiry_date"] = certs_work["expiry_date"].fillna("").astype(str).str.strip()
        certs_work["expiry_date_dt"] = pd.to_datetime(certs_work["expiry_date"], errors="coerce")

        cert_agg = (
            certs_work.groupby("supplier_id", dropna=False)
            .agg(
                cert_count=("supplier_id", "size"),
                cert_names=(
                    "cert_name_normalized",
                    lambda s: "; ".join(sorted({x for x in s if x})),
                ),
                cert_with_expiry_count=("expiry_date_dt", lambda s: int(s.notna().sum())),
                earliest_expiry=("expiry_date_dt", "min"),
                latest_expiry=("expiry_date_dt", "max"),
            )
            .reset_index()
            .rename(columns={"supplier_id": "supplier_id_effective"})
        )
        cert_agg["earliest_expiry"] = cert_agg["earliest_expiry"].dt.strftime("%Y-%m-%d").fillna("")
        cert_agg["latest_expiry"] = cert_agg["latest_expiry"].dt.strftime("%Y-%m-%d").fillna("")

    report_df = report_df.merge(cert_agg, how="left", on="supplier_id_effective")

    report_df["cert_count"] = report_df["cert_count"].fillna(0).astype(int)
    report_df["cert_names"] = report_df["cert_names"].fillna("")
    report_df["cert_with_expiry_count"] = report_df["cert_with_expiry_count"].fillna(0).astype(int)
    report_df["earliest_expiry"] = report_df["earliest_expiry"].fillna("")
    report_df["latest_expiry"] = report_df["latest_expiry"].fillna("")
    report_df["supplier_name_normalized"] = report_df["supplier_name_normalized"].fillna("")
    report_df["supplier_category"] = report_df["supplier_category"].fillna("")

    final_cols = order_cols + [
        "supplier_id_effective",
        "supplier_exists",
        "supplier_name_normalized",
        "supplier_category",
        "cert_count",
        "cert_names",
        "cert_with_expiry_count",
        "earliest_expiry",
        "latest_expiry",
    ]
    return report_df[final_cols]

def write_outputs(out_dir: str,
                  suppliers: pd.DataFrame,
                  certs: pd.DataFrame,
                  orders: pd.DataFrame,
                  orders_supplier_cert_report: pd.DataFrame,
                  norm_logs: List[Dict[str, Any]],
                  match_warnings: List[Dict[str, Any]],
                  unmatched_orders: List[Dict[str, Any]],
                  unmatched_suppliers: pd.DataFrame,
                  cert_warnings: List[Dict[str, Any]],
                  summary: Dict[str, Any],
                  suppliers_master: pd.DataFrame | None = None,
                  suppliers_registry_not_used_in_orders: pd.DataFrame | None = None,
                  suppliers_master_unmatched_registry: pd.DataFrame | None = None,
                  registry_to_master_mapping: pd.DataFrame | None = None) -> None:
    os.makedirs(out_dir, exist_ok=True)

    suppliers.to_csv(os.path.join(out_dir, "suppliers_clean.csv"), index=False)
    suppliers.to_csv(os.path.join(out_dir, "suppliers_registry_clean.csv"), index=False)
    certs.to_csv(os.path.join(out_dir, "certifications_clean.csv"), index=False)
    orders.to_csv(os.path.join(out_dir, "orders_clean.csv"), index=False)
    orders_supplier_cert_report.to_csv(
        os.path.join(out_dir, "orders_supplier_cert_report.csv"), index=False
    )

    if suppliers_master is not None:
        suppliers_master.to_csv(os.path.join(out_dir, "suppliers_master.csv"), index=False)
    if suppliers_master_unmatched_registry is not None:
        suppliers_master_unmatched_registry.to_csv(
            os.path.join(out_dir, "suppliers_master_unmatched_registry.csv"), index=False
        )
    if suppliers_registry_not_used_in_orders is not None:
        suppliers_registry_not_used_in_orders.to_csv(
            os.path.join(out_dir, "suppliers_registry_not_used_in_orders.csv"), index=False
        )
    if registry_to_master_mapping is not None:
        registry_to_master_mapping.to_csv(
            os.path.join(out_dir, "registry_to_supplier_master_mapping.csv"), index=False
        )

    pd.DataFrame(norm_logs).to_csv(os.path.join(out_dir, "normalization_log.csv"), index=False)
    pd.DataFrame(match_warnings).to_csv(os.path.join(out_dir, "match_warnings.csv"), index=False)
    pd.DataFrame(unmatched_orders).to_csv(os.path.join(out_dir, "unmatched_orders.csv"), index=False)
    pd.DataFrame(cert_warnings).to_csv(os.path.join(out_dir, "certification_warnings.csv"), index=False)

    if unmatched_suppliers is not None and not unmatched_suppliers.empty:
        unmatched_suppliers.to_csv(os.path.join(out_dir, "unmatched_suppliers.csv"), index=False)

    with open(os.path.join(out_dir, "summary_report.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
