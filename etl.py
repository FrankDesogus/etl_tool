import argparse
import json
import os

import openpyxl
import pandas as pd

from config import (
    Thresholds,
    ORDER_HEADER_KEYWORDS,
    SUPPLIER_HEADER_KEYWORDS,
    ORDER_CANONICAL_MAP,
    SUPPLIER_CANONICAL_MAP,
)
from io_excel import sheet_is_orders, extract_table_with_provenance, canonicalize_columns
from suppliers import build_suppliers_clean, deduplicate_suppliers
from certifications import split_certifications
from orders import clean_and_match_orders
from report import write_outputs, build_orders_supplier_cert_report
from utils import now_iso


def _write_non_cdc_pack(
    *,
    out_dir: str,
    xlsx_path: str,
    df_sup_dedup: pd.DataFrame,
    df_certs: pd.DataFrame,
    df_orders_clean: pd.DataFrame,
    df_orders_supplier_cert_report: pd.DataFrame,
    norm_logs: list,
    match_warnings: list,
    unmatched_orders: list,
) -> None:
    """
    Produce the same reports but only for orders where JOB/CDC does NOT start with 'CDC'.
    Writes to: <out_dir>/non_cdc/
    """
    non_cdc_dir = os.path.join(out_dir, "non_cdc")
    os.makedirs(non_cdc_dir, exist_ok=True)

    if "job_cdc_is_cdc" not in df_orders_clean.columns:
        raise RuntimeError(
            "Missing column 'job_cdc_is_cdc' in orders_clean. "
            "Update orders.clean_and_match_orders() to add CDC classification columns."
        )

    # Filter orders (NON-CDC)
    df_orders_non_cdc = df_orders_clean[df_orders_clean["job_cdc_is_cdc"] == False].copy()
    non_cdc_order_ids = set(df_orders_non_cdc["order_id"].astype(str))
    df_orders_supplier_cert_report_non_cdc = df_orders_supplier_cert_report[
        df_orders_supplier_cert_report["order_id"].astype(str).isin(non_cdc_order_ids)
    ].copy()

    # Filter match_warnings for those orders
    df_match_warnings_all = pd.DataFrame(match_warnings)
    if not df_match_warnings_all.empty and "order_id" in df_match_warnings_all.columns:
        df_match_warnings_non_cdc = df_match_warnings_all[
            df_match_warnings_all["order_id"].astype(str).isin(non_cdc_order_ids)
        ].copy()
    else:
        df_match_warnings_non_cdc = pd.DataFrame()

    # Filter unmatched_orders for those orders
    df_unmatched_orders_all = pd.DataFrame(unmatched_orders)
    if not df_unmatched_orders_all.empty and "order_id" in df_unmatched_orders_all.columns:
        df_unmatched_orders_non_cdc = df_unmatched_orders_all[
            df_unmatched_orders_all["order_id"].astype(str).isin(non_cdc_order_ids)
        ].copy()
    else:
        df_unmatched_orders_non_cdc = pd.DataFrame()

    # Filter normalization log to transformations for those orders
    df_norm_all = pd.DataFrame(norm_logs)
    if not df_norm_all.empty and {"entity", "record_id"}.issubset(df_norm_all.columns):
        df_norm_non_cdc = df_norm_all[
            (df_norm_all["entity"] == "order")
            & (df_norm_all["record_id"].astype(str).isin(non_cdc_order_ids))
        ].copy()
    else:
        df_norm_non_cdc = pd.DataFrame()

    # Write NON-CDC outputs (same filenames, separate folder)
    df_orders_non_cdc.to_csv(os.path.join(non_cdc_dir, "orders_clean.csv"), index=False)
    df_orders_supplier_cert_report_non_cdc.to_csv(
        os.path.join(non_cdc_dir, "orders_supplier_cert_report.csv"), index=False
    )
    df_match_warnings_non_cdc.to_csv(os.path.join(non_cdc_dir, "match_warnings.csv"), index=False)
    df_unmatched_orders_non_cdc.to_csv(os.path.join(non_cdc_dir, "unmatched_orders.csv"), index=False)
    df_norm_non_cdc.to_csv(os.path.join(non_cdc_dir, "normalization_log.csv"), index=False)

    # Optional: include suppliers/certs so the folder is a complete “pack”
    df_sup_dedup.to_csv(os.path.join(non_cdc_dir, "suppliers_clean.csv"), index=False)
    df_certs.to_csv(os.path.join(non_cdc_dir, "certifications_clean.csv"), index=False)

    summary_non_cdc = {
        "source_file": os.path.basename(xlsx_path),
        "scope": "orders where JOB/CDC does NOT start with CDC",
        "run_finished_at": now_iso(),
        "counts": {
            "orders_rows_clean": int(len(df_orders_non_cdc)),
            "orders_supplier_cert_report_rows": int(len(df_orders_supplier_cert_report_non_cdc)),
            "match_warnings": int(len(df_match_warnings_non_cdc)),
            "unmatched_orders": int(len(df_unmatched_orders_non_cdc)),
            "normalization_ops_orders": int(len(df_norm_non_cdc)),
        },
    }
    with open(os.path.join(non_cdc_dir, "summary_report.json"), "w", encoding="utf-8") as f:
        json.dump(summary_non_cdc, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_xlsx")
    ap.add_argument("--out", default="./output")
    ap.add_argument("--low", type=int, default=80)
    ap.add_argument("--high", type=int, default=92)

    # NEW: enable/disable the split pack from CLI
    ap.add_argument(
        "--split-non-cdc",
        action="store_true",
        help="Also produce a second report pack under <out>/non_cdc with only orders whose JOB/CDC does NOT start with 'CDC'.",
    )

    args = ap.parse_args()

    xlsx_path = os.path.abspath(args.input_xlsx)
    out_dir = os.path.abspath(args.out)
    thresholds = Thresholds(low=args.low, high=args.high)

    norm_logs = []
    match_warnings = []
    unmatched_orders = []
    unmatched_suppliers_parts = []
    suppliers_parts = []
    orders_parts = []
    supplier_sheet_diagnostics = []

    os.makedirs(out_dir, exist_ok=True)

    wb = openpyxl.load_workbook(xlsx_path, data_only=True)

    for sh in wb.sheetnames:
        df_preview = pd.read_excel(
            xlsx_path, sheet_name=sh, header=None, engine="openpyxl", nrows=40
        )

        if sheet_is_orders(df_preview, ORDER_HEADER_KEYWORDS):
            df_raw, _ = extract_table_with_provenance(
                xlsx_path, sh, "orders", ORDER_HEADER_KEYWORDS, SUPPLIER_HEADER_KEYWORDS
            )
            if df_raw.empty:
                continue
            df_can = canonicalize_columns(df_raw, ORDER_CANONICAL_MAP)
            orders_parts.append(df_can)
        else:
            df_raw, header_idx = extract_table_with_provenance(
                xlsx_path, sh, "suppliers", ORDER_HEADER_KEYWORDS, SUPPLIER_HEADER_KEYWORDS
            )
            if df_raw.empty:
                continue
            header_norm = [str(c) for c in df_raw.columns if str(c) not in {"source_file", "source_sheet", "source_row"}]
            mapped = []
            for c in header_norm:
                tgt = SUPPLIER_CANONICAL_MAP.get(c)
                if tgt:
                    mapped.append(f"{c}->{tgt}")
            supplier_sheet_diagnostics.append({
                "source_file": os.path.basename(xlsx_path),
                "source_sheet": sh,
                "header_row_index_0_based": header_idx,
                "header_row_excel_1_based": header_idx + 1,
                "raw_headers_normalized": json.dumps(header_norm, ensure_ascii=False),
                "mapped_columns": json.dumps(mapped, ensure_ascii=False),
                "expiry_field_mapped": any(m.endswith("->cert_expiry_raw") for m in mapped),
            })
            df_can = canonicalize_columns(df_raw, SUPPLIER_CANONICAL_MAP)
            df_clean, df_bad = build_suppliers_clean(
                df_can, supplier_category=sh, norm_logs=norm_logs
            )
            suppliers_parts.append(df_clean)
            if not df_bad.empty:
                unmatched_suppliers_parts.append(df_bad)

    df_sup_all = (
        pd.concat(suppliers_parts, ignore_index=True)
        if suppliers_parts
        else pd.DataFrame()
    )
    df_orders_raw = (
        pd.concat(orders_parts, ignore_index=True) if orders_parts else pd.DataFrame()
    )
    df_unmatched_sup = (
        pd.concat(unmatched_suppliers_parts, ignore_index=True)
        if unmatched_suppliers_parts
        else pd.DataFrame()
    )

    df_sup_dedup = deduplicate_suppliers(
        df_sup_all, match_warnings, low_threshold=thresholds.low
    )
    df_certs, cert_warnings, cert_tokens_dropped_nan_like = split_certifications(df_sup_dedup)
    df_orders_clean = clean_and_match_orders(
        df_orders_raw,
        df_sup_dedup,
        thresholds.low,
        thresholds.high,
        norm_logs,
        match_warnings,
        unmatched_orders,
    )
    df_orders_supplier_cert_report = build_orders_supplier_cert_report(
        orders=df_orders_clean,
        suppliers=df_sup_dedup,
        certs=df_certs,
    )

    summary = {
        "source_file": os.path.basename(xlsx_path),
        "run_started_at": now_iso(),
        "run_finished_at": now_iso(),
        "thresholds": {"low": thresholds.low, "high": thresholds.high},
        "counts": {
            "sheets_total": len(wb.sheetnames),
            "orders_rows_read": int(len(df_orders_raw)),
            "orders_rows_clean": int(len(df_orders_clean)),
            "orders_supplier_cert_report_rows": int(len(df_orders_supplier_cert_report)),
            "suppliers_rows_read": int(len(df_sup_all)),
            "suppliers_rows_clean": int(len(df_sup_dedup)),
            "suppliers_dedup_merged": int(max(0, len(df_sup_all) - len(df_sup_dedup))),
            "certifications_rows": int(len(df_certs)),
            "suppliers_rows_with_expiry_raw_nonempty": int((df_sup_dedup.get("cert_expiry_raw", pd.Series(dtype=str)).fillna("").astype(str).str.strip() != "").sum()),
            "cert_rows_with_expiry_parsed": int((df_certs.get("expiry_date", pd.Series(dtype=str)).fillna("").astype(str).str.strip() != "").sum()),
            "cert_rows_with_expiry_unparseable": int(sum(1 for w in cert_warnings if w.get("warning_type") == "cert_expiry_unparseable")),
            "cert_rows_with_expiry_missing": int(sum(1 for w in cert_warnings if w.get("warning_type") == "cert_expiry_missing")),
            "cert_tokens_dropped_nan_like": int(cert_tokens_dropped_nan_like),
            "normalization_ops": int(len(norm_logs)),
            "match_warnings": int(len(match_warnings)),
            "unmatched_orders": int(len(unmatched_orders)),
            "unmatched_suppliers": int(len(df_unmatched_sup)) if not df_unmatched_sup.empty else 0,
        },
    }

    # Main (full) outputs
    write_outputs(
        out_dir=out_dir,
        suppliers=df_sup_dedup,
        certs=df_certs,
        orders=df_orders_clean,
        orders_supplier_cert_report=df_orders_supplier_cert_report,
        norm_logs=norm_logs,
        match_warnings=match_warnings,
        unmatched_orders=unmatched_orders,
        unmatched_suppliers=df_unmatched_sup,
        cert_warnings=cert_warnings,
        summary=summary,
    )

    pd.DataFrame(supplier_sheet_diagnostics).to_csv(
        os.path.join(out_dir, "supplier_header_diagnostics.csv"), index=False
    )

    # Optional split pack: NON-CDC
    if args.split_non_cdc:
        _write_non_cdc_pack(
            out_dir=out_dir,
            xlsx_path=xlsx_path,
            df_sup_dedup=df_sup_dedup,
            df_certs=df_certs,
            df_orders_clean=df_orders_clean,
            df_orders_supplier_cert_report=df_orders_supplier_cert_report,
            norm_logs=norm_logs,
            match_warnings=match_warnings,
            unmatched_orders=unmatched_orders,
        )

    print(f"Done. Outputs in: {out_dir}")
    if args.split_non_cdc:
        print(f"NON-CDC pack in: {os.path.join(out_dir, 'non_cdc')}")


if __name__ == "__main__":
    main()
