import os
from typing import Any, Dict, List, Tuple

import pandas as pd


DATE_FMT = "%Y-%m-%d"
DATETIME_FMT = "%Y-%m-%d %H:%M:%S"


def _empty_if_na(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _format_date_series(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    return dt.dt.strftime(DATE_FMT).fillna("")


def _format_datetime_value(value: Any) -> str:
    text = _empty_if_na(value)
    if not text:
        return ""
    dt = pd.to_datetime(text, errors="coerce")
    if pd.isna(dt):
        return ""
    return dt.strftime(DATETIME_FMT)


def _sanitize_token(value: Any) -> str:
    text = _empty_if_na(value)
    if not text:
        return ""
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text.upper())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def _ensure_columns(df: pd.DataFrame, columns: List[str], warnings: List[Dict[str, str]], dataset_name: str) -> pd.DataFrame:
    work = df.copy() if df is not None else pd.DataFrame()
    for col in columns:
        if col not in work.columns:
            work[col] = ""
            warnings.append(
                {
                    "dataset": dataset_name,
                    "warning_type": "missing_column",
                    "column": col,
                }
            )
    return work


def map_supplier_tipo(supplier_category: Any) -> str:
    category = _empty_if_na(supplier_category).upper()
    if "PROGETTAZIONE" in category:
        return "PROGETTAZIONE"
    if "BROKER" in category:
        return "BROKER"
    return "PRODUZIONE"


def map_certificazione_tipo(cert_name_normalized: Any) -> str:
    cert_name = _empty_if_na(cert_name_normalized).upper()
    if "ISO 9001" in cert_name:
        return "OPZIONE_1"
    if "ISO 14001" in cert_name:
        return "OPZIONE_2"
    return "ALTRO"


def map_order_stato(order_status: Any, warnings: List[Dict[str, str]]) -> str:
    raw = _empty_if_na(order_status).upper()
    mapping = {
        "EVASO": "EVASO",
        "CONFERMATO": "CONFERMATO",
        "INVIATO": "INVIATO",
        "CONS_PARZ": "CONS_PARZ",
        "CONSEGNA PARZIALE": "CONS_PARZ",
        "CONSEGNATO PARZIALE": "CONS_PARZ",
    }
    if raw in mapping:
        return mapping[raw]
    if raw:
        warnings.append(
            {
                "dataset": "orders",
                "warning_type": "status_unmapped",
                "value": raw,
                "fallback": "INVIATO",
            }
        )
    return "INVIATO" if raw else ""


def build_fornitori_import(df_suppliers: pd.DataFrame, run_ts: str, warnings: List[Dict[str, str]]) -> pd.DataFrame:
    required = ["supplier_id", "supplier_name_normalized", "supplier_category", "supply_scope", "notes"]
    suppliers = _ensure_columns(df_suppliers, required, warnings, "suppliers")

    out = pd.DataFrame()
    out["external_uid"] = suppliers["supplier_id"].apply(_empty_if_na)
    out["ragione_sociale"] = suppliers["supplier_name_normalized"].apply(_empty_if_na)
    out["tipo"] = suppliers["supplier_category"].apply(map_supplier_tipo)
    out["ambito_fornitura"] = suppliers["supply_scope"].apply(_empty_if_na)
    out["stato_approvazione"] = ""
    out["puntualita_consegne_pct"] = ""
    out["kpi_last_update"] = _format_datetime_value(run_ts)
    out["note"] = suppliers["notes"].apply(_empty_if_na)
    out["ultimo_sync"] = _format_datetime_value(run_ts)

    out = out[out["external_uid"] != ""].copy()
    out = out.drop_duplicates(subset=["external_uid"], keep="first")
    out["ragione_sociale"] = out["ragione_sociale"].replace("", "")
    return out


def build_certificazioni_import(
    df_certs: pd.DataFrame,
    run_ts: str,
    warnings: List[Dict[str, str]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required = [
        "certification_id",
        "supplier_id",
        "cert_name_raw",
        "cert_name_normalized",
        "expiry_date",
    ]
    certs = _ensure_columns(df_certs, required, warnings, "certifications")

    certs["expiry_date_fmt"] = _format_date_series(certs["expiry_date"])
    missing_expiry = certs[certs["expiry_date_fmt"] == ""].copy()

    valid = certs[certs["expiry_date_fmt"] != ""].copy()
    valid["tipo_certificazione"] = valid["cert_name_normalized"].apply(map_certificazione_tipo)

    cert_id = valid["certification_id"].apply(_empty_if_na)
    generated = (
        "CERT::"
        + valid["supplier_id"].apply(_sanitize_token)
        + "::"
        + valid["cert_name_normalized"].apply(_sanitize_token)
        + "::"
        + valid["expiry_date_fmt"].apply(_sanitize_token)
    )

    out = pd.DataFrame()
    out["external_uid"] = cert_id.where(cert_id != "", generated)
    out["fornitore_external_uid"] = valid["supplier_id"].apply(_empty_if_na)
    out["tipo_certificazione"] = valid["tipo_certificazione"]
    out["data_scadenza"] = valid["expiry_date_fmt"]
    out["nome_certificazione_altro"] = ""
    out.loc[
        out["tipo_certificazione"] == "ALTRO",
        "nome_certificazione_altro",
    ] = valid["cert_name_raw"].apply(_empty_if_na).where(
        valid["cert_name_raw"].apply(_empty_if_na) != "",
        valid["cert_name_normalized"].apply(_empty_if_na),
    )
    out["codice_certificazione"] = ""
    out["ultimo_sync"] = _format_datetime_value(run_ts)

    out = out[(out["external_uid"] != "") & (out["fornitore_external_uid"] != "")].copy()
    out = out.drop_duplicates(subset=["external_uid"], keep="first")

    missing_expiry_out = missing_expiry.copy()
    return out, missing_expiry_out


def build_ordini_import(
    df_orders: pd.DataFrame,
    run_ts: str,
    warnings: List[Dict[str, str]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required = [
        "order_id",
        "order_number",
        "order_date",
        "matched_supplier_id",
        "operator",
        "amount_total",
        "amount_open",
        "amount_fulfilled",
        "requested_date",
        "confirmed_date",
        "effective_date",
        "job_cdc",
        "job_cdc_is_cdc",
        "status",
    ]
    orders = _ensure_columns(df_orders, required, warnings, "orders")

    matched_mask = orders["matched_supplier_id"].apply(_empty_if_na) != ""
    matched = orders[matched_mask].copy()
    unmatched = orders[~matched_mask].copy()

    out = pd.DataFrame()
    out["external_uid"] = matched["order_id"].apply(_empty_if_na)
    out["numero"] = matched["order_number"].apply(_empty_if_na)
    out["data"] = _format_date_series(matched["order_date"])
    out["fornitore_external_uid"] = matched["matched_supplier_id"].apply(_empty_if_na)
    out["operatore_codice"] = ""
    out["operatore_nome"] = matched["operator"].apply(_empty_if_na)
    out["importo_totale_eur"] = matched["amount_total"].apply(_empty_if_na)
    out["imp_tot_da_evadere_eur"] = matched["amount_open"].apply(_empty_if_na)
    out["imp_tot_evaso_eur"] = matched["amount_fulfilled"].apply(_empty_if_na)
    out["valuta_usd"] = ""
    out["valuta_gbp"] = ""
    out["data_richiesta"] = _format_date_series(matched["requested_date"])
    out["data_conferma"] = _format_date_series(matched["confirmed_date"])
    out["data_effettiva"] = _format_date_series(matched["effective_date"])
    out["tipo_imputazione"] = matched["job_cdc_is_cdc"].apply(lambda x: "CDC" if bool(x) else "JOB")
    out["imputazione"] = matched["job_cdc"].apply(_empty_if_na)
    out["stato"] = matched["status"].apply(lambda x: map_order_stato(x, warnings))
    out["conferma"] = ""
    out["note"] = ""
    out["ultimo_sync"] = _format_datetime_value(run_ts)
    out["name"] = matched["order_number"].apply(_empty_if_na)

    out = out[
        (out["external_uid"] != "")
        & (out["fornitore_external_uid"] != "")
        & (out["data"] != "")
    ].copy()
    out = out.drop_duplicates(subset=["external_uid"], keep="first")

    return out, unmatched


def write_odoo_outputs(
    out_dir: str,
    suppliers: pd.DataFrame,
    certs: pd.DataFrame,
    orders: pd.DataFrame,
    run_ts: str,
) -> Dict[str, int]:
    warnings: List[Dict[str, str]] = []
    odoo_dir = os.path.join(out_dir, "odoo")
    os.makedirs(odoo_dir, exist_ok=True)

    df_fornitori = build_fornitori_import(suppliers, run_ts, warnings)
    df_certificazioni, df_cert_missing_expiry = build_certificazioni_import(certs, run_ts, warnings)
    df_ordini, df_ordini_unmatched = build_ordini_import(orders, run_ts, warnings)

    df_fornitori.to_csv(os.path.join(odoo_dir, "fornitori_import.csv"), index=False)
    df_certificazioni.to_csv(os.path.join(odoo_dir, "certificazioni_import.csv"), index=False)
    df_ordini.to_csv(os.path.join(odoo_dir, "ordini_import.csv"), index=False)
    df_ordini_unmatched.to_csv(os.path.join(odoo_dir, "ordini_import_unmatched.csv"), index=False)
    df_cert_missing_expiry.to_csv(os.path.join(odoo_dir, "certificazioni_missing_expiry.csv"), index=False)
    pd.DataFrame(warnings).to_csv(os.path.join(odoo_dir, "odoo_export_warnings.csv"), index=False)

    non_cdc_dir = os.path.join(out_dir, "non_cdc")
    if "job_cdc_is_cdc" in orders.columns:
        os.makedirs(os.path.join(non_cdc_dir, "odoo"), exist_ok=True)
        non_cdc_orders = orders[orders["job_cdc_is_cdc"] == False].copy()
        df_ordini_non_cdc, _ = build_ordini_import(non_cdc_orders, run_ts, warnings=[])
        df_ordini_non_cdc.to_csv(os.path.join(non_cdc_dir, "odoo", "ordini_import.csv"), index=False)

    return {
        "odoo_fornitori_rows": int(len(df_fornitori)),
        "odoo_certificazioni_rows": int(len(df_certificazioni)),
        "odoo_ordini_rows": int(len(df_ordini)),
        "odoo_ordini_unmatched_rows": int(len(df_ordini_unmatched)),
        "odoo_cert_missing_expiry_rows": int(len(df_cert_missing_expiry)),
    }
