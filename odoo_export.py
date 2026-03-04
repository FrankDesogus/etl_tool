import os
import re
import hashlib
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
    cert_name = _normalize_cert_for_detection(cert_name_normalized)
    if re.search(r"\bISO\s*9001\b", cert_name):
        return "OPZIONE_1"
    # Regola stretta: OPZIONE_2 deve riconoscere solo EN 9100.
    if re.search(r"\bEN\s*9100\b", cert_name):
        return "OPZIONE_2"
    return "ALTRO"


def _normalize_cert_for_detection(cert_name: Any) -> str:
    text = _empty_if_na(cert_name).upper()
    if not text:
        return ""
    text = re.sub(r"[:\-_/]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _split_certification_tokens(cert_cell_raw: Any, cert_name_raw: Any) -> List[str]:
    if pd.isna(cert_cell_raw):
        cert_cell_raw = ""
    source = _empty_if_na(cert_cell_raw) or _empty_if_na(cert_name_raw)
    if not source:
        return []
    raw_tokens = [tok.strip() for tok in re.split(r"[;,\n/]+", source)]
    if not raw_tokens:
        raw_tokens = [source.strip()]

    ignored_tokens = {"", "nan", "none", "null", "-", "n/a"}
    cleaned_tokens: List[str] = []
    for token in raw_tokens:
        if not token:
            continue
        normalized = re.sub(r"\s+", " ", token).strip().lower()
        if normalized in ignored_tokens:
            continue
        cleaned_tokens.append(token)

    return cleaned_tokens


def _extract_expiry_dates(expiry_date_raw: Any, expiry_date_fallback: Any) -> Tuple[List[str], str]:
    raw = _empty_if_na(expiry_date_raw)
    fallback = _empty_if_na(expiry_date_fallback)

    source = raw or fallback
    if not source:
        return [], "missing"

    source = source.replace("\r", "\n")
    date_like = re.findall(r"\d{1,4}[./-]\d{1,2}[./-]\d{2,4}", source)
    candidates = date_like if date_like else [source]

    parsed: List[str] = []
    for token in candidates:
        dt = pd.to_datetime(token.strip(), errors="coerce", dayfirst=True)
        if pd.isna(dt):
            return [], "unparseable"
        parsed.append(dt.strftime(DATE_FMT))

    return parsed, "ok"


def _build_cert_external_uid(supplier_uid: str, idx: int, token: str, expiry: str) -> str:
    fingerprint = hashlib.sha1(f"{token}|{expiry}".encode("utf-8")).hexdigest()[:12].upper()
    return f"CERT::{_sanitize_token(supplier_uid)}::{idx}::{fingerprint}"


def _extract_multi_cert_info(cert_cell_raw: Any, cert_name_raw: Any) -> Tuple[str, str]:
    tokens = _split_certification_tokens(cert_cell_raw, cert_name_raw)
    if not tokens:
        return "", ""

    detected = []
    altro_tokens = []
    for token in tokens:
        cert_type = map_certificazione_tipo(token)
        if cert_type not in detected:
            detected.append(cert_type)
        if cert_type == "ALTRO" and token not in altro_tokens:
            altro_tokens.append(token)

    ordered = [tipo for tipo in ["OPZIONE_1", "OPZIONE_2", "ALTRO"] if tipo in detected]
    multi = ",".join(ordered)
    altro_dettaglio = "; ".join(altro_tokens)
    return multi, altro_dettaglio


def _select_single_cert_type(multi_types: str) -> str:
    # Priorità fissa per compatibilità col campo selection single-choice in Odoo.
    for preferred in ["OPZIONE_2", "OPZIONE_1", "ALTRO"]:
        if preferred in multi_types.split(","):
            return preferred
    return ""


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
    required = [
        "supplier_external_uid",
        "supplier_name_raw",
        "registry_supplier_category",
        "registry_supply_scope",
        "registry_notes",
        "matched_registry_supplier_id",
    ]
    suppliers = _ensure_columns(df_suppliers, required, warnings, "suppliers")
    unmatched_mask = suppliers["matched_registry_supplier_id"].apply(_empty_if_na) == ""

    out = pd.DataFrame()
    out["external_uid"] = suppliers["supplier_external_uid"].apply(_empty_if_na)
    out["ragione_sociale"] = suppliers["supplier_name_raw"].apply(_empty_if_na)
    out["tipo"] = suppliers["registry_supplier_category"].apply(map_supplier_tipo)
    out.loc[unmatched_mask, "tipo"] = "PRODUZIONE"
    out["ambito_fornitura"] = suppliers["registry_supply_scope"].apply(_empty_if_na)
    out["stato_approvazione"] = ""
    out.loc[unmatched_mask, "stato_approvazione"] = "IN VALUTAZIONE"
    out["puntualita_consegne_pct"] = ""
    out["kpi_last_update"] = _format_datetime_value(run_ts)
    out["note"] = suppliers["registry_notes"].apply(_empty_if_na)
    out.loc[unmatched_mask, "note"] = "Creato da ORDINI: non presente in registro fornitori"
    out["ultimo_sync"] = _format_datetime_value(run_ts)

    out = out[out["external_uid"] != ""].copy()
    out = out.drop_duplicates(subset=["external_uid"], keep="first")
    out["ragione_sociale"] = out["ragione_sociale"].replace("", "")
    return out


def build_certificazioni_import(
    df_certs: pd.DataFrame,
    registry_to_master_mapping: pd.DataFrame,
    run_ts: str,
    warnings: List[Dict[str, str]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required = [
        "certification_id",
        "supplier_id",
        "supplier_name_normalized",
        "cert_cell_raw",
        "cert_name_raw",
        "cert_name_normalized",
        "expiry_date_raw",
        "expiry_date",
        "source_file",
        "source_sheet",
        "source_row",
        "source_column",
    ]
    certs = _ensure_columns(df_certs, required, warnings, "certifications")
    mapping = _ensure_columns(
        registry_to_master_mapping,
        ["registry_supplier_id", "supplier_external_uid"],
        warnings,
        "registry_to_master_mapping",
    )

    certs = certs.merge(
        mapping.drop_duplicates(subset=["registry_supplier_id"], keep="first"),
        how="left",
        left_on="supplier_id",
        right_on="registry_supplier_id",
    )

    orphan_count = int(certs[certs["supplier_external_uid"].fillna("").astype(str).str.strip() == ""].shape[0])
    if orphan_count:
        warnings.append(
            {
                "dataset": "certifications",
                "warning_type": "orphan_registry_cert",
                "value": str(orphan_count),
                "fallback": "skipped_from_export",
            }
        )

    export_rows: List[Dict[str, str]] = []
    missing_expiry_rows: List[Dict[str, str]] = []

    for _, row in certs.iterrows():
        supplier_uid = _empty_if_na(row.get("supplier_external_uid", ""))
        if not supplier_uid:
            continue

        cert_tokens = _split_certification_tokens(row.get("cert_cell_raw", ""), row.get("cert_name_raw", ""))
        if not cert_tokens:
            cert_tokens = [_empty_if_na(row.get("cert_name_raw", "")) or _empty_if_na(row.get("cert_name_normalized", ""))]
        cert_tokens = [tok for tok in cert_tokens if _empty_if_na(tok)]
        if not cert_tokens:
            continue

        expiry_dates, expiry_state = _extract_expiry_dates(row.get("expiry_date_raw", ""), row.get("expiry_date", ""))
        if expiry_state != "ok":
            missing_expiry_rows.append(
                {
                    "certification_id": _empty_if_na(row.get("certification_id", "")),
                    "supplier_id": _empty_if_na(row.get("supplier_id", "")),
                    "supplier_external_uid": supplier_uid,
                    "supplier_name_normalized": _empty_if_na(row.get("supplier_name_normalized", "")),
                    "cert_cell_raw": _empty_if_na(row.get("cert_cell_raw", "")),
                    "cert_name_raw": _empty_if_na(row.get("cert_name_raw", "")),
                    "expiry_date_raw": _empty_if_na(row.get("expiry_date_raw", "")),
                    "expiry_date": _empty_if_na(row.get("expiry_date", "")),
                    "diagnostic": f"expiry_{expiry_state}",
                    "source_file": _empty_if_na(row.get("source_file", "")),
                    "source_sheet": _empty_if_na(row.get("source_sheet", "")),
                    "source_row": _empty_if_na(row.get("source_row", "")),
                    "source_column": _empty_if_na(row.get("source_column", "")),
                }
            )
            warnings.append(
                {
                    "dataset": "certifications",
                    "warning_type": "expiry_unparseable",
                    "supplier_id": _empty_if_na(row.get("supplier_id", "")),
                    "supplier_external_uid": supplier_uid,
                    "cert_cell_raw": _empty_if_na(row.get("cert_cell_raw", "")),
                    "expiry_date_raw": _empty_if_na(row.get("expiry_date_raw", "")),
                    "source_file": _empty_if_na(row.get("source_file", "")),
                    "source_sheet": _empty_if_na(row.get("source_sheet", "")),
                    "source_row": _empty_if_na(row.get("source_row", "")),
                }
            )
            continue

        aligned_dates: List[str]
        if len(expiry_dates) == len(cert_tokens):
            aligned_dates = expiry_dates
        elif len(expiry_dates) == 1:
            aligned_dates = expiry_dates * len(cert_tokens)
        else:
            aligned_dates = []
            for idx in range(len(cert_tokens)):
                if idx < len(expiry_dates):
                    aligned_dates.append(expiry_dates[idx])
                else:
                    aligned_dates.append(expiry_dates[-1])
            warnings.append(
                {
                    "dataset": "certifications",
                    "warning_type": "expiry_dates_count_mismatch",
                    "supplier_id": _empty_if_na(row.get("supplier_id", "")),
                    "supplier_external_uid": supplier_uid,
                    "tokens_count": str(len(cert_tokens)),
                    "dates_count": str(len(expiry_dates)),
                    "resolution": "reuse_last_date",
                    "cert_cell_raw": _empty_if_na(row.get("cert_cell_raw", "")),
                    "expiry_date_raw": _empty_if_na(row.get("expiry_date_raw", "")),
                    "source_file": _empty_if_na(row.get("source_file", "")),
                    "source_sheet": _empty_if_na(row.get("source_sheet", "")),
                    "source_row": _empty_if_na(row.get("source_row", "")),
                }
            )

        base_cert_id = _empty_if_na(row.get("certification_id", ""))
        for idx, token in enumerate(cert_tokens):
            token_clean = _empty_if_na(token)
            expiry_fmt = aligned_dates[idx]
            cert_type = map_certificazione_tipo(token_clean)
            if base_cert_id and len(cert_tokens) == 1:
                external_uid = base_cert_id
            else:
                external_uid = _build_cert_external_uid(supplier_uid, idx + 1, token_clean, expiry_fmt)

            export_rows.append(
                {
                    "external_uid": external_uid,
                    "fornitore_external_uid": supplier_uid,
                    "tipo_certificazione": cert_type,
                    "certificazioni_altro_dettaglio": token_clean if cert_type == "ALTRO" else "",
                    "data_scadenza": expiry_fmt,
                    "codice_certificazione": "",
                    "ultimo_sync": _format_datetime_value(run_ts),
                }
            )

    out = pd.DataFrame(export_rows)
    if out.empty:
        out = pd.DataFrame(
            columns=[
                "external_uid",
                "fornitore_external_uid",
                "tipo_certificazione",
                "certificazioni_altro_dettaglio",
                "data_scadenza",
                "codice_certificazione",
                "ultimo_sync",
            ]
        )
    out = out[(out["external_uid"] != "") & (out["fornitore_external_uid"] != "")].copy()
    out = out.drop_duplicates(subset=["external_uid"], keep="first")

    missing_expiry_out = pd.DataFrame(missing_expiry_rows)
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
        "supplier_external_uid",
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

    matched_mask = orders["supplier_external_uid"].apply(_empty_if_na) != ""
    matched = orders[matched_mask].copy()
    unmatched = orders[~matched_mask].copy()

    out = pd.DataFrame()
    out["external_uid"] = matched["order_id"].apply(_empty_if_na)
    out["numero"] = matched["order_number"].apply(_empty_if_na)
    out["data"] = _format_date_series(matched["order_date"])
    out["fornitore_external_uid"] = matched["supplier_external_uid"].apply(_empty_if_na)
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
    registry_to_master_mapping: pd.DataFrame,
    run_ts: str,
) -> Dict[str, int]:
    warnings: List[Dict[str, str]] = []
    odoo_dir = os.path.join(out_dir, "odoo")
    os.makedirs(odoo_dir, exist_ok=True)

    df_fornitori = build_fornitori_import(suppliers, run_ts, warnings)
    df_certificazioni, df_cert_missing_expiry = build_certificazioni_import(
        certs,
        registry_to_master_mapping,
        run_ts,
        warnings,
    )
    df_ordini, df_ordini_unmatched = build_ordini_import(orders, run_ts, warnings)

    df_fornitori.to_csv(os.path.join(odoo_dir, "fornitori_import.csv"), index=False)
    df_certificazioni.to_csv(os.path.join(odoo_dir, "certificazioni_import.csv"), index=False)
    df_ordini.to_csv(os.path.join(odoo_dir, "ordini_import.csv"), index=False)
    df_ordini_unmatched.to_csv(os.path.join(odoo_dir, "ordini_import_unmatched.csv"), index=False)
    df_cert_missing_expiry.to_csv(os.path.join(odoo_dir, "certificazioni_missing_expiry.csv"), index=False)
    pd.DataFrame(warnings).to_csv(os.path.join(odoo_dir, "odoo_export_warnings.csv"), index=False)

    return {
        "odoo_fornitori_rows": int(len(df_fornitori)),
        "odoo_certificazioni_rows": int(len(df_certificazioni)),
        "odoo_ordini_rows": int(len(df_ordini)),
        "odoo_ordini_unmatched_rows": int(len(df_ordini_unmatched)),
        "odoo_cert_missing_expiry_rows": int(len(df_cert_missing_expiry)),
    }
