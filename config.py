from dataclasses import dataclass

@dataclass
class Thresholds:
    low: int = 80
    high: int = 92

ORDER_HEADER_KEYWORDS = {"OPERATORE", "NUMERO", "DATA", "FORNITORE"}
SUPPLIER_HEADER_KEYWORDS = {"RAGIONE SOCIALE"}

ORDER_CANONICAL_MAP = {
    "OPERATORE": "operator",
    "NUMERO": "order_number",
    "DATA": "order_date_raw",
    "FORNITORE": "supplier_text_raw",
    "IMPORTO TOTALE": "amount_total_raw",
    "IMP. TOT. DA EVADERE": "amount_open_raw",
    "IMP. TOT. EVASO": "amount_fulfilled_raw",
    "DATA RICHIESTA": "requested_date_raw",
    "DATA CONFERMAYA": "confirmed_date_raw",
    "DATA EFFETTIVA": "effective_date_raw",
    "JOB / CDC": "job_cdc",
    "JOB/CDC": "job_cdc",
    "JOB CDC": "job_cdc",
    "JOB-CDC": "job_cdc",
    "STATO": "status",
}

SUPPLIER_CANONICAL_MAP = {
    "RAGIONE SOCIALE": "supplier_name_raw",
    "AMBITO DI FORNITURA": "supply_scope",
    "TIPO": "supplier_type",
    "NOTE": "notes",
    "CERTIFICAZIONE": "certifications_raw",
    "CERTIFICAZIONI": "certifications_raw",
    "DATA SCADENZA CERTIFICAZIONE": "cert_expiry_raw",
    "DATA SCADENZA CERT.": "cert_expiry_raw",
    "DATA SCAD. CERTIFICAZIONE": "cert_expiry_raw",
    "SCADENZA CERTIFICAZIONE": "cert_expiry_raw",
    "SCADENZA CERT.": "cert_expiry_raw",
    "DATA SCADENZA": "cert_expiry_raw",
    "SCADENZA": "cert_expiry_raw",
    "KPI\nQUALITA'": "kpi_quality",
    "KPI\nTEMPI": "kpi_time",
    "DISTRIBUTORE UFFICIALE O BROKER": "distributor_or_broker",
}
