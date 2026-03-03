import json
import os
import pandas as pd
from typing import Any, Dict, List
from utils import now_iso

def write_outputs(out_dir: str,
                  suppliers: pd.DataFrame,
                  certs: pd.DataFrame,
                  orders: pd.DataFrame,
                  norm_logs: List[Dict[str, Any]],
                  match_warnings: List[Dict[str, Any]],
                  unmatched_orders: List[Dict[str, Any]],
                  unmatched_suppliers: pd.DataFrame,
                  summary: Dict[str, Any]) -> None:
    os.makedirs(out_dir, exist_ok=True)

    suppliers.to_csv(os.path.join(out_dir, "suppliers_clean.csv"), index=False)
    certs.to_csv(os.path.join(out_dir, "certifications_clean.csv"), index=False)
    orders.to_csv(os.path.join(out_dir, "orders_clean.csv"), index=False)

    pd.DataFrame(norm_logs).to_csv(os.path.join(out_dir, "normalization_log.csv"), index=False)
    pd.DataFrame(match_warnings).to_csv(os.path.join(out_dir, "match_warnings.csv"), index=False)
    pd.DataFrame(unmatched_orders).to_csv(os.path.join(out_dir, "unmatched_orders.csv"), index=False)

    if unmatched_suppliers is not None and not unmatched_suppliers.empty:
        unmatched_suppliers.to_csv(os.path.join(out_dir, "unmatched_suppliers.csv"), index=False)

    with open(os.path.join(out_dir, "summary_report.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)