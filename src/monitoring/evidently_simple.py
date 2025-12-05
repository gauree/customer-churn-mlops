# src/monitoring/evidently_simple.py
import os
from datetime import datetime

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def generate_data_drift_report(
    reference_path: str = "data/processed/train.csv",
    current_path: str = "data/processed/test.csv",
    output_dir: str = "monitoring/reports",
) -> str:
    """
    Generate a simple Evidently data drift report comparing reference vs current data.

    - reference_path: CSV used for training (e.g. train.csv)
    - current_path:   latest batch / test data (e.g. test.csv or a production extract)

    Returns the path to the generated HTML report.
    """
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference data not found at {reference_path}")
    if not os.path.exists(current_path):
        raise FileNotFoundError(f"Current data not found at {current_path}")

    ref_df = pd.read_csv(reference_path)
    cur_df = pd.read_csv(current_path)

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"evidently_data_drift_{ts}.html")

    # Simple report: only data drift preset
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)
    report.save_html(out_path)

    return out_path


if __name__ == "__main__":
    # default: compare train vs test
    report_path = generate_data_drift_report()
    print(f"Evidently data drift report generated at: {report_path}")
    print("Open this HTML file in a browser to view the dashboard.")
