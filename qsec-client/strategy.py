"""
QRT Academy Strategy Runner
============================
Runs mean reversion (and eventually combines with momentum + value),
then formats and uploads targets to QRT.
"""

import pandas as pd
from pathlib import Path
from qsec_client.sample_code import prepare_targets_file, upload_targets_file
from constants import private_key
from mean_reversion import generate_mean_reversion_targets

# ── Credentials & Config ─────────────────────────────────────────────────────

GROUP_ID = "CAM18"
SFTP_USERNAME = "q9968"
SFTP_HOST = "sftp.qrt.cloud"
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Set your LSEG Eikon app key here (or in mean_reversion.py)
EIKON_APP_KEY = None  # e.g. "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"


# ── Generate Targets ─────────────────────────────────────────────────────────

def run():
    all_targets = []

    # Mean reversion — AMER
    amer_targets = generate_mean_reversion_targets("AMER", app_key=EIKON_APP_KEY)
    if not amer_targets.empty:
        all_targets.append(("AMER", amer_targets))

    # Mean reversion — EMEA
    emea_targets = generate_mean_reversion_targets("EMEA", app_key=EIKON_APP_KEY)
    if not emea_targets.empty:
        all_targets.append(("EMEA", emea_targets))

    # ── Upload each region ────────────────────────────────────────────────────

    for region, targets in all_targets:
        print(f"\n--- Uploading {region}: {len(targets)} positions ---")

        csv_path = prepare_targets_file(targets, GROUP_ID, region, OUTPUT_DIR)

        upload_targets_file(
            csv_path,
            region,
            sftp_username=SFTP_USERNAME,
            private_key_path=private_key,
            sftp_host=SFTP_HOST,
        )

        print(f"  ✓ {region} uploaded successfully")

    if not all_targets:
        print("\n⚠ No targets generated for any region. Check data/connection.")


if __name__ == "__main__":
    run()