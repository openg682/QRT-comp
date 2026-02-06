import pandas as pd
from pathlib import Path
from qsec_client import prepare_targets_file, upload_targets_file

# ============================================
# YOUR STRATEGY LOGIC GOES HERE
# ============================================
# - Fetch/load market data
# - Generate signals
# - Decide which stocks to go long/short
# - Size positions (respecting the $2M cap, 2.5% ADV, $500k risk limit)
# ============================================

# Your strategy must produce a DataFrame like this:
targets = pd.DataFrame({
    "internal_code": ["AAPL.O", "MSFT.O", "TSLA.O"],  # RIC codes
    "currency": ["USD", "USD", "USD"],
    "target_notional": [500000, -300000, 200000],  # positive = long, negative = short
})

# === PIPELINE (don't change much here) ===
GROUP_ID = "CAM18"
REGION = "AMER"  # or "EMEA"
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Format the CSV
csv_path = prepare_targets_file(targets, GROUP_ID, REGION, OUTPUT_DIR)

# 2. Upload to QRT
upload_targets_file(
    csv_path,
    REGION,
    sftp_username="q9968",
    private_key_path=Path("/Users/om/Downloads/cam18_id_rsa"),
    sftp_host="sftp.qrt.cloud",
)