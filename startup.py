"""
Railway startup — generates data if needed, then launches Streamlit.
"""
import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Railway injects PORT automatically — MUST use it
PORT = os.environ.get("PORT", "8501")
print(f"[startup] PORT = {PORT}", flush=True)

def run(cmd: list, label: str):
    print(f"[startup] Running: {label}", flush=True)
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=False)
    if result.returncode != 0:
        print(f"[startup] ERROR in {label} (exit {result.returncode})", flush=True)
        sys.exit(result.returncode)
    print(f"[startup] Done: {label}", flush=True)

# Step 1 — generate raw data
raw = ROOT / "data" / "raw" / "ecommerce_data.csv"
if not raw.exists():
    print("[startup] Generating dataset...", flush=True)
    run([sys.executable, "data/generate_data.py"], "generate_data")
else:
    print(f"[startup] Raw data exists ({raw.stat().st_size // 1024} KB)", flush=True)

# Step 2 — clean data
proc = ROOT / "data" / "processed" / "cleaned_data.csv"
if not proc.exists():
    print("[startup] Cleaning data...", flush=True)
    script = (
        "import sys; sys.path.insert(0,'.');"
        "from src.data_loader import DataLoader;"
        "from src.data_cleaner import DataCleaner;"
        "from pathlib import Path;"
        "df=DataLoader().load_raw_data();"
        "dc=DataCleaner().clean(df);"
        "p=Path('data/processed/cleaned_data.csv');"
        "p.parent.mkdir(parents=True,exist_ok=True);"
        "dc.to_csv(p,index=False);"
        "print('Saved',len(dc),'rows')"
    )
    run([sys.executable, "-c", script], "data_cleaner")
else:
    print(f"[startup] Processed data exists ({proc.stat().st_size // 1024} KB)", flush=True)

# Step 3 — launch Streamlit on Railway's PORT
print(f"[startup] Launching Streamlit on 0.0.0.0:{PORT}", flush=True)
os.execvp(sys.executable, [
    sys.executable, "-m", "streamlit", "run",
    "dashboard/app.py",
    "--server.port",              PORT,
    "--server.address",           "0.0.0.0",
    "--server.headless",          "true",
    "--server.enableCORS",        "false",
    "--server.enableXsrfProtection", "false",
])
