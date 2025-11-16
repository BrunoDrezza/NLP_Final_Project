from pathlib import Path

# Pasta raiz do projeto (onde está requirements.txt, Data/, results/, etc.)
BASE_DIR = Path(__file__).resolve().parents[1]

# Data
DATA_DIR = BASE_DIR / "Data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Results
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = RESULTS_DIR / "models"
CSV_DIR = RESULTS_DIR / "csv"
FIGURES_DIR = RESULTS_DIR / "figures"

# Garante que as pastas existem
for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, CSV_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)