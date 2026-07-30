import torch
import os

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR        = r"/kaggle/working/"
RAW_DIR         = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR   = os.path.join(BASE_DIR, "data", "processed")
DATA_PATH       = os.path.join(BASE_DIR, "data", "hetero_graph.pt")
RAW_CLOSE_PATH  = os.path.join(RAW_DIR,  "stocks_close.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "model", "raht_model.pth")

# ── Time range ────────────────────────────────────────────────────
START_DATE = "2018-01-01"
END_DATE   = "2025-01-01"

# ── Stock universe — expanded to S&P 500 representative (150 stocks) ─
TARGET_STOCKS = [
    # Technology (30)
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO",
    "AMD",  "CRM",  "ADBE", "CSCO",  "INTC", "ORCL", "QCOM", "TXN",
    "IBM",  "NOW",  "INTU", "AMAT",  "MU",   "LRCX", "ADI",  "PANW",
    "KLAC", "SNPS", "CDNS", "APH",   "MSI",  "FTNT",
    # Financials (25)
    "JPM", "BAC", "V",   "MA",   "WFC",  "MS",  "GS",  "BLK",
    "C",   "AXP", "SPGI","CME",  "CB",   "PGR", "MMC", "SCHW",
    "AON", "ICE", "MCO", "USB",  "PNC",  "TFC", "COF", "BK",  "PRU",
    # Healthcare (25)
    "LLY",  "UNH",  "JNJ",  "ABBV", "MRK",  "TMO",  "PFE",  "ABT",
    "AMGN", "ISRG", "SYK",  "DHR",  "MDT",  "VRTX", "REGN", "BSX",
    "CVS",  "ZTS",  "BDX",  "CI",   "MCK",  "HUM",  "BMY",  "GILD",
    "ELV",
    # Consumer Discretionary (15)
    "HD",  "MCD", "NKE", "DIS",  "SBUX", "TGT", "TJX",
    "LOW", "BKNG","CMG", "MAR",  "YUM",  "ORLY","AZO",  "ROST",
    # Consumer Staples (12)
    "WMT", "PG",  "COST","KO",  "PEP",  "PM",
    "MO",  "CL",  "KMB", "SYY", "GIS",  "HSY",
    # Industrials (15)
    "CAT", "GE",  "UPS", "BA",  "HON",  "UNP",
    "RTX", "LMT", "DE",  "ADP", "CSX",  "FDX", "NOC", "NSC", "GD",
    # Energy (10)
    "XOM", "CVX", "COP", "EOG", "SLB",
    "MPC", "PSX", "VLO", "OXY", "HAL",
    # Others: Utilities, Materials, Real Estate, Communication (18)
    "NEE", "DUK", "SO",  "AEP", "EXC",
    "LIN", "SHW", "ECL", "APD", "FCX",
    "PLD", "AMT", "CCI", "PSA", "EQIX",
    "NFLX","T",   "VZ",
]

# ── Sector ETFs (11 sectors) ──────────────────────────────────────
SECTOR_ETFS = {
    "Technology":             "XLK",
    "Financials":             "XLF",
    "Energy":                 "XLE",
    "Health Care":            "XLV",
    "Consumer Discretionary": "XLY",
    "Consumer Staples":       "XLP",
    "Industrials":            "XLI",
    "Utilities":              "XLU",
    "Materials":              "XLB",
    "Real Estate":            "XLRE",
    "Communication":          "XLC",
}

# ── Macro tickers ─────────────────────────────────────────────────
MACRO_TICKERS = {
    "VIX":          "^VIX",
    "Crude_Oil":    "CL=F",
    "10Y_Treasury": "^TNX",
    "Dollar_Index": "DX-Y.NYB",
}

# ── Stock → Sector map ────────────────────────────────────────────
STOCK_SECTOR_MAP = {
    # Technology
    "AAPL":"Technology","MSFT":"Technology","NVDA":"Technology",
    "GOOGL":"Technology","AMZN":"Technology","META":"Technology",
    "TSLA":"Technology","AVGO":"Technology","AMD":"Technology",
    "CRM":"Technology","ADBE":"Technology","CSCO":"Technology",
    "INTC":"Technology","ORCL":"Technology","QCOM":"Technology",
    "TXN":"Technology","IBM":"Technology","NOW":"Technology",
    "INTU":"Technology","AMAT":"Technology","MU":"Technology",
    "LRCX":"Technology","ADI":"Technology","PANW":"Technology",
    "KLAC":"Technology","SNPS":"Technology","CDNS":"Technology",
    "APH":"Technology","MSI":"Technology","FTNT":"Technology",
    # Financials
    "JPM":"Financials","BAC":"Financials","V":"Financials",
    "MA":"Financials","WFC":"Financials","MS":"Financials",
    "GS":"Financials","BLK":"Financials","C":"Financials",
    "AXP":"Financials","SPGI":"Financials","CME":"Financials",
    "CB":"Financials","PGR":"Financials","MMC":"Financials",
    "SCHW":"Financials","AON":"Financials","ICE":"Financials",
    "MCO":"Financials","USB":"Financials","PNC":"Financials",
    "TFC":"Financials","COF":"Financials","BK":"Financials","PRU":"Financials",
    # Healthcare
    "LLY":"Health Care","UNH":"Health Care","JNJ":"Health Care",
    "ABBV":"Health Care","MRK":"Health Care","TMO":"Health Care",
    "PFE":"Health Care","ABT":"Health Care","AMGN":"Health Care",
    "ISRG":"Health Care","SYK":"Health Care","DHR":"Health Care",
    "MDT":"Health Care","VRTX":"Health Care","REGN":"Health Care",
    "BSX":"Health Care","CVS":"Health Care","ZTS":"Health Care",
    "BDX":"Health Care","CI":"Health Care","MCK":"Health Care",
    "HUM":"Health Care","BMY":"Health Care","GILD":"Health Care",
    "ELV":"Health Care",
    # Consumer Discretionary
    "HD":"Consumer Discretionary","MCD":"Consumer Discretionary",
    "NKE":"Consumer Discretionary","DIS":"Consumer Discretionary",
    "SBUX":"Consumer Discretionary","TGT":"Consumer Discretionary",
    "TJX":"Consumer Discretionary","LOW":"Consumer Discretionary",
    "BKNG":"Consumer Discretionary","CMG":"Consumer Discretionary",
    "MAR":"Consumer Discretionary","YUM":"Consumer Discretionary",
    "ORLY":"Consumer Discretionary","AZO":"Consumer Discretionary",
    "ROST":"Consumer Discretionary",
    # Consumer Staples
    "WMT":"Consumer Staples","PG":"Consumer Staples",
    "COST":"Consumer Staples","KO":"Consumer Staples",
    "PEP":"Consumer Staples","PM":"Consumer Staples",
    "MO":"Consumer Staples","CL":"Consumer Staples",
    "KMB":"Consumer Staples","SYY":"Consumer Staples",
    "GIS":"Consumer Staples","HSY":"Consumer Staples",
    # Industrials
    "CAT":"Industrials","GE":"Industrials","UPS":"Industrials",
    "BA":"Industrials","HON":"Industrials","UNP":"Industrials",
    "RTX":"Industrials","LMT":"Industrials","DE":"Industrials",
    "ADP":"Industrials","CSX":"Industrials","FDX":"Industrials",
    "NOC":"Industrials","NSC":"Industrials","GD":"Industrials",
    # Energy
    "XOM":"Energy","CVX":"Energy","COP":"Energy","EOG":"Energy",
    "SLB":"Energy","MPC":"Energy","PSX":"Energy","VLO":"Energy",
    "OXY":"Energy","HAL":"Energy",
    # Utilities
    "NEE":"Utilities","DUK":"Utilities","SO":"Utilities",
    "AEP":"Utilities","EXC":"Utilities",
    # Materials
    "LIN":"Materials","SHW":"Materials","ECL":"Materials",
    "APD":"Materials","FCX":"Materials",
    # Real Estate
    "PLD":"Real Estate","AMT":"Real Estate","CCI":"Real Estate",
    "PSA":"Real Estate","EQIX":"Real Estate",
    # Communication
    "NFLX":"Communication","T":"Communication","VZ":"Communication",
}

# ── Feature dimensions ────────────────────────────────────────────
WINDOW_SIZE     = 60
STOCK_FEATURES  = 11   # 11 technical indicators
SECTOR_FEATURES = 6
MACRO_FEATURES  = 1

# ── Graph params ──────────────────────────────────────────────────
CORR_WINDOW  = 30
CORR_TOP_K   = 10
CORR_MIN_ABS = 0.20

# ── Walk-forward validation ───────────────────────────────────────
# 4 folds, each fold: train grows by 1 year, val=6m-9m, test=6m
# Fold 1: train 2018-2019 | val 2020H1 | test 2020H2
# Fold 2: train 2018-2021Q1 | val 2021 | test 2022H1
# Fold 3: train 2018-2022Q1 | val 2022 | test 2023H1
# Fold 4: train 2018-2023Q1 | val 2023 | test 2024H1
WALK_FORWARD_FOLDS = [
    {"train_end": "2019-12-31", "val_end": "2020-06-30", "test_end": "2020-12-31"},
    {"train_end": "2021-03-31", "val_end": "2021-12-31", "test_end": "2022-06-30"},
    {"train_end": "2022-03-31", "val_end": "2022-12-31", "test_end": "2023-06-30"},
    {"train_end": "2023-03-31", "val_end": "2023-12-31", "test_end": "2024-06-30"},
]
PURGE_DAYS = 5

# ── Encoder ───────────────────────────────────────────────────────
ENCODER = "tft"

# ── Model hyperparams ─────────────────────────────────────────────
HIDDEN_CHANNELS = 64
HEADS           = 2
DROPOUT         = 0.2

# ── Training hyperparams ──────────────────────────────────────────
LEARNING_RATE  = 3.51e-4
WEIGHT_DECAY   = 2.04e-4
EPOCHS         = 400
BATCH_SIZE     = 32      # Optuna best
PREDICT_WINDOW = 5
RANK_WEIGHT    = 1.0
OVERFIT_TEST   = False
RECENCY_HALFLIFE = 0.8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
