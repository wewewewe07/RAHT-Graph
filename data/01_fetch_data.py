"""
01_fetch_data.py
================
Tải OHLCV đầy đủ cho Stock (Open/High/Low/Close/Volume),
Sector ETFs, và Macro tickers từ Yahoo Finance.

Output (lưu vào config.RAW_DIR):
  - stocks_open.csv, stocks_high.csv, stocks_low.csv,
    stocks_close.csv, stocks_volume.csv
  - sectors.csv
  - macro.csv
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yfinance as yf
import pandas as pd
import config


def _retry_missing(dfs, tickers, start, end):
    """Thử lại từng ticker bị lỗi riêng lẻ."""
    price_types = list(dfs.keys())
    failed = [t for t in tickers
              if t not in dfs["Close"].columns or dfs["Close"][t].isna().all()]
    if not failed:
        return dfs, []

    print(f"  [RETRY] {len(failed)} tickers → retry individually...")
    still_missing = []
    for t in failed:
        try:
            d = yf.download(t, start=start, end=end,
                            auto_adjust=True, progress=False, repair=True)
            if d.empty:
                still_missing.append(t); continue
            for pt in price_types:
                col = pt if pt != "Volume" else "Volume"
                if col in d.columns:
                    series = d[col] if isinstance(d[col], pd.Series) else d[col].squeeze()
                    dfs[pt][t] = series.reindex(dfs["Close"].index)
            print(f"    {t}: recovered {dfs['Close'][t].notna().sum()} days")
        except Exception as e:
            print(f"    {t}: failed — {e}")
            still_missing.append(t)
    return dfs, still_missing


def fetch_stocks():
    print(f"--- Fetching {len(config.TARGET_STOCKS)} STOCKS (OHLCV) ---")
    raw = yf.download(
        config.TARGET_STOCKS,
        start=config.START_DATE, end=config.END_DATE,
        auto_adjust=True, progress=True,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        price_types = ["Open", "High", "Low", "Close", "Volume"]
        dfs = {pt: raw[pt].copy() for pt in price_types if pt in raw.columns}
    else:
        dfs = {"Close": raw[["Close"]].copy(), "Volume": raw[["Volume"]].copy()}
        for pt in ["Open", "High", "Low"]:
            dfs[pt] = dfs["Close"].copy()

    dfs, missing = _retry_missing(dfs, config.TARGET_STOCKS,
                                  config.START_DATE, config.END_DATE)

    if missing:
        print(f"  [WARN] Permanently missing: {missing}")
        with open(os.path.join(config.RAW_DIR, "missing_tickers.txt"), "w") as f:
            f.write("\n".join(missing))

    for pt, df in dfs.items():
        df = df.dropna(axis=1, how="all")
        df.to_csv(os.path.join(config.RAW_DIR, f"stocks_{pt.lower()}.csv"))

    n_days    = dfs["Close"].shape[0]
    n_tickers = dfs["Close"].dropna(axis=1, how="all").shape[1]
    print(f"  Saved: {n_tickers} tickers × {n_days} days")


def fetch_sectors():
    print("--- Fetching SECTOR ETFs ---")
    etfs = list(config.SECTOR_ETFS.values())
    raw  = yf.download(etfs, start=config.START_DATE, end=config.END_DATE,
                       auto_adjust=True, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close.to_csv(os.path.join(config.RAW_DIR, "sectors.csv"))
    print(f"  Saved: {close.shape[1]} sector ETFs")


def fetch_macro():
    print("--- Fetching MACRO tickers ---")
    tickers = list(config.MACRO_TICKERS.values())
    raw     = yf.download(tickers, start=config.START_DATE, end=config.END_DATE,
                          auto_adjust=True, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close.to_csv(os.path.join(config.RAW_DIR, "macro.csv"))
    print(f"  Saved: {close.shape[1]} macro tickers")


def fetch_data():
    os.makedirs(config.RAW_DIR, exist_ok=True)
    fetch_stocks()
    fetch_sectors()
    fetch_macro()
    print("\n[DONE] Raw data saved.")


if __name__ == "__main__":
    fetch_data()
