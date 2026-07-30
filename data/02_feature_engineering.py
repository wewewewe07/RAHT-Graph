"""
02_feature_engineering.py
=========================
Tính toán features cho từng node type và phát hiện market regime.

Output (lưu vào config.PROCESSED_DIR):
  - stocks_proc.csv, sectors_proc.csv, macro_proc.csv (z-scored)
  - market_regime.csv  — 0=Normal, 1=Correction, 2=Crisis
  - stocks_returns_raw.csv — raw 1-day log returns
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings, config

warnings.filterwarnings("ignore")


def rsi(s, w=14):
    d = s.diff()
    g = d.where(d > 0, 0).rolling(w).mean()
    l = (-d.where(d < 0, 0)).rolling(w).mean()
    return 100 - 100 / (1 + g / (l + 1e-10))

def macd_hist(s, fast=12, slow=26, sig=9):
    m = s.ewm(span=fast).mean() - s.ewm(span=slow).mean()
    return m - m.ewm(span=sig).mean()

def rel_vol(v, w=20):
    return v / (v.rolling(w).mean() + 1e-10)


def stock_features(open_df, high_df, low_df, close_df, vol_df):
    """11 technical features: Ret1d, Ret5d, Dir, Gap, Vol20, Range, RSI, MACD, RelVol, Shadow, Hi52"""
    tickers = [t for t in config.TARGET_STOCKS if t in close_df.columns]
    feats   = []
    for t in tickers:
        c = close_df[t]
        o = open_df[t]  if t in open_df.columns  else c
        h = high_df[t]  if t in high_df.columns  else c
        l = low_df[t]   if t in low_df.columns   else c
        v = vol_df[t]   if t in vol_df.columns   else pd.Series(0, index=c.index)
        log_ret  = np.log(c / c.shift(1))
        hl_range = (h - l) / (c + 1e-10)
        upper_s  = (h - pd.concat([o, c], axis=1).max(axis=1)) / (h - l + 1e-10)
        feats.append(pd.DataFrame({
            f"{t}_Ret1d":  log_ret,
            f"{t}_Ret5d":  np.log(c / c.shift(5)),
            f"{t}_Dir":    (c - o) / (o + 1e-10),
            f"{t}_Gap":    (o - c.shift(1)) / (c.shift(1) + 1e-10),
            f"{t}_Vol20":  log_ret.rolling(20).std(),
            f"{t}_Range":  hl_range,
            f"{t}_RSI":    rsi(c),
            f"{t}_MACD":   macd_hist(c),
            f"{t}_RelVol": rel_vol(v),
            f"{t}_Shadow": upper_s,
            f"{t}_Hi52":   c / (c.rolling(252).max() + 1e-10),
        }))
    out = pd.concat(feats, axis=1).fillna(0)
    print(f"  Stock features : {out.shape}  ({len(tickers)} stocks x 11 features)")
    return out


def sector_features(close_df, spy_close=None):
    """6 features per sector: Ret1d, Mom50, Vol20, RSI14, RelSPY, BBpct"""
    if spy_close is None:
        try:
            import yfinance as yf
            spy_close = yf.download("SPY", start=close_df.index[0],
                                    end=close_df.index[-1],
                                    auto_adjust=True, progress=False)["Close"]
            spy_close = spy_close.reindex(close_df.index).ffill().bfill()
        except Exception:
            spy_close = None
    if spy_close is not None:
        if isinstance(spy_close, pd.DataFrame):
            spy_close = spy_close.squeeze()
        spy_ret = np.log(spy_close / spy_close.shift(1)).fillna(0)
        spy_ret = spy_ret.reindex(close_df.index).fillna(0)
        if not isinstance(spy_ret, pd.Series):
            spy_ret = pd.Series(spy_ret, index=close_df.index)
    else:
        spy_ret = pd.Series(0.0, index=close_df.index)
    feats = []
    for t in close_df.columns:
        c       = close_df[t]
        log_ret = np.log(c / c.shift(1))
        mid     = c.rolling(20).mean()
        std20   = c.rolling(20).std()
        feats.append(pd.DataFrame({
            f"{t}_Ret1d":  log_ret,
            f"{t}_Mom50":  c / (c.rolling(50).mean() + 1e-10),
            f"{t}_Vol20":  log_ret.rolling(20).std(),
            f"{t}_RSI14":  rsi(c),
            f"{t}_RelSPY": log_ret.rolling(20).sum() - spy_ret.rolling(20).sum(),
            f"{t}_BBpct":  (c - (mid - 2*std20)) / (4*std20 + 1e-10),
        }))
    out = pd.concat(feats, axis=1).fillna(0)
    print(f"  Sector features: {out.shape}  ({len(close_df.columns)} sectors x 6 features)")
    return out


def macro_features(close_df):
    feats = []
    for t in close_df.columns:
        c = close_df[t]
        v = c if "VIX" in t else np.log(c / c.shift(1)).fillna(0)
        feats.append(pd.DataFrame({f"{t}_Feat": v}))
    out = pd.concat(feats, axis=1).fillna(0)
    print(f"  Macro features : {out.shape}")
    return out


def detect_regime(macro_df, stock_close_df, sector_close_df):
    """
    Regime detection using VIX, max drawdown, and sector correlation cohesion.
    Expanding window ranks to avoid lookahead bias.
    Returns Series with labels: 0=Normal, 1=Correction, 2=Crisis
    """
    vix_col  = next(c for c in macro_df.columns if "VIX" in c)
    vix      = macro_df[vix_col].ffill()
    mkt      = stock_close_df.mean(axis=1)
    peak     = mkt.expanding().max()
    dd       = ((peak - mkt) / peak).fillna(0)
    sec_ret  = np.log(sector_close_df / sector_close_df.shift(1)).fillna(0)
    mkt_ret  = sec_ret.mean(axis=1)
    corr_cols = [sec_ret[c].rolling(30).corr(mkt_ret) for c in sec_ret.columns]
    cohesion  = pd.concat(corr_cols, axis=1).mean(axis=1).fillna(0)
    vix_rank = vix.expanding().apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    dd_rank  = dd.expanding().apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    coh_rank = cohesion.expanding().apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    csi  = (vix_rank + dd_rank + coh_rank) / 3
    labs = pd.qcut(csi, q=3, labels=[0, 1, 2]).astype(int)
    reg  = pd.Series(labs, index=macro_df.index, name="Regime")
    names = ["Normal", "Correction", "Crisis"]
    print(f"\n  {'Regime':<12} {'Days':>5}  {'%':>5}  {'VIX':>6}  {'DD':>7}")
    print(f"  {'-'*42}")
    for i in range(3):
        mask = reg == i; n = mask.sum()
        print(f"  {names[i]:<12} {n:>5}  {n/len(reg)*100:>4.1f}%"
              f"  {vix[mask].mean():>6.1f}  {dd[mask].mean():>6.1%}")
    return reg


def process_data():
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)

    def read(name):
        p  = os.path.join(config.RAW_DIR, f"{name}.csv")
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        return df.apply(pd.to_numeric, errors="coerce").dropna(how="all")

    print("--- Loading raw data ---")
    stocks_o = read("stocks_open")
    stocks_h = read("stocks_high")
    stocks_l = read("stocks_low")
    stocks_c = read("stocks_close")
    stocks_v = read("stocks_volume")
    sectors  = read("sectors")
    macro    = read("macro")

    idx      = stocks_c.index
    stocks_o = stocks_o.reindex(idx).ffill().fillna(0)
    stocks_h = stocks_h.reindex(idx).ffill().fillna(0)
    stocks_l = stocks_l.reindex(idx).ffill().fillna(0)
    stocks_v = stocks_v.reindex(idx).ffill().fillna(0)
    sectors  = sectors.reindex(idx).ffill().bfill()
    macro    = macro.reindex(idx).ffill().bfill()

    avail   = [t for t in config.TARGET_STOCKS if t in stocks_c.columns]
    missing = [t for t in config.TARGET_STOCKS if t not in stocks_c.columns]
    print(f"  Available: {len(avail)}/{len(config.TARGET_STOCKS)}"
          f"  | Missing: {missing or 'none'}")
    print(f"  Date range: {idx[0].date()} to {idx[-1].date()} ({len(idx)} days)")

    print("\n--- Feature engineering ---")
    sf  = stock_features(stocks_o, stocks_h, stocks_l, stocks_c, stocks_v)
    sec = sector_features(sectors)
    mf  = macro_features(macro)

    print("\n--- Regime detection ---")
    reg = detect_regime(macro, stocks_c[avail], sectors)

    print("\n--- Z-score normalisation ---")
    scaler = StandardScaler()

    def save(df, name):
        arr = scaler.fit_transform(df.values)
        out = pd.DataFrame(arr, index=df.index, columns=df.columns)
        out.to_csv(os.path.join(config.PROCESSED_DIR, f"{name}.csv"))
        print(f"  {name}.csv  {out.shape}")

    save(sf,  "stocks_proc")
    save(sec, "sectors_proc")
    save(mf,  "macro_proc")
    reg.to_csv(os.path.join(config.PROCESSED_DIR, "market_regime.csv"))

    ret_cols = [c for c in sf.columns if c.endswith("_Ret1d")]
    sf[ret_cols].to_csv(os.path.join(config.PROCESSED_DIR, "stocks_returns_raw.csv"))

    print("\n[DONE] Feature engineering complete.")


if __name__ == "__main__":
    process_data()
