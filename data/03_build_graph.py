"""
03_build_graph.py
=================
Xây dựng HeteroData graph với:

Static edges:
  stock -> belongs_to -> sector   (N_stock edges, fixed)
  sector -> depends_on -> macro   (N_sector x N_macro edges, full bipartite)

Dynamic edges:
  stock <-> stock (corr)          (E_t edges, rolling Pearson correlation)
    - Rolling 30-day Pearson correlation
    - Keep top-K per stock with |r| >= 0.20
    - E_t increases during crisis (all stocks correlate) and
      decreases in calm markets (stocks diverge by fundamentals)

Output: data/hetero_graph.pt (HeteroData object)
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
import warnings, config

warnings.filterwarnings("ignore")


def sliding_window(df, window):
    """Convert [T_raw, F] DataFrame to [T, W, F] tensor via sliding window."""
    arr  = df.values.astype(np.float32)
    seqs = np.stack([arr[i:i+window] for i in range(len(arr)-window)])
    return torch.tensor(seqs)   # [T, W, F]


def build_stock_sector_edges(available_stocks):
    """Static edges: each stock -> its sector (1 edge per stock)."""
    sector_names = list(config.SECTOR_ETFS.keys())
    N_sec        = len(sector_names)
    src, dst     = [], []
    for i, t in enumerate(available_stocks):
        sec = config.STOCK_SECTOR_MAP.get(t, sector_names[0])
        j   = sector_names.index(sec) if sec in sector_names else 0
        src.append(i); dst.append(max(0, min(j, N_sec-1)))
    return torch.tensor([src, dst], dtype=torch.long)


def build_sector_macro_edges(N_sector, N_macro):
    """Static edges: full bipartite sector -> macro (N_sector x N_macro)."""
    return torch.tensor([
        np.repeat(np.arange(N_sector), N_macro),
        np.tile(  np.arange(N_macro),  N_sector),
    ], dtype=torch.long)


def build_dynamic_edges(returns_df, available_stocks, window, top_k, min_abs):
    """
    Dynamic correlation edges (stock <-> stock) for each timestep t:
      1. Take 30-day window of past returns: ret[t-30:t]
      2. Normalize each column: norm = (x - mean) / std
      3. Compute Pearson correlation: corr = (norm.T @ norm) / (W-1)
      4. For each stock i, keep top-K stocks with |r| >= min_abs

    Returns list of T tensors [2, E_t] where E_t varies per day.
    """
    cols    = [f"{t}_Ret1d" for t in available_stocks
               if f"{t}_Ret1d" in returns_df.columns]
    avail   = [t for t in available_stocks if f"{t}_Ret1d" in returns_df.columns]
    N       = len(avail)
    ret_arr = returns_df[cols].values.astype(np.float32)
    T_steps = ret_arr.shape[0] - window

    print(f"  Building {N} stocks x {T_steps} timesteps "
          f"(top-{top_k}, |r|>={min_abs}) ...")

    src_lists, dst_lists = [], []
    for t in range(window, T_steps + window):
        w    = ret_arr[t - window : t]
        mu   = w.mean(axis=0, keepdims=True)
        std  = w.std(axis=0,  keepdims=True) + 1e-8
        norm = (w - mu) / std
        corr = (norm.T @ norm) / (window - 1)
        np.fill_diagonal(corr, 0.0)

        abs_c = np.abs(corr)
        src_t, dst_t = [], []
        for i in range(N):
            row    = abs_c[i].copy(); row[i] = 0.0
            top_ij = np.argpartition(row, -min(top_k, N-1))[-top_k:]
            for j in top_ij:
                if abs_c[i, j] >= min_abs:
                    src_t.append(i); dst_t.append(j)
        src_lists.append(src_t); dst_lists.append(dst_t)

    edge_tensors = [
        torch.tensor([s, d], dtype=torch.long) if s
        else torch.empty((2, 0), dtype=torch.long)
        for s, d in zip(src_lists, dst_lists)
    ]

    counts = [e.shape[1] for e in edge_tensors]
    print(f"  Edges/step — min:{min(counts)}  mean:{np.mean(counts):.0f}"
          f"  max:{max(counts)}  density:{np.mean(counts)/(N*(N-1))*100:.1f}%")
    return edge_tensors, avail


def build_graph():
    print("=== BUILDING HETEROGENEOUS GRAPH ===\n")
    proc = config.PROCESSED_DIR

    s_proc   = pd.read_csv(os.path.join(proc, "stocks_proc.csv"),        index_col=0, parse_dates=True)
    sec_proc = pd.read_csv(os.path.join(proc, "sectors_proc.csv"),       index_col=0, parse_dates=True)
    m_proc   = pd.read_csv(os.path.join(proc, "macro_proc.csv"),         index_col=0, parse_dates=True)
    s_ret    = pd.read_csv(os.path.join(proc, "stocks_returns_raw.csv"), index_col=0, parse_dates=True)
    regime   = pd.read_csv(os.path.join(proc, "market_regime.csv"),      index_col=0, parse_dates=True)

    avail_stocks = [t for t in config.TARGET_STOCKS
                    if any(c.startswith(f"{t}_") for c in s_proc.columns)]
    N_stock  = len(avail_stocks)
    N_sector = len(config.SECTOR_ETFS)
    N_macro  = len(config.MACRO_TICKERS)
    print(f"  Stocks: {N_stock}/{len(config.TARGET_STOCKS)}"
          f"  |  Sectors: {N_sector}  |  Macro: {N_macro}\n")

    data = HeteroData()

    print("1. Node tensors (sliding window)...")

    def node_tensor(df, tickers):
        parts = []
        for t in tickers:
            cols = [c for c in df.columns if c.startswith(f"{t}_")]
            if not cols:
                raise ValueError(f"No columns for '{t}' — re-run 02.")
            parts.append(sliding_window(df[cols], config.WINDOW_SIZE))
        stacked = torch.stack(parts)   # [N, T, W, F]
        print(f"  {tickers[0]}..{tickers[-1]}: {tuple(stacked.shape)}")
        return stacked

    data["stock"].x  = node_tensor(s_proc,  avail_stocks)
    data["sector"].x = node_tensor(sec_proc, list(config.SECTOR_ETFS.values()))
    data["macro"].x  = node_tensor(m_proc,   list(config.MACRO_TICKERS.values()))

    T = data["stock"].x.shape[1]
    print(f"  Timesteps T = {T}\n")

    reg_vals = regime.values.flatten()[:T].astype(int)
    data["global_regime"] = torch.tensor(reg_vals, dtype=torch.long)

    print("2. Static edges...")
    stock_sec = build_stock_sector_edges(avail_stocks)
    assert stock_sec[0].max() < N_stock and stock_sec[1].max() < N_sector
    data["stock", "belongs_to", "sector"].edge_index = stock_sec
    print(f"  stock->sector: {stock_sec.shape[1]} edges [OK]")

    sec_mac = build_sector_macro_edges(N_sector, N_macro)
    data["sector", "depends_on", "macro"].edge_index = sec_mac
    print(f"  sector->macro: {sec_mac.shape[1]} edges [OK]\n")

    print("3. Dynamic correlation edges (rolling Pearson)...")
    edge_seq, _ = build_dynamic_edges(
        s_ret, avail_stocks,
        window  = config.CORR_WINDOW,
        top_k   = config.CORR_TOP_K,
        min_abs = config.CORR_MIN_ABS,
    )

    if len(edge_seq) > T:
        edge_seq = edge_seq[:T]
    elif len(edge_seq) < T:
        edge_seq = edge_seq + [edge_seq[-1]] * (T - len(edge_seq))

    for s_t in [0, T//2, T-1]:
        ei = edge_seq[s_t]
        if ei.numel() > 0:
            assert ei[0].max() < N_stock and ei[1].max() < N_stock, \
                f"OOB edge at t={s_t}"

    data["stock", "corr", "stock"].edge_index     = edge_seq[0]
    data["stock", "corr", "stock"].edge_index_seq = edge_seq
    print(f"  Dynamic edge sequence: {len(edge_seq)} steps [OK]\n")

    os.makedirs(os.path.dirname(config.DATA_PATH), exist_ok=True)
    torch.save(data, config.DATA_PATH)
    print(f"[DONE] Saved -> {config.DATA_PATH}")
    print(data)


if __name__ == "__main__":
    build_graph()
