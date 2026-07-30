"""
utils.py
========
Utility functions for data loading, loss functions, and metrics.

Sections:
  - Data Loading: load_graph_data, prepare_edge_sets, build_edge_dict
  - Forward Returns: build_forward_returns (rank-normalized)
  - Loss Functions: pairwise_ranking_loss, listnet_loss, diversity_loss, ranking_loss
  - Metrics: compute_metrics (Pair Acc, Dir Acc, IC), compute_ers (ERS)
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════

def load_graph_data(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Graph not found: {path}\n  Run 03_build_graph.py first.")
    data = torch.load(path, weights_only=False)
    print(f"[INFO] Loaded graph from {path}")
    return data


def prepare_edge_sets(data, device):
    fwd_ss   = data["stock",  "belongs_to", "sector"].edge_index.to(device)
    fwd_sm   = data["sector", "depends_on",  "macro"].edge_index.to(device)
    rev_ss   = fwd_ss.flip(0)
    rev_sm   = fwd_sm.flip(0)
    corr_seq = [e.to(device) for e in data["stock", "corr", "stock"].edge_index_seq]
    static   = {
        ("stock",  "belongs_to",     "sector"): fwd_ss,
        ("sector", "depends_on",     "macro"):  fwd_sm,
        ("sector", "rev_belongs_to", "stock"):  rev_ss,
        ("macro",  "rev_depends_on", "sector"): rev_sm,
    }
    return static, corr_seq


def build_edge_dict(static_edges, corr_edge_t):
    d = dict(static_edges)
    d[("stock", "corr", "stock")] = corr_edge_t
    return d


# ══════════════════════════════════════════════════════════════════
#  FORWARD RETURNS
# ══════════════════════════════════════════════════════════════════

def build_forward_returns(raw_close_path, tickers, window_size,
                           predict_window, device):
    """
    Build rank-normalized forward returns tensor [N_stocks, T_steps].
    Rank normalization: converts raw returns to percentile rank in [-1, 1]
    per timestep. This makes the target distribution consistent across
    different market regimes.
    """
    close = pd.read_csv(raw_close_path, index_col=0, parse_dates=True)
    if isinstance(close.columns, pd.MultiIndex):
        close = close.xs("Close", axis=1, level=0)

    avail   = [t for t in tickers if t in close.columns]
    missing = [t for t in tickers if t not in close.columns]
    if missing:
        print(f"[WARN] Forward returns: missing {len(missing)} tickers")
    close = close[avail]

    log_ret  = np.log(close / close.shift(1)).fillna(0).values
    T_raw, N = log_ret.shape
    T_steps  = T_raw - window_size
    raw_tgt  = np.zeros((N, T_steps), dtype=np.float32)

    for t in range(T_steps):
        raw_t         = t + window_size
        raw_tgt[:, t] = log_ret[raw_t : raw_t + predict_window].sum(axis=0)

    normed = np.zeros_like(raw_tgt)
    for t in range(T_steps):
        col          = raw_tgt[:, t]
        rnk          = col.argsort().argsort().astype(np.float32)
        normed[:, t] = 2 * rnk / (N - 1) - 1

    tensor = torch.tensor(normed, dtype=torch.float32)
    print(f"[INFO] Forward returns: {tuple(tensor.shape)}")
    print(f"       Mean={tensor.mean():.4f}  Std={tensor.std():.4f}"
          f"  %Pos={(tensor>0).float().mean()*100:.1f}%")
    return tensor.to(device)


# ══════════════════════════════════════════════════════════════════
#  LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════

MARGIN_PAIR = 0.05
MARGIN_EVAL = 0.05


def pairwise_ranking_loss(scores, returns, sector_ids=None):
    """
    RankNet pairwise cross-entropy loss.
    Intra-sector pairs get 2x weight (same sector -> cleaner signal).
    """
    scores  = scores.view(-1)
    returns = returns.view(-1)

    si = scores.unsqueeze(1);  sj = scores.unsqueeze(0)
    ri = returns.unsqueeze(1); rj = returns.unsqueeze(0)

    pos_mask = (ri - rj) >  MARGIN_PAIR
    neg_mask = (ri - rj) < -MARGIN_PAIR
    diff     = si - sj

    loss_pos = F.softplus(-diff)
    loss_neg = F.softplus( diff)

    if sector_ids is not None:
        same_sector = (sector_ids.unsqueeze(1) == sector_ids.unsqueeze(0)).float()
        weight      = 1.0 + same_sector
        loss_pos    = loss_pos * weight
        loss_neg    = loss_neg * weight

    loss_pos = loss_pos[pos_mask]
    loss_neg = loss_neg[neg_mask]
    n_pairs  = pos_mask.sum() + neg_mask.sum()

    if n_pairs == 0:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)
    return (loss_pos.sum() + loss_neg.sum()) / n_pairs


def listnet_loss(scores, returns):
    """ListNet top-1 probability loss, normalized by N."""
    scores  = scores.view(-1)
    returns = returns.view(-1)
    N       = scores.shape[0]
    tau     = 0.8
    q       = torch.softmax(returns / tau, dim=0)
    log_p   = torch.log_softmax(scores,   dim=0)
    return -(q * log_p).sum() / N


def diversity_loss(scores, target_std=0.2):
    """Prevent score collapse: penalize when std(scores) < target_std."""
    gap = target_std - scores.view(-1).std()
    return torch.clamp(gap, min=0.0) ** 2


def ranking_loss(scores, returns, sector_ids=None):
    """Combined ranking loss: 70% pairwise + 23% listnet + 7% diversity."""
    scores = (scores - scores.mean()) / (scores.std() + 1e-6)
    return (0.70 * pairwise_ranking_loss(scores, returns, sector_ids)
          + 0.23 * listnet_loss(scores, returns)
          + 0.07 * diversity_loss(scores, target_std=0.2)
           )


# ══════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════

def compute_metrics(scores, fut_ret):
    """
    Returns (correct_pairs, total_pairs, correct_dir, total_dir, ic)

    Pair Accuracy: fraction of clearly-ordered pairs ranked correctly.
    Directional Accuracy: fraction of stocks with correct up/down prediction.
    IC (Information Coefficient): Spearman rank correlation between
        predicted scores and actual returns. Range [-1, 1].
        IC=0 -> random, IC=0.05 -> good, IC=0.10 -> excellent.
    """
    N  = len(scores)
    si = scores.unsqueeze(1).expand(N, N)
    sj = scores.unsqueeze(0).expand(N, N)
    ri = fut_ret.unsqueeze(1).expand(N, N)
    rj = fut_ret.unsqueeze(0).expand(N, N)

    triu    = torch.triu(torch.ones(N, N, device=scores.device, dtype=torch.bool), diagonal=1)
    valid   = triu & ((ri - rj).abs() > MARGIN_EVAL)
    correct = ((si - sj) * (ri - rj) > 0) & valid

    cp = correct.sum().item()
    tp = valid.sum().item()

    nz = fut_ret.abs() > 0.05
    cd = (scores[nz].sign() == fut_ret[nz].sign()).sum().item()
    td = nz.sum().item()

    s_np  = scores.detach().cpu().numpy().flatten()
    r_np  = fut_ret.detach().cpu().numpy().flatten()
    s_rank = s_np.argsort().argsort().astype(np.float32)
    r_rank = r_np.argsort().argsort().astype(np.float32)
    s_rank -= s_rank.mean(); r_rank -= r_rank.mean()
    denom = (np.sqrt((s_rank**2).sum()) * np.sqrt((r_rank**2).sum()))
    ic    = float((s_rank * r_rank).sum() / denom) if denom > 1e-8 else 0.0

    return cp, tp, cd, td, ic


def compute_ers(scores, fut_ret, n_quantiles=5):
    """
    Expected Return Spread (ERS):
    Return difference between top and bottom quintile stocks.

    Returns:
      ers          -- float, % per 5-day period
      top_ret      -- float, % mean return of top quintile
      bot_ret      -- float, % mean return of bottom quintile
      annualized   -- float, % annualized ERS (252/5 trading periods)
    """
    scores  = scores.detach().cpu().numpy().flatten()
    fut_ret = fut_ret.detach().cpu().numpy().flatten()
    N       = len(scores)
    q_size  = max(1, N // n_quantiles)

    sorted_idx = scores.argsort()
    top_idx    = sorted_idx[-q_size:]
    bot_idx    = sorted_idx[:q_size]

    top_ret  = float(fut_ret[top_idx].mean() * 100)
    bot_ret  = float(fut_ret[bot_idx].mean() * 100)
    ers      = top_ret - bot_ret
    annualized = (((1 + ers / 100) ** (252 / 5)) - 1) * 100

    return ers, top_ret, bot_ret, annualized
