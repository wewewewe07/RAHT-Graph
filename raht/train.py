"""
train.py
========
Walk-forward validation training for RAHT-Graph.

Strategy:
  - 4 folds, each growing training window by ~1 year
  - Purge gap (5 days) between train/val and val/test to prevent leakage
  - Recency-weighted sampling: recent timesteps sampled ~3x more
  - Early stopping on validation IC with patience=50 epochs
  - Overfit guard: stop if train-val pair accuracy gap > 5%

Metrics reported per fold:
  - Pair Accuracy: % of correctly ordered pairs (random=50%)
  - IC: Spearman rank correlation (good: 0.05, excellent: 0.10)
  - Directional Accuracy: % correct up/down predictions
  - ERS: Expected Return Spread between top/bottom quintile (annualized)
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import json
import pandas as pd
import config
from raht.model import RAHT_Graph_Model
from raht.utils import (
    load_graph_data, prepare_edge_sets, build_edge_dict,
    build_forward_returns, ranking_loss, compute_metrics, compute_ers
)


def overfit_test(model, stock_x, sector_x, macro_x,
                 regime_seq, static_edges, corr_seq, fwd_ret):
    print("\n" + "="*60)
    print("OVERFIT TEST — 20 fixed timesteps, expect >85% pair acc")
    print("="*60)
    fixed = list(range(100, 120))
    opt   = torch.optim.Adam(model.parameters(), lr=5e-3)

    for step in range(400):
        model.train()
        opt.zero_grad()
        loss = torch.tensor(0.0, device=config.DEVICE, requires_grad=True)
        for t in fixed:
            edges  = build_edge_dict(static_edges, corr_seq[t])
            x_dict = {"stock":  stock_x[:, t],
                      "sector": sector_x[:, t],
                      "macro":  macro_x[:, t]}
            rs, _  = model(x_dict, edges, regime_seq[t].item())
            loss   = loss + ranking_loss(rs, fwd_ret[:, t])
        (loss / len(fixed)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % 100 == 0:
            model.eval()
            cp = tp = 0
            with torch.no_grad():
                for t in fixed:
                    edges  = build_edge_dict(static_edges, corr_seq[t])
                    x_dict = {"stock":  stock_x[:, t],
                              "sector": sector_x[:, t],
                              "macro":  macro_x[:, t]}
                    rs, _  = model(x_dict, edges, regime_seq[t].item())
                    c, p, _, _, _ = compute_metrics(rs.view(-1), fwd_ret[:, t])
                    cp += c; tp += p
            print(f"  Step {step+1:3d}: Pair Acc={cp/max(tp,1)*100:.1f}%"
                  f"  Score Std={rs.std():.4f}")
    print()


def train_one_fold(fold_num, train_idx, val_idx, test_idx,
                   stock_x, sector_x, macro_x, regime_seq,
                   static_edges, corr_seq, fwd_ret,
                   N_stock, N_sector, N_macro, sector_ids=None):
    """Train model for one fold, return test metrics dict."""
    print(f"\n{'='*60}")
    print(f"FOLD {fold_num} | Train: {len(train_idx)} | "
          f"Val: {len(val_idx)} | Test: {len(test_idx)}")
    print(f"{'='*60}")

    model = RAHT_Graph_Model(
        hidden_size  = config.HIDDEN_CHANNELS,
        heads        = config.HEADS,
        dropout      = config.DROPOUT,
        n_stock      = N_stock,
        n_sector     = N_sector,
        n_macro      = N_macro,
        encoder_type = config.ENCODER,
    ).to(config.DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=1, eta_min=1e-5
    )

    best_val  = -999.0
    patience  = 0
    PATIENCE  = 50
    best_path = config.MODEL_SAVE_PATH.replace(".pth", f"_fold{fold_num}_best.pth")
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)

    _w = np.exp((train_idx - train_idx[0]) /
                (len(train_idx) * config.RECENCY_HALFLIFE))
    recency_weights = (_w / _w.sum()).astype(np.float64)

    for epoch in range(config.EPOCHS):
        model.train()
        batch_t    = np.random.choice(train_idx, size=config.BATCH_SIZE,
                                       replace=False, p=recency_weights)
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=config.DEVICE, requires_grad=True)

        for t in batch_t:
            edges  = build_edge_dict(static_edges, corr_seq[t])
            x_dict = {"stock":  stock_x[:, t],
                      "sector": sector_x[:, t],
                      "macro":  macro_x[:, t]}
            rank_s, _ = model(x_dict, edges, regime_seq[t].item())
            l_rank     = ranking_loss(rank_s, fwd_ret[:, t], sector_ids)
            total_loss = total_loss + config.RANK_WEIGHT * l_rank

        avg_loss = total_loss / config.BATCH_SIZE
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(epoch)

        if (epoch + 1) % 5 == 0:
            model.eval()

            def evaluate(indices):
                cp = tp = cd = td = 0
                stds = []; ics = []
                with torch.no_grad():
                    for t in indices:
                        edges  = build_edge_dict(static_edges, corr_seq[t])
                        x_dict = {"stock":  stock_x[:, t],
                                  "sector": sector_x[:, t],
                                  "macro":  macro_x[:, t]}
                        rs, _  = model(x_dict, edges, regime_seq[t].item())
                        c, p, cd_, td_, ic_ = compute_metrics(rs.view(-1), fwd_ret[:, t])
                        cp += c; tp += p; cd += cd_; td += td_
                        stds.append(rs.std().item())
                        ics.append(ic_)
                return (cp/max(tp,1)*100), (cd/max(td,1)*100), float(np.mean(stds)), float(np.mean(ics))

            tr_p, tr_d, _, _      = evaluate(np.random.choice(
                train_idx, min(150, len(train_idx)), replace=False))
            va_p, va_d, vss, vic = evaluate(val_idx)
            lr              = optimizer.param_groups[0]["lr"]

            print(f"  Epoch {epoch+1:03d} | Loss: {avg_loss.item():.4f} | "
                  f"Train: {tr_p:.1f}% | Val: {va_p:.2f}% | "
                  f"IC: {vic:.4f} | Std: {vss:.4f} | LR: {lr:.6f}")

            if vic > best_val:
                best_val = vic
                patience = 0
                torch.save(model.state_dict(), best_path)
                print(f"     ✓ Best (Val IC: {best_val:.4f} | Pair: {va_p:.2f}%)")
            else:
                patience += 5
                if patience >= PATIENCE:
                    print(f"  [EARLY STOP] No improvement for {PATIENCE} epochs.")
                    break

            if tr_p - va_p > 5.0:
                print(f"  [OVERFIT GUARD] Gap {tr_p-va_p:.1f}% > 5%. Stopping.")
                break

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, weights_only=True))
    model.eval()

    cp = tp = cd = td = 0
    ics = []; top_rets = []; bot_rets = []

    with torch.no_grad():
        for t in test_idx:
            edges  = build_edge_dict(static_edges, corr_seq[t])
            x_dict = {"stock":  stock_x[:, t],
                      "sector": sector_x[:, t],
                      "macro":  macro_x[:, t]}
            rs, _  = model(x_dict, edges, regime_seq[t].item())
            c, p, cd_, td_, ic_ = compute_metrics(rs.view(-1), fwd_ret[:, t])
            cp += c; tp += p; cd += cd_; td += td_; ics.append(ic_)
            ers_, top_, bot_, _ = compute_ers(rs.view(-1), fwd_ret[:, t])
            top_rets.append(top_)
            bot_rets.append(bot_)

    test_pair = cp / max(tp, 1) * 100
    test_dir  = cd / max(td, 1) * 100
    test_ic   = float(np.mean(ics))
    mean_top  = float(np.mean(top_rets))
    mean_bot  = float(np.mean(bot_rets))
    test_ers  = mean_top - mean_bot
    test_ers_ann = (((1 + test_ers / 100) ** (252 / 5)) - 1) * 100

    print(f"\n  Fold {fold_num} Test | Pair: {test_pair:.2f}% | "
          f"IC: {test_ic:.4f} | Dir: {test_dir:.2f}% | "
          f"ERS: {test_ers:+.3f}% per 5d "
          f"(Ann: {test_ers_ann:+.1f}%) | "
          f"Top Q: {mean_top:+.3f}% | Bot Q: {mean_bot:+.3f}%")

    return {
        "fold":         fold_num,
        "best_ic":      round(best_val,    4),
        "test_pair":    round(test_pair,   4),
        "test_dir":     round(test_dir,    4),
        "test_ic":      round(test_ic,     4),
        "test_ers":     round(test_ers,    4),
        "test_ers_ann": round(test_ers_ann,2),
        "top_quintile": round(mean_top,    4),
        "bot_quintile": round(mean_bot,    4),
    }


def date_to_idx(date_str, regime_index):
    target = pd.Timestamp(date_str)
    arr = regime_index.get_indexer([target], method="ffill")
    return max(0, int(arr[0]))


def main():
    print(f"--- WALK-FORWARD TRAINING ON {config.DEVICE} ---\n")

    data = load_graph_data(config.DATA_PATH, config.DEVICE)
    static_edges, corr_seq = prepare_edge_sets(data, config.DEVICE)

    N_stock  = data["stock"].x.shape[0]
    N_sector = data["sector"].x.shape[0]
    N_macro  = data["macro"].x.shape[0]

    avail_stocks = [t for t in config.TARGET_STOCKS
                    if t in config.STOCK_SECTOR_MAP]
    sector_names = list(config.SECTOR_ETFS.keys())
    _sector_ids  = []
    for t in avail_stocks[:N_stock]:
        sec = config.STOCK_SECTOR_MAP.get(t, sector_names[0])
        j   = sector_names.index(sec) if sec in sector_names else 0
        _sector_ids.append(j)
    sector_ids = torch.tensor(_sector_ids, dtype=torch.long, device=config.DEVICE)

    stock_x    = data["stock"].x.to(config.DEVICE)
    sector_x   = data["sector"].x.to(config.DEVICE)
    macro_x    = data["macro"].x.to(config.DEVICE)
    regime_seq = data["global_regime"].to(config.DEVICE)
    T          = stock_x.shape[1]

    fwd_ret = build_forward_returns(
        config.RAW_CLOSE_PATH, config.TARGET_STOCKS,
        config.WINDOW_SIZE, config.PREDICT_WINDOW, config.DEVICE
    )

    T = min(T, fwd_ret.shape[1])
    stock_x    = stock_x[:, :T]
    sector_x   = sector_x[:, :T]
    macro_x    = macro_x[:, :T]
    regime_seq = regime_seq[:T]
    fwd_ret    = fwd_ret[:, :T]
    corr_seq   = corr_seq[:T]

    N_fr = fwd_ret.shape[0]
    if stock_x.shape[0] > N_fr:
        stock_x = stock_x[:N_fr]
        N_stock = N_fr

    regime_df    = pd.read_csv(
        os.path.join(config.PROCESSED_DIR, "market_regime.csv"),
        index_col=0, parse_dates=True
    )
    regime_index = regime_df.index[:T]

    if config.OVERFIT_TEST:
        test_model = RAHT_Graph_Model(
            hidden_size=config.HIDDEN_CHANNELS, heads=config.HEADS,
            dropout=config.DROPOUT, n_stock=N_stock,
            n_sector=N_sector, n_macro=N_macro,
            encoder_type=config.ENCODER,
        ).to(config.DEVICE)
        overfit_test(test_model, stock_x, sector_x, macro_x,
                     regime_seq, static_edges, corr_seq, fwd_ret)
        del test_model

    fold_results = []
    P = config.PURGE_DAYS

    for fold_num, fold in enumerate(config.WALK_FORWARD_FOLDS, start=1):
        train_end_t = date_to_idx(fold["train_end"], regime_index)
        val_end_t   = date_to_idx(fold["val_end"],   regime_index)
        test_end_t  = date_to_idx(fold["test_end"],  regime_index)
        test_end_t  = min(test_end_t, T - config.PREDICT_WINDOW)

        train_idx = np.arange(config.WINDOW_SIZE,
                               train_end_t - config.PREDICT_WINDOW)
        val_idx   = np.arange(train_end_t + P,
                               val_end_t   - config.PREDICT_WINDOW)
        test_idx  = np.arange(val_end_t + P, test_end_t)

        if len(train_idx) < 100 or len(val_idx) < 20 or len(test_idx) < 20:
            print(f"\n[SKIP] Fold {fold_num}: insufficient data")
            continue

        result = train_one_fold(
            fold_num, train_idx, val_idx, test_idx,
            stock_x, sector_x, macro_x, regime_seq,
            static_edges, corr_seq, fwd_ret,
            N_stock, N_sector, N_macro, sector_ids
        )
        fold_results.append(result)

    print("\n" + "="*60)
    print("WALK-FORWARD RESULTS")
    print("="*60)

    if fold_results:
        avg_test_pair = np.mean([r["test_pair"]    for r in fold_results])
        avg_test_ic   = np.mean([r.get("test_ic",0)for r in fold_results])
        avg_test_ers  = np.mean([r.get("test_ers",0)for r in fold_results])
        avg_ers_ann   = np.mean([r.get("test_ers_ann",0) for r in fold_results])
        avg_top_q     = np.mean([r.get("top_quintile",0) for r in fold_results])
        avg_bot_q     = np.mean([r.get("bot_quintile",0) for r in fold_results])

        print(f"  {'Fold':<6} {'Test Pair':>10} {'Test IC':>8} {'ERS/5d':>8} {'Ann ERS':>9}")
        print(f"  {'-'*50}")
        for r in fold_results:
            print(f"  Fold {r['fold']}  "
                  f"{r['test_pair']:>9.2f}%  "
                  f"{r.get('test_ic',0):>7.4f}  "
                  f"{r.get('test_ers',0):>+7.3f}%  "
                  f"{r.get('test_ers_ann',0):>+8.1f}%")
        print(f"  {'-'*50}")
        print(f"  Mean    "
              f"{avg_test_pair:>9.2f}%  "
              f"{avg_test_ic:>7.4f}  "
              f"{avg_test_ers:>+7.3f}%  "
              f"{avg_ers_ann:>+8.1f}%")
        print(f"\n  Random baseline = 50.00% pair | IC=0.000")
        print(f"  RAHT-Graph mean = {avg_test_pair:.2f}% pair | IC={avg_test_ic:.4f}")

        out = {
            "folds":            fold_results,
            "avg_test_pair":    round(avg_test_pair, 4),
            "avg_test_ic":      round(avg_test_ic,   4),
            "avg_test_ers":     round(avg_test_ers,  4),
            "avg_ers_ann":      round(avg_ers_ann,   2),
            "avg_top_quintile": round(avg_top_q,     4),
            "avg_bot_quintile": round(avg_bot_q,     4),
        }
        result_path = config.MODEL_SAVE_PATH.replace(".pth", "_wf_results.json")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Results saved -> {result_path}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
