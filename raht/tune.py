"""
tune.py
=======
Optuna hyperparameter search for RAHT-Graph model.

Run:  python raht/tune.py
After completion: best params printed and saved to tune_results.json.
Use results to update config.py then run train.py.

Search space:
  - lr, wd: AdamW learning rate and weight decay
  - dropout: regularization strength
  - batch_size: 32 or 64
  - margin: pairwise loss margin
  - pw_w, ln_w: pairwise and listnet loss weights (div_w = 1 - pw_w - ln_w)
  - tau: ListNet temperature
  - target_std: diversity loss target standard deviation
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

import config
from raht.model import RAHT_Graph_Model
from raht.utils import (
    load_graph_data, prepare_edge_sets, build_edge_dict,
    build_forward_returns, compute_metrics
)
import torch.nn.functional as F


def _pairwise_loss(scores, returns, margin):
    scores  = scores.view(-1)
    returns = returns.view(-1)
    si = scores.unsqueeze(1);  sj = scores.unsqueeze(0)
    ri = returns.unsqueeze(1); rj = returns.unsqueeze(0)
    pos_mask = (ri - rj) >  margin
    neg_mask = (ri - rj) < -margin
    diff     = si - sj
    loss_pos = F.softplus(-diff)[pos_mask]
    loss_neg = F.softplus( diff)[neg_mask]
    n_pairs  = pos_mask.sum() + neg_mask.sum()
    if n_pairs == 0:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)
    return (loss_pos.sum() + loss_neg.sum()) / n_pairs


def _listnet_loss(scores, returns, tau):
    scores  = scores.view(-1)
    returns = returns.view(-1)
    N       = scores.shape[0]
    q       = torch.softmax(returns / tau, dim=0)
    log_p   = torch.log_softmax(scores,   dim=0)
    return -(q * log_p).sum() / N


def _diversity_loss(scores, target_std):
    gap = target_std - scores.view(-1).std()
    return torch.clamp(gap, min=0.0) ** 2


def build_loss_fn(margin, pw_w, ln_w, div_w, tau, target_std):
    def loss_fn(scores, returns):
        l_pw  = _pairwise_loss(scores, returns, margin)
        l_ln  = _listnet_loss(scores, returns, tau)
        l_div = _diversity_loss(scores, target_std)
        return pw_w * l_pw + ln_w * l_ln + div_w * l_div
    return loss_fn


def load_data():
    print("Loading data (once for all trials)...")
    data = load_graph_data(config.DATA_PATH, config.DEVICE)
    static_edges, corr_seq = prepare_edge_sets(data, config.DEVICE)

    stock_x    = data["stock"].x.to(config.DEVICE)
    sector_x   = data["sector"].x.to(config.DEVICE)
    macro_x    = data["macro"].x.to(config.DEVICE)
    regime_seq = data["global_regime"].to(config.DEVICE)
    N_stock    = stock_x.shape[0]
    N_sector   = sector_x.shape[0]
    N_macro    = macro_x.shape[0]
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

    train_end = int(T * 0.75)
    val_end   = int(T * 0.90)
    P         = config.PURGE_DAYS
    train_idx = np.arange(config.WINDOW_SIZE,  train_end - config.PREDICT_WINDOW)
    val_idx   = np.arange(train_end + P,        val_end   - config.PREDICT_WINDOW)

    print(f"  T={T} | Train={len(train_idx)} | Val={len(val_idx)}")
    return dict(
        static_edges=static_edges, corr_seq=corr_seq,
        stock_x=stock_x, sector_x=sector_x, macro_x=macro_x,
        regime_seq=regime_seq, fwd_ret=fwd_ret,
        train_idx=train_idx, val_idx=val_idx,
        N_stock=N_stock, N_sector=N_sector, N_macro=N_macro,
    )


N_TUNE_EPOCHS = 60
EVAL_EVERY    = 5


def objective(trial, data):
    lr          = trial.suggest_float("lr",          1e-4, 5e-3, log=True)
    wd          = trial.suggest_float("wd",          1e-5, 1e-2, log=True)
    dropout     = trial.suggest_float("dropout",     0.10, 0.45, step=0.05)
    batch_size  = trial.suggest_categorical("batch", [32, 64])
    margin      = trial.suggest_float("margin",      0.05, 0.20, step=0.05)
    pw_w        = trial.suggest_float("pw_w",        0.40, 0.80, step=0.05)
    ln_w        = trial.suggest_float("ln_w",        0.10, 0.40, step=0.05)
    tau         = trial.suggest_float("tau",         0.20, 1.00, step=0.10)
    target_std  = trial.suggest_float("target_std",  0.10, 0.60, step=0.10)
    div_w       = round(1.0 - pw_w - ln_w, 2)

    if div_w < 0:
        raise optuna.exceptions.TrialPruned()

    model = RAHT_Graph_Model(
        hidden_size  = config.HIDDEN_CHANNELS,
        heads        = config.HEADS,
        dropout      = dropout,
        n_stock      = data["N_stock"],
        n_sector     = data["N_sector"],
        n_macro      = data["N_macro"],
        encoder_type = config.ENCODER,
    ).to(config.DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=1, eta_min=1e-5
    )
    loss_fn = build_loss_fn(margin, pw_w, ln_w, div_w, tau, target_std)

    static_edges = data["static_edges"]
    corr_seq     = data["corr_seq"]
    stock_x      = data["stock_x"]
    sector_x     = data["sector_x"]
    macro_x      = data["macro_x"]
    regime_seq   = data["regime_seq"]
    fwd_ret      = data["fwd_ret"]
    train_idx    = data["train_idx"]
    val_idx      = data["val_idx"]

    best_val = -999.0

    for epoch in range(N_TUNE_EPOCHS):
        model.train()
        batch_t    = np.random.choice(train_idx, size=batch_size, replace=False)
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=config.DEVICE, requires_grad=True)

        for t in batch_t:
            edges  = build_edge_dict(static_edges, corr_seq[t])
            x_dict = {"stock":  stock_x[:, t],
                      "sector": sector_x[:, t],
                      "macro":  macro_x[:, t]}
            rs, _  = model(x_dict, edges, regime_seq[t].item())
            total_loss = total_loss + loss_fn(rs, fwd_ret[:, t])

        (total_loss / batch_size).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(epoch)

        if (epoch + 1) % EVAL_EVERY == 0:
            model.eval()
            cp = tp = 0
            with torch.no_grad():
                for t in val_idx:
                    edges  = build_edge_dict(static_edges, corr_seq[t])
                    x_dict = {"stock":  stock_x[:, t],
                              "sector": sector_x[:, t],
                              "macro":  macro_x[:, t]}
                    rs, _  = model(x_dict, edges, regime_seq[t].item())
                    c, p, _, _, _ = compute_metrics(rs.view(-1), fwd_ret[:, t])
                    cp += c; tp += p
            val_pair = cp / max(tp, 1) * 100

            if val_pair > best_val:
                best_val = val_pair

            trial.report(val_pair, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return best_val


def _progress_callback(study, trial):
    if trial.number % 5 == 0 or trial.state == optuna.trial.TrialState.COMPLETE:
        n_done   = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        best_so_far = study.best_value if study.best_value else 0.0
        status = "PRUNED" if trial.state == optuna.trial.TrialState.PRUNED else \
                 f"{trial.value:.2f}%" if trial.value else "?"
        print(f"  Trial #{trial.number:3d} | {status:>10} | "
              f"Best: {best_so_far:.2f}% | "
              f"Done: {n_done} | Pruned: {n_pruned}")


def main():
    N_TRIALS = 50

    print("=" * 60)
    print(f"OPTUNA HYPERPARAMETER SEARCH -- {N_TRIALS} trials")
    print(f"  Epochs per trial : {N_TUNE_EPOCHS}")
    print(f"  Device           : {config.DEVICE}")
    print("=" * 60 + "\n")

    data = load_data()

    study = optuna.create_study(
        direction  = "maximize",
        sampler    = TPESampler(seed=42),
        pruner     = MedianPruner(
            n_startup_trials  = 10,
            n_warmup_steps    = 20,
            interval_steps    = EVAL_EVERY,
        ),
        study_name = "raht_graph_tuning",
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(
        lambda trial: objective(trial, data),
        n_trials  = N_TRIALS,
        callbacks = [_progress_callback],
    )

    best = study.best_trial
    print("\n" + "=" * 60)
    print("BEST TRIAL")
    print("=" * 60)
    print(f"  Val Pair Acc : {best.value:.2f}%")
    print(f"  Trial number : #{best.number}")
    print("\n  Hyperparameters:")
    for k, v in best.params.items():
        print(f"    {k:<15} = {v}")

    pw_w = best.params["pw_w"]
    ln_w = best.params["ln_w"]
    print(f"    {'div_w':<15} = {round(1-pw_w-ln_w, 2)}  (derived)")

    result = {
        "best_val_pair_acc": best.value,
        "trial_number": best.number,
        "params": best.params,
        "div_w": round(1 - pw_w - ln_w, 2),
        "n_trials_completed": len(study.trials),
        "n_trials_pruned": sum(
            1 for t in study.trials
            if t.state == optuna.trial.TrialState.PRUNED
        ),
    }
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "tune_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved -> {out_path}")


if __name__ == "__main__":
    main()
