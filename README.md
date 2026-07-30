# RAHT-Graph: Regime-Adaptive Hierarchical Temporal Graph Neural Networks for Financial Contagion Modeling and Stock Ranking

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-orange)](https://pytorch.org)
[![PyG](https://img.shields.io/badge/PyG-2.7%2B-green)](https://pyg.org)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

> **Luận văn tốt nghiệp** — Nguyễn Thành Tùng (11226750)  
> Mô hình học sâu trên đồ thị dị thể (Heterogeneous Graph Neural Network) kết hợp Temporal Fusion Transformer để dự đoán và xếp hạng cổ phiếu thích ứng theo trạng thái thị trường.

---

## 📌 Tổng quan

**RAHT-Graph** (Regime-Adaptive **Hierarchical** Temporal Graph Neural Networks) là mô hình GNN dị thể kết hợp 4 thành phần chính:

| Layer | Module | Mô tả |
|-------|--------|-------|
| 1 | **TFT Encoder** | Variable Selection Network → GRU → Multi-head Attention → GRN → Attention Pooling |
| 2 | **HeteroGAT** | Lan truyền thông tin theo cấu trúc phân cấp: Stock ↔ Sector ↔ Macro |
| 3 | **Regime Fusion** | Embedding 3 trạng thái thị trường (Normal / Correction / Crisis) |
| 4 | **Ranking Head** | MLP cho ra alpha score không giới hạn để xếp hạng cổ phiếu |

### Đồ thị dị thể (Heterogeneous Graph)

```
[Macro nodes]  ←→  [Sector nodes]  ←→  [Stock nodes]
    4 nodes           11 nodes          ~149 nodes
 (VIX, Oil, ...)   (XLK, XLF, ...)   (S&P 500 subset)

Static edges:
  stock  → belongs_to → sector  (1 edge per stock)
  sector → depends_on → macro   (full bipartite: 11×4 = 44 edges)

Dynamic edges (per timestep t):
  stock ↔ stock  (rolling Pearson correlation, top-10, |r| ≥ 0.20)
```

### Market Regime Detection

Regime được xác định tự động (không có lookahead bias) dựa trên 3 chỉ số:
- **VIX level** — mức độ sợ hãi thị trường
- **Max Drawdown** — độ sụt giảm từ đỉnh
- **Sector Correlation Cohesion** — mức độ đồng bộ giữa các ngành (systemic risk)

---

## 🗂️ Cấu trúc dự án

```
RAHT-Graph/
├── config.py                   # Toàn bộ hyperparameters & cấu hình
├── run_pipeline.py             # Entry point: chạy toàn bộ pipeline
├── requirements.txt            # Dependencies
│
├── data/                       # Data pipeline scripts
│   ├── 01_fetch_data.py        # Tải OHLCV từ Yahoo Finance (150 S&P 500 stocks)
│   ├── 02_feature_engineering.py  # Tính 11 technical features + regime detection
│   └── 03_build_graph.py       # Xây dựng HeteroData graph (PyG)
│
└── raht/                       # Model package
    ├── layers.py               # GRN, VSN, GRUEncoder, TFTEncoder
    ├── model.py                # RAHT_Graph_Model (4 layers)
    ├── utils.py                # Data loading, loss functions, metrics
    ├── train.py                # Walk-forward validation training
    └── tune.py                 # Optuna hyperparameter search
```

---

## 🚀 Hướng dẫn chạy

### 1. Cài đặt môi trường

```bash
# Tạo virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Cài PyTorch + PyG (theo CUDA version của máy)
pip install torch torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.5.1+cu124.html

# Cài các thư viện còn lại
pip install -r requirements.txt
```

### 2. Chạy toàn bộ pipeline

```bash
# Full pipeline (fetch data -> features -> graph -> train)
python run_pipeline.py

# Bỏ qua bước fetch data (dùng data đã cache)
python run_pipeline.py --skip-data

# Chỉ train (đã có graph.pt)
python run_pipeline.py --train-only

# Hyperparameter tuning với Optuna (50 trials)
python run_pipeline.py --tune
```

### 3. Chạy từng bước

```bash
python data/01_fetch_data.py          # Fetch 150 stocks, 11 sectors, 4 macro
python data/02_feature_engineering.py # Features + regime labels
python data/03_build_graph.py         # Build hetero_graph.pt
python raht/train.py                  # Walk-forward training (4 folds)
python raht/tune.py                   # Optuna search
```

---

## 📊 Dữ liệu

| Loại | Nguồn | Số lượng | Thời gian |
|------|-------|----------|-----------|
| Stock OHLCV | Yahoo Finance | ~149 S&P 500 stocks | 2018–2025 |
| Sector ETFs | Yahoo Finance | 11 sectors (XLK, XLF, ...) | 2018–2025 |
| Macro tickers | Yahoo Finance | VIX, Oil (CL=F), 10Y Treasury, DXY | 2018–2025 |

**Features per node:**
- Stock: 11 technical indicators (Ret1d, Ret5d, Dir, Gap, Vol20, Range, RSI, MACD, RelVol, Shadow, Hi52)
- Sector: 6 indicators (Ret1d, Mom50, Vol20, RSI14, RelSPY, BBpct)
- Macro: 1 (raw level for VIX, log-return for others)

---

## 📈 Đánh giá mô hình

### Metrics

| Metric | Mô tả | Baseline | Good |
|--------|-------|----------|------|
| **Pair Accuracy** | % cặp cổ phiếu được xếp hạng đúng | 50% (random) | >52% |
| **IC** | Spearman rank correlation (score vs return) | 0.000 | >0.05 |
| **ERS** | Expected Return Spread: top 20% vs bottom 20% | 0% | >2% per 5d |
| **Dir Acc** | % cổ phiếu dự đoán đúng chiều | 50% | >52% |

### Walk-Forward Validation (4 folds)

| Fold | Period | Train | Val | Test |
|------|--------|-------|-----|------|
| 1 | 2018–2020 | 2018–2019 | 2020H1 | 2020H2 |
| 2 | 2018–2022 | 2018–2021Q1 | 2021 | 2022H1 |
| 3 | 2018–2023 | 2018–2022Q1 | 2022 | 2023H1 |
| 4 | 2018–2024 | 2018–2023Q1 | 2023 | 2024H1 |

> Purge gap = 5 ngày giữa train/val và val/test để tránh data leakage.

### Kết quả thực nghiệm (Kaggle T4 GPU)

| Fold | Pair Acc | IC | ERS/5d | Ann. ERS |
|------|----------|----|--------|----------|
| 1 | 51.21% | 0.0324 | +9.64% | +10213% |
| 2 | 51.17% | 0.0325 | +3.40% | +438% |
| 3 | 51.32% | 0.0377 | +3.75% | +540% |
| 4 | ~51% | ~0.035 | ~+3% | — |
| **Mean** | **~51.2%** | **~0.034** | **~+5%** | — |

---

## ⚙️ Cấu hình chính (`config.py`)

```python
# Model
ENCODER         = "tft"   # "tft" hoặc "gru"
HIDDEN_CHANNELS = 64
HEADS           = 2
DROPOUT         = 0.2

# Training
LEARNING_RATE   = 3.51e-4
WEIGHT_DECAY    = 2.04e-4
EPOCHS          = 400
BATCH_SIZE      = 32
PREDICT_WINDOW  = 5       # 5-day forward return
RECENCY_HALFLIFE = 0.8    # recent timesteps sampled more

# Graph
CORR_WINDOW     = 30      # rolling correlation window
CORR_TOP_K      = 10      # top-K correlation edges per stock
CORR_MIN_ABS    = 0.20    # minimum |correlation| threshold
WINDOW_SIZE     = 60      # temporal encoder window
```

---

## 📚 Tham khảo

- **TFT**: Lim et al. (2021) — [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363)
- **MASTER**: Wu et al. (2023) — [MASTER: Market-Guided Stock Transformer for Stock Price Forecasting](https://arxiv.org/abs/2312.15235)
- **THGNN**: Xiang et al. (2022) — Temporal and Heterogeneous Graph Neural Network for Financial Time Series Prediction
- **HeteroGAT**: Wang et al. (2019) — [Heterogeneous Graph Attention Network](https://arxiv.org/abs/1903.07293)
- **RankNet**: Burges et al. (2005) — [Learning to Rank using Gradient Descent](https://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf)
- **ListNet**: Cao et al. (2007) — Learning to Rank: From Pairwise Approach to Listwise Approach

---

## 👤 Tác giả

**Nguyễn Thành Tùng** — MSSV: 11226750  
Luận văn tốt nghiệp, 2026  

---

## 📄 License

MIT License — xem file [LICENSE](LICENSE) để biết chi tiết.
