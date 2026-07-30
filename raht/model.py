"""
model.py
========
RAHT-Graph: Regime-Adaptive Hierarchical Temporal Graph Neural Network.

Architecture (4 layers):
  1. TFT Encoder: VSN -> GRU -> MHSA -> GRN -> Attention Pooling
     - 3 independent encoders for Stock / Sector / Macro node types
     - Input: [N_nodes, W=60, F] -> Output: [N_nodes, H]

  2. HeteroGAT Propagation:
     - Information flows upward: stock -> sector -> macro
     - And backward: macro -> sector -> stock
     - Dynamic stock-stock correlation edges (change every timestep)

  3. Regime Fusion:
     - Embedding lookup for regime 0/1/2 -> [H] context vector
     - Concatenate with GNN output -> GRN -> final representation

  4. Ranking Head:
     - MLP (Linear -> GELU -> Linear)
     - Output: unbounded alpha score per stock
     - Only relative ordering matters, not absolute values
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv
from raht.layers import build_encoder, GatedResidualNetwork


class RAHT_Graph_Model(nn.Module):
    """
    RAHT-Graph: Regime-Adaptive Hierarchical Temporal Graph Neural Network
    for financial contagion modeling and stock ranking.
    """

    def __init__(self, hidden_size=64, heads=2, dropout=0.2,
                 n_stock=149, n_sector=11, n_macro=4,
                 encoder_type="tft"):
        super().__init__()
        self.encoder_type = encoder_type

        # Layer 1: TFT Encoders (independent per node type)
        self.enc_stock  = build_encoder(encoder_type, _config_feats("stock"),  hidden_size, dropout)
        self.enc_sector = build_encoder(encoder_type, _config_feats("sector"), hidden_size, dropout)
        self.enc_macro  = build_encoder(encoder_type, _config_feats("macro"),  hidden_size, dropout)

        # Layer 2: HeteroGAT (hierarchical message passing)
        self.gnn = HeteroConv({
            ("stock",  "belongs_to",     "sector"): GATConv(
                hidden_size, hidden_size, heads=heads,
                dropout=dropout, concat=False, add_self_loops=False),
            ("sector", "depends_on",     "macro"):  GATConv(
                hidden_size, hidden_size, heads=heads,
                dropout=dropout, concat=False, add_self_loops=False),
            # Reverse edges: macro/sector info flows back to stock
            ("sector", "rev_belongs_to", "stock"):  GATConv(
                hidden_size, hidden_size, heads=heads,
                dropout=dropout, concat=False, add_self_loops=False),
            ("macro",  "rev_depends_on", "sector"): GATConv(
                hidden_size, hidden_size, heads=heads,
                dropout=dropout, concat=False, add_self_loops=False),
            # Dynamic peer-to-peer: correlation edges change every day
            ("stock",  "corr",           "stock"):  GATConv(
                hidden_size, hidden_size, heads=heads,
                dropout=dropout, concat=False, add_self_loops=True),
        }, aggr="sum")

        self.gnn_norm = nn.LayerNorm(hidden_size)

        # Layer 3: Regime Fusion
        self.regime_embed = nn.Embedding(3, hidden_size)
        self.fusion_grn   = GatedResidualNetwork(
            hidden_size * 2, hidden_size, dropout=dropout
        )

        # Layer 4: Ranking Head (no LayerNorm at output to preserve score spread)
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )
        nn.init.normal_(self.score_mlp[-1].weight, std=0.1)
        nn.init.zeros_(self.score_mlp[-1].bias)

    def forward(self, x_dict, edge_index_t, regime_idx):
        """
        Args:
            x_dict       : {"stock":  [N_s,   W, 11],
                            "sector": [N_sec, W, 6],
                            "macro":  [N_mac, W, 1]}
            edge_index_t : edge dict from build_edge_dict() for timestep t
            regime_idx   : int -- 0=Normal, 1=Correction, 2=Crisis

        Returns:
            scores : [N_s, 1] -- Alpha score per stock (unbounded)
        """
        # Layer 1: TFT Encoding
        h_stock  = self.enc_stock(x_dict["stock"])    # [N_s, H]
        h_sector = self.enc_sector(x_dict["sector"])  # [N_sec, H]
        h_macro  = self.enc_macro(x_dict["macro"])    # [N_mac, H]

        # Layer 2: Hierarchical Graph Propagation
        gnn_out = self.gnn(
            {"stock": h_stock, "sector": h_sector, "macro": h_macro},
            edge_index_t
        )
        h_gnn   = gnn_out.get("stock", torch.zeros_like(h_stock))
        h_fused = self.gnn_norm(h_gnn + h_stock)

        # Layer 3: Regime Fusion
        r_emb   = self.regime_embed(
            torch.tensor(regime_idx, device=h_fused.device)
        ).unsqueeze(0).expand(h_fused.shape[0], -1)   # [N_s, H]
        h_final = self.fusion_grn(torch.cat([h_fused, r_emb], dim=-1))

        # Layer 4: Ranking Head
        scores = self.score_mlp(h_final)               # [N_s, 1]
        return scores, torch.zeros_like(scores)


def _config_feats(node_type):
    """Return number of features for given node type from config."""
    import config as cfg
    return {"stock": cfg.STOCK_FEATURES,
            "sector": cfg.SECTOR_FEATURES,
            "macro": cfg.MACRO_FEATURES}[node_type]
