"""
layers.py
=========
Shared neural network building blocks for RAHT-Graph model.

Components:
  - GatedResidualNetwork (GRN): Gated residual connection from TFT paper
  - VariableSelectionNetwork (VSN): Learns which features matter most per timestep
  - GRUEncoder: Lightweight temporal encoder (GRU + attention pooling)
  - TFTEncoder: Full Temporal Fusion Transformer encoder (VSN + GRU + MHSA)
  - build_encoder(): Factory function
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════
#  GATED RESIDUAL NETWORK
# ══════════════════════════════════════════════════════════════════

class GatedResidualNetwork(nn.Module):
    """
    GRN from the TFT paper:
      gate = sigmoid(W2 * ELU(W1 * x))
      out  = LayerNorm(gate * W3*h + (1-gate) * skip(x))
    """
    def __init__(self, input_size, hidden_size, output_size=None, dropout=0.1):
        super().__init__()
        self.out_size      = output_size or hidden_size
        self.fc1           = nn.Linear(input_size, hidden_size)
        self.fc2           = nn.Linear(hidden_size, self.out_size)
        self.gate          = nn.Linear(hidden_size, self.out_size)
        self.residual_proj = (nn.Linear(input_size, self.out_size)
                              if input_size != self.out_size else nn.Identity())
        self.dropout       = nn.Dropout(dropout)
        self.norm          = nn.LayerNorm(self.out_size)

    def forward(self, x):
        res  = self.residual_proj(x)
        h    = F.elu(self.fc1(x))
        h    = self.dropout(h)
        out  = self.fc2(h)
        gate = torch.sigmoid(self.gate(h))
        return self.norm(out * gate + res * (1.0 - gate))


# ══════════════════════════════════════════════════════════════════
#  VARIABLE SELECTION NETWORK
# ══════════════════════════════════════════════════════════════════

class VariableSelectionNetwork(nn.Module):
    """
    Learns which input features matter most at each timestep.

    For F input features each of dim 1 (scalar):
      1. Embed each feature independently through its own GRN  -> [B, H]
      2. Compute softmax importance weights over features       -> [B, F]
      3. Weighted sum of embeddings                            -> [B, H]

    Input  : [B, F]  where B = N_nodes * W
    Output : [B, H], weights [B, F]
    """
    def __init__(self, input_feats, hidden_size, dropout=0.1):
        super().__init__()
        self.input_feats = input_feats
        self.hidden_size = hidden_size
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_size, hidden_size, dropout)
            for _ in range(input_feats)
        ])
        self.selection_grn = GatedResidualNetwork(
            input_feats, hidden_size, input_feats, dropout
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        embeddings = []
        for i, grn in enumerate(self.feature_grns):
            embeddings.append(grn(x[:, i:i+1]))   # [B, H]
        embeddings = torch.stack(embeddings, dim=-1)   # [B, H, F]
        weights = self.softmax(self.selection_grn(x))  # [B, F]
        out = (embeddings * weights.unsqueeze(1)).sum(dim=-1)  # [B, H]
        return out, weights


# ══════════════════════════════════════════════════════════════════
#  ENCODER OPTION 1: GRU  (fast baseline)
# ══════════════════════════════════════════════════════════════════

class GRUEncoder(nn.Module):
    """
    Lightweight temporal encoder: Linear projection -> 2-layer GRU -> attention pooling.
    Input  : [N_nodes, Window, input_feats]
    Output : [N_nodes, hidden_size]
    """
    def __init__(self, input_feats, hidden_size, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_feats, hidden_size)
        self.gru  = nn.GRU(hidden_size, hidden_size,
                           num_layers=2, batch_first=True,
                           dropout=dropout if dropout > 0 else 0)
        self.attn = nn.Linear(hidden_size, 1)
        self.norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h      = self.drop(self.proj(x))               # [N, W, H]
        out, _ = self.gru(h)                            # [N, W, H]
        w      = torch.softmax(self.attn(out), dim=1)   # [N, W, 1]
        agg    = (out * w).sum(dim=1)                   # [N, H]
        return self.norm(agg)


# ══════════════════════════════════════════════════════════════════
#  ENCODER OPTION 2: TFT WITH VARIABLE SELECTION
# ══════════════════════════════════════════════════════════════════

class TFTEncoder(nn.Module):
    """
    Full Temporal Fusion Transformer encoder with Variable Selection Network.

    Pipeline per node type:
      1. VSN: learn which features matter         -> [N*W, H]
      2. Reshape back to [N, W, H]
      3. 2-layer GRU over W timesteps
      4. Multi-head self-attention
      5. Add & Norm
      6. Post-attention GRN
      7. Learned attention pooling                -> [N, H]

    Input  : [N_nodes, Window, input_feats]
    Output : [N_nodes, hidden_size]
    """
    def __init__(self, input_feats, hidden_size, dropout=0.1):
        super().__init__()
        self.input_feats = input_feats
        self.hidden_size = hidden_size

        self.vsn  = VariableSelectionNetwork(input_feats, hidden_size, dropout)
        self.gru  = nn.GRU(hidden_size, hidden_size,
                           num_layers=2, batch_first=True,
                           dropout=dropout if dropout > 0 else 0)

        num_heads = next(h for h in [8, 4, 2, 1] if hidden_size % h == 0)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm1     = nn.LayerNorm(hidden_size)
        self.post_grn  = GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)
        self.pool_attn = nn.Linear(hidden_size, 1)
        self.norm2     = nn.LayerNorm(hidden_size)
        self.drop      = nn.Dropout(dropout)

    def forward(self, x):
        N, W, F = x.shape
        x_flat  = x.reshape(N * W, F)
        x_sel, _ = self.vsn(x_flat)
        x_sel   = x_sel.reshape(N, W, self.hidden_size)

        gru_out, _ = self.gru(x_sel)
        attn_out, _ = self.attn(gru_out, gru_out, gru_out)
        x_mid  = self.norm1(gru_out + attn_out)

        x_mid_flat = x_mid.reshape(N * W, self.hidden_size)
        x_post     = self.post_grn(x_mid_flat).reshape(N, W, self.hidden_size)

        pool_w = torch.softmax(self.pool_attn(x_post), dim=1)
        agg    = (x_post * pool_w).sum(dim=1)
        return self.norm2(agg)


# ══════════════════════════════════════════════════════════════════
#  FACTORY
# ══════════════════════════════════════════════════════════════════

def build_encoder(encoder_type: str, input_feats: int,
                  hidden_size: int, dropout: float):
    """
    encoder_type : "gru" or "tft"
    The rest of the model (GNN, fusion, heads) is identical either way.
    """
    if encoder_type == "tft":
        return TFTEncoder(input_feats, hidden_size, dropout)
    elif encoder_type == "gru":
        return GRUEncoder(input_feats, hidden_size, dropout)
    else:
        raise ValueError(f"Unknown encoder '{encoder_type}'. Choose 'gru' or 'tft'.")
