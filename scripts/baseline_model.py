# -*- coding: utf-8 -*-
"""
Baseline EGNN Score Network (Version B)
========================================
Identical backbone to EGNNScoreNetwork but WITHOUT:
  - Gibbs / property prediction head
  - PINO auxiliary loss term
Used purely as a score-matching (denoising) model.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool


class EGNNLayerB(MessagePassing):
    def __init__(self, hidden_dim, edge_dim=4):
        super().__init__(aggr="mean")
        self.phi_e = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1 + edge_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.phi_h = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
        )

    def forward(self, h, pos, edge_index, edge_attr):
        return self.propagate(edge_index, x=h, pos=pos, edge_attr=edge_attr)

    def message(self, x_i, x_j, pos_i, pos_j, edge_attr):
        dist = (pos_i - pos_j).norm(dim=-1, keepdim=True)
        return self.phi_e(torch.cat([x_i, x_j, dist, edge_attr], dim=-1))

    def update(self, aggr_out, x):
        return self.phi_h(torch.cat([x, aggr_out], dim=-1))


class TimeEmbeddingB(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim*2), nn.SiLU(), nn.Linear(dim*2, dim))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half-1))
        emb = t[:, None].float() * freqs[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.proj(emb)


class BaselineScoreNetwork(nn.Module):
    """
    Version B: EGNN score network WITHOUT Gibbs head / PINO loss.
    forward() returns (score, None) so it is API-compatible with Version A.
    """
    def __init__(self, node_feat_dim=11, edge_feat_dim=4, hidden_dim=128, num_layers=4):
        super().__init__()
        self.node_enc = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_emb = TimeEmbeddingB(hidden_dim)
        self.egnn_layers = nn.ModuleList()
        self.film_scale  = nn.ModuleList()
        self.film_shift  = nn.ModuleList()
        for _ in range(num_layers):
            self.egnn_layers.append(EGNNLayerB(hidden_dim, edge_feat_dim))
            self.film_scale.append(nn.Linear(hidden_dim, hidden_dim))
            self.film_shift.append(nn.Linear(hidden_dim, hidden_dim))
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3)
        )
        # NO prop_head / Gibbs head

    def forward(self, x, pos, edge_index, edge_attr, batch, t):
        h = self.node_enc(x)
        t_emb = self.time_emb(t)
        t_emb_per_atom = t_emb[batch]
        for i, layer in enumerate(self.egnn_layers):
            h = layer(h, pos, edge_index, edge_attr)
            scale = self.film_scale[i](t_emb_per_atom)
            shift = self.film_shift[i](t_emb_per_atom)
            h = F.silu(h * scale + shift)
        score = self.score_head(h)
        return score, None   # None for g_pred (no Gibbs head)
