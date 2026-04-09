"""
Unified Model Architecture for PGMD v3
======================================
Contains:
1. Equivariant EGNN Score Network (Denoiser)
2. Fourier Neural Operator (PINO Refinement)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool


# ── EGNN Core ───────────────────────────────────────────────────────────────

class EGNNLayer(MessagePassing):
    """
    Equivariant Graph Neural Network Layer.
    Matches the architecture of pgmd_v3_full.pt.
    """
    def __init__(self, hidden_dim, edge_dim=4):
        super().__init__(aggr="mean")
        self.phi_e = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.phi_x = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), 
            nn.SiLU(), 
            nn.Linear(hidden_dim // 2, 1)
        )
        self.phi_h = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, h, pos, edge_index, edge_attr):
        return self.propagate(edge_index, x=h, pos=pos, edge_attr=edge_attr)

    def message(self, x_i, x_j, pos_i, pos_j, edge_attr):
        dist = (pos_i - pos_j).norm(dim=-1, keepdim=True)
        return self.phi_e(torch.cat([x_i, x_j, dist, edge_attr], dim=-1))

    def update(self, aggr_out, x):
        return self.phi_h(torch.cat([x, aggr_out], dim=-1))


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim)
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        emb = t[:, None].float() * freqs[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.proj(emb)


class EGNNScoreNetwork(nn.Module):
    """
    Predicts the noise (score) on atomic positions.
    """
    def __init__(self, node_feat_dim, edge_feat_dim=4, hidden_dim=128, num_layers=4):
        super().__init__()
        self.node_enc = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_emb = TimeEmbedding(hidden_dim)
        self.egnn_layers = nn.ModuleList()
        self.film_scale = nn.ModuleList()
        self.film_shift = nn.ModuleList()
        for _ in range(num_layers):
            self.egnn_layers.append(EGNNLayer(hidden_dim, edge_feat_dim))
            self.film_scale.append(nn.Linear(hidden_dim, hidden_dim))
            self.film_shift.append(nn.Linear(hidden_dim, hidden_dim))
            
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3)
        )
        self.prop_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 12) # Matches checkpoint
        )

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
        h_graph = global_mean_pool(h, batch)
        prop_pred = self.prop_head(h_graph)
        return score, prop_pred


# ── PINO Core (FNO) ─────────────────────────────────────────────────────────

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input_tensor, weights):
        return torch.einsum("bix,iox->box", input_tensor, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        N = x.shape[-1]
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(batchsize, self.out_channels, N // 2 + 1, dtype=torch.cfloat, device=x.device)
        modes = min(self.modes, N // 2 + 1)
        out_ft[:, :, :modes] = self.compl_mul1d(x_ft[:, :, :modes], self.weights[:, :, :modes])
        return torch.fft.irfft(out_ft, n=N, dim=-1)


class FourierNeuralOperator(nn.Module):
    def __init__(self, modes=8, width=32, num_layers=4):
        super().__init__()
        self.modes = modes
        self.width = width
        self.num_layers = num_layers
        self.fc_lift = nn.Linear(3, width)
        self.spectral_convs = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(num_layers)])
        self.local_convs = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(num_layers)])
        self.fc_proj1 = nn.Linear(width, width // 2)
        self.fc_proj2 = nn.Linear(width // 2, 3)

    def forward(self, coords):
        # coords: (N, 3)
        x = coords.unsqueeze(0).permute(0, 2, 1) # (1, 3, N)
        x = self.fc_lift(x.transpose(1, 2)).transpose(1, 2)
        
        for i in range(self.num_layers):
            res = x
            x_ft = self.spectral_convs[i](x)
            x_loc = self.local_convs[i](x)
            x = x + x_ft + x_loc
            x = self.norms[i](x.transpose(1, 2)).transpose(1, 2)
            x = F.gelu(x)
            
        x = x.transpose(1, 2)
        x = F.gelu(self.fc_proj1(x))
        delta = self.fc_proj2(x).squeeze(0)
        return coords + delta
