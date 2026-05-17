"""
=============================================================================
  Physics-Guided Molecular Diffusion (PGMD) Framework for Catalyst Design
=============================================================================
  Author  : Senior Research Engineer, SciML & Cheminformatics
  Dataset : QM9 (auto-downloaded via torch_geometric)
  Goal    : Learn a score-based diffusion model over molecular graphs,
            guided by thermodynamic stability (Gibbs Free Energy, QM9 target 10)
            via a PINO (Physics-Informed Neural Operator) loss term.

  Pipeline
  --------
  Stage 1 : Automatic QM9 download & preprocessing
  Stage 2 : GNN Score Network (the "denoiser" inside the diffusion model)
  Stage 3 : PINO loss (Gibbs Free Energy thermodynamic penalty)
  Stage 4 : Training loop + molecule sampling + RDKit SMILES validation

  Physics motivation
  ------------------
  A molecule with low Gibbs Free Energy (G) is thermodynamically stable.
  The PINO term penalises the model whenever its predicted score corresponds
  to a conformer whose G prediction deviates from the target G of the
  reference molecule, thereby nudging the generative process to stay in
  low-energy regions of chemical space — exactly what a catalyst designer
  needs.
=============================================================================
"""

# ---------------------------------------------------------------------------
#  0.  IMPORTS
# ---------------------------------------------------------------------------
import os
import math
import random
import warnings
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch_geometric
from torch_geometric.datasets import QM9
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import (
    MessagePassing,
    global_mean_pool,
    global_add_pool,
)
from torch_geometric.transforms import Compose, NormalizeFeatures
from torch_geometric.utils import to_dense_adj, dense_to_sparse

# RDKit for chemical validity
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Draw, AllChem, Descriptors

RDLogger.DisableLog("rdApp.*")           # suppress verbose RDKit warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
#  1.  GLOBAL CONFIGURATION
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT    = "./data"                  # QM9 will be downloaded here
BATCH_SIZE   = 32
HIDDEN_DIM   = 128
NUM_LAYERS   = 4                         # GNN message-passing rounds
T_MAX        = 1000                      # diffusion timesteps
EPOCHS       = 50                        # increase for real training
LR           = 3e-4
PINO_WEIGHT  = 0.1                       # λ in Loss = L_score + λ·L_PINO
GIBBS_IDX    = 10                        # QM9 target index for G (Gibbs Free Energy)

# QM9 atomic numbers (H, C, N, O, F) and their one-hot size
ATOM_LIST    = [1, 6, 7, 8, 9]          # H C N O F
NUM_ATOM_FEAT = 11                       # QM9 node feature dimensionality

print(f"[INFO] Running on device: {DEVICE}")
print(f"[INFO] PyTorch {torch.__version__} | PyG {torch_geometric.__version__}")


# ===========================================================================
#  STAGE 1 — QM9 DATASET LOADING & PREPROCESSING
# ===========================================================================

def load_qm9(root: str = DATA_ROOT,
             max_samples: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
    """
    Automatically downloads QM9 (if not present) and returns train/val
    DataLoaders.

    QM9 contains ~134k organic molecules with up to 9 heavy atoms (C, N, O, F)
    and their DFT-computed quantum-chemical properties.  Each molecule is
    represented as a graph where:
        - Nodes   : atoms  (11-dim feature vector)
        - Edges   : bonds  (4-dim one-hot bond-type feature)
        - y       : 19-dim vector of molecular properties

    Target index 10  →  G (Gibbs Free Energy) in eV  [thermodynamics]
    """
    print("[Stage 1] Downloading / loading QM9 dataset ...")
    dataset = QM9(root=root)

    # Normalise Gibbs Free Energy column to zero-mean, unit-variance
    # (important for stable PINO loss scaling)
    g_vals = dataset.data.y[:, GIBBS_IDX]          # shape: (N_mols,)
    g_mean = g_vals.mean().item()
    g_std  = g_vals.std().item()
    dataset.data.y[:, GIBBS_IDX] = (g_vals - g_mean) / (g_std + 1e-8)

    print(f"[Stage 1] Total molecules: {len(dataset):,}")
    print(f"[Stage 1] G (Gibbs) mean={g_mean:.4f} eV  std={g_std:.4f} eV")

    # Subsample for quick experimentation
    if max_samples:
        indices = torch.randperm(len(dataset))[:max_samples]
        dataset = dataset[indices]
        print(f"[Stage 1] Using subset of {max_samples:,} molecules")

    # 90/10 train-val split
    split   = int(0.9 * len(dataset))
    train_d = dataset[:split]
    val_d   = dataset[split:]

    train_loader = DataLoader(train_d, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_d,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    print(f"[Stage 1] Train: {len(train_d):,} | Val: {len(val_d):,}")
    return train_loader, val_loader, g_mean, g_std


# ===========================================================================
#  STAGE 2 — GNN SCORE NETWORK (Core of the Diffusion Model)
# ===========================================================================

class EquivariantEdgeConv(MessagePassing):
    """
    A simple equivariant-inspired message-passing layer.

    Physics rationale
    -----------------
    Molecular energy is invariant to global rotation and translation of the
    molecule (SE(3) symmetry).  We approximate this by conditioning on
    inter-atomic *distances* (a roto-translationally invariant scalar)
    rather than raw 3-D coordinates.

    Message = MLP( h_i ⊕ h_j ⊕ ||r_ij|| )
    where r_ij is the relative position vector between atoms i and j.
    """

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 4):
        super().__init__(aggr="mean")       # mean aggregation
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + 1 + edge_dim, out_dim),   # +1 for distance
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: Tensor, pos: Tensor,
                edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        return self.propagate(edge_index, x=x, pos=pos, edge_attr=edge_attr)

    def message(self, x_i: Tensor, x_j: Tensor,
                pos_i: Tensor, pos_j: Tensor,
                edge_attr: Tensor) -> Tensor:
        # Roto-translationally invariant distance scalar
        dist = (pos_i - pos_j).norm(dim=-1, keepdim=True)      # (E, 1)
        msg_input = torch.cat([x_i, x_j, dist, edge_attr], dim=-1)
        return self.mlp(msg_input)


class TimeEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding (same as in DDPM / Ho et al. 2020).

    Diffusion physics:  at timestep t the noisy position is
        x_t = sqrt(ᾱ_t) · x_0 + sqrt(1 - ᾱ_t) · ε,   ε ~ N(0, I)
    The score network must know t to correctly predict ε.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        # t : (B,)  integer timestep
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        emb = t[:, None].float() * freqs[None, :]   # (B, half)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, dim)
        return self.proj(emb)


class GNNScoreNetwork(nn.Module):
    """
    Score Network  s_θ(x_t, t) ≈ -∇_{x_t} log p(x_t | t)

    Architecture
    ------------
    1. Atom feature encoder  →  hidden embeddings
    2. Time embedding injected at every GNN layer (FiLM-style conditioning)
    3. NUM_LAYERS of EquivariantEdgeConv for message passing
    4. Readout MLP that predicts the *noise* ε (score proxy) for each atom's
       3-D position, used in the denoising score-matching objective.

    The network also has a side-head that predicts G (Gibbs Free Energy)
    directly from the graph embedding — this shared representation is what
    allows the PINO loss to back-propagate thermodynamic constraints through
    the same parameters.
    """

    def __init__(self,
                 node_feat_dim: int = NUM_ATOM_FEAT,
                 edge_feat_dim: int = 4,
                 hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = NUM_LAYERS):
        super().__init__()

        # --- atom feature encoder ---
        self.node_enc = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # --- time embedding ---
        self.time_emb = TimeEmbedding(hidden_dim)

        # --- GNN layers + FiLM scale/shift per layer ---
        self.gnn_layers   = nn.ModuleList()
        self.film_scale   = nn.ModuleList()
        self.film_shift   = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(EquivariantEdgeConv(hidden_dim, hidden_dim, edge_feat_dim))
            self.film_scale.append(nn.Linear(hidden_dim, hidden_dim))
            self.film_shift.append(nn.Linear(hidden_dim, hidden_dim))

        # --- score head: predicts ε (noise) on 3-D positions ---
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),          # Δx, Δy, Δz per atom
        )

        # --- PINO auxiliary head: predicts G from graph embedding ---
        # This is the "physics awareness" of the network: it shares weights
        # with the score predictor, so thermodynamic info flows into ε.
        self.gibbs_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),     # single scalar G per molecule
        )

    def forward(self,
                x:          Tensor,       # node features  (N, F)
                pos:        Tensor,       # 3-D coordinates (N, 3)
                edge_index: Tensor,       # (2, E)
                edge_attr:  Tensor,       # edge features   (E, 4)
                batch:      Tensor,       # batch vector    (N,)
                t:          Tensor        # timestep        (B,)
                ) -> Tuple[Tensor, Tensor]:
        """
        Returns
        -------
        score   : (N, 3)  predicted noise on atomic positions
        g_pred  : (B, 1)  predicted Gibbs Free Energy (normalised)
        """
        h = self.node_enc(x)                         # (N, H)
        t_emb = self.time_emb(t)                     # (B, H)
        t_emb_per_atom = t_emb[batch]                # (N, H)

        for i, layer in enumerate(self.gnn_layers):
            h = layer(h, pos, edge_index, edge_attr)  # message passing
            # FiLM conditioning: scale and shift by time embedding
            scale = self.film_scale[i](t_emb_per_atom)
            shift = self.film_shift[i](t_emb_per_atom)
            h = F.silu(h * scale + shift)             # (N, H)

        # Per-atom score prediction
        score = self.score_head(h)                    # (N, 3)

        # Graph-level G prediction (mean pooling → MLP)
        h_graph = global_mean_pool(h, batch)          # (B, H)
        g_pred  = self.gibbs_head(h_graph)            # (B, 1)

        return score, g_pred


# ---------------------------------------------------------------------------
#  Diffusion Schedule (Variance-Preserving, DDPM style)
# ---------------------------------------------------------------------------

class DiffusionSchedule:
    """
    Pre-computes the cosine noise schedule:
        ᾱ_t  = cos²(π/2 · (t/T + s) / (1 + s))    [Nichol & Dhariwal 2021]

    Key quantities
    --------------
    alpha_bar   : ᾱ_t  (signal retention at timestep t)
    sqrt_ab     : √ᾱ_t (scale of x_0 in the forward process)
    sqrt_1m_ab  : √(1 - ᾱ_t) (scale of noise in the forward process)
    """

    def __init__(self, T: int = T_MAX, s: float = 0.008):
        t = torch.arange(T + 1, dtype=torch.float32)
        f = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f / f[0]
        self.alpha_bar    = alpha_bar.to(DEVICE)
        self.sqrt_ab      = alpha_bar.sqrt().to(DEVICE)
        self.sqrt_1m_ab   = (1 - alpha_bar).sqrt().to(DEVICE)
        self.T = T

    def q_sample(self, x0: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward diffusion: add Gaussian noise to positions at timestep t.
            x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
        Returns (x_t, ε) — ε is the ground-truth noise the model must predict.
        """
        eps = torch.randn_like(x0)
        sqrt_ab   = self.sqrt_ab[t][..., None]      # broadcast over (N,3)
        sqrt_1mab = self.sqrt_1m_ab[t][..., None]
        return sqrt_ab * x0 + sqrt_1mab * eps, eps

    @torch.no_grad()
    def p_sample(self, model: nn.Module, data: Data,
                 num_steps: int = 200) -> Tensor:
        """
        Reverse diffusion (DDPM sampler) — generates atomic positions.
        Starting from pure Gaussian noise, iteratively denoises.
        """
        model.eval()
        x = data.pos.clone()
        pos = torch.randn_like(x).to(DEVICE)       # start from noise

        step_ids = torch.linspace(self.T - 1, 0, num_steps).long().to(DEVICE)

        for t_val in step_ids:
            t_batch = t_val.expand(data.batch.max().item() + 1)
            ab      = self.alpha_bar[t_val]
            ab_prev = self.alpha_bar[t_val - 1] if t_val > 0 else torch.tensor(1.0)

            score, _ = model(data.x.to(DEVICE), pos,
                             data.edge_index.to(DEVICE),
                             data.edge_attr.to(DEVICE),
                             data.batch.to(DEVICE),
                             t_batch)

            # DDPM reverse step:  x_{t-1} = (x_t - β_t/√(1-ᾱ_t) · ε_θ) / √α_t
            beta  = 1 - ab / ab_prev
            coeff = beta / (1 - ab).sqrt()
            pos   = (pos - coeff * score) / (1 - beta).sqrt()

            if t_val > 0:
                pos = pos + beta.sqrt() * torch.randn_like(pos)

        return pos


# ===========================================================================
#  STAGE 3 — PINO LOSS (Physics-Informed Neural Operator)
# ===========================================================================

class PINOThermodynamicLoss(nn.Module):
    """
    Physics-Informed Neural Operator (PINO) Loss

    Motivation
    ----------
    Classical diffusion models optimise only the score-matching (denoising)
    MSE.  This is purely data-driven — it will replicate the training
    distribution but has no guarantee of thermodynamic feasibility.

    We add a PINO residual loss:

        L_PINO  =  || G_pred - G_target ||²

    where G_pred is the model's auxiliary prediction of Gibbs Free Energy,
    and G_target is the true (normalised) DFT-computed G from QM9.

    Physical interpretation
    -----------------------
    By minimising L_PINO, the shared GNN backbone learns representations
    that encode thermodynamic stability.  The score-head then inherits this
    inductive bias, steering generated geometries toward low-G regions.

    Total loss:
        L_total = L_score  +  λ · L_PINO
            L_score  = MSE(ε_pred, ε)           [denoising score matching]
            L_PINO   = MSE(G_pred, G_target)    [thermodynamic constraint]
    """

    def __init__(self, pino_weight: float = PINO_WEIGHT):
        super().__init__()
        self.lam = pino_weight

    def forward(self,
                eps_pred:  Tensor,    # (N, 3) — predicted noise on positions
                eps_true:  Tensor,    # (N, 3) — true added noise
                g_pred:    Tensor,    # (B, 1) — predicted G (normalised)
                g_target:  Tensor,    # (B,)   — true G from QM9 (normalised)
                ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns
        -------
        loss_total, loss_score, loss_pino
        """
        # Denoising score matching (DSM) — the standard diffusion objective
        loss_score = F.mse_loss(eps_pred, eps_true)

        # Thermodynamic (PINO) penalty — Gibbs Free Energy residual
        loss_pino  = F.mse_loss(g_pred.squeeze(-1), g_target)

        # Combined loss
        loss_total = loss_score + self.lam * loss_pino

        return loss_total, loss_score, loss_pino


# ===========================================================================
#  STAGE 4A — TRAINING LOOP
# ===========================================================================

def train_epoch(model:     nn.Module,
                loader:    DataLoader,
                schedule:  DiffusionSchedule,
                criterion: PINOThermodynamicLoss,
                optimizer: torch.optim.Optimizer) -> dict:
    """
    One epoch of training.

    For each graph batch:
    1. Sample a random timestep t ~ Uniform{1, T}
    2. Apply forward diffusion to 3-D coordinates:  x_t = q_sample(x_0, t)
    3. Run the GNN score network to predict ε and G
    4. Compute combined loss L_total = L_score + λ · L_PINO
    5. Back-propagate and update θ
    """
    model.train()
    total_loss, total_score, total_pino = 0., 0., 0.
    n_batches = 0

    for data in loader:
        data = data.to(DEVICE)

        # Guard: skip batch if no positional data
        if data.pos is None:
            continue

        # --- Step 1: sample random timestep per molecule in batch ---
        B = data.batch.max().item() + 1
        t = torch.randint(1, schedule.T, (B,), device=DEVICE)  # (B,)
        t_per_atom = t[data.batch]                               # (N,)

        # --- Step 2: forward diffusion on atomic positions ---
        x0  = data.pos                                           # (N, 3)
        x_t, eps_true = schedule.q_sample(x0, t_per_atom)       # noisy coords

        # Detach x_t from x0 graph — we don't want gradients through noise
        x_t = x_t.detach()

        # --- Step 3: run score network ---
        eps_pred, g_pred = model(data.x, x_t, data.edge_index,
                                 data.edge_attr, data.batch, t)

        # --- Step 4: compute PINO + DSM loss ---
        g_target = data.y[:, GIBBS_IDX]         # (B,) normalised G values
        loss, l_score, l_pino = criterion(eps_pred, eps_true, g_pred, g_target)

        # --- Step 5: back-prop ---
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss  += loss.item()
        total_score += l_score.item()
        total_pino  += l_pino.item()
        n_batches   += 1

    return {
        "loss":       total_loss  / max(n_batches, 1),
        "loss_score": total_score / max(n_batches, 1),
        "loss_pino":  total_pino  / max(n_batches, 1),
    }


@torch.no_grad()
def validate_epoch(model:     nn.Module,
                   loader:    DataLoader,
                   schedule:  DiffusionSchedule,
                   criterion: PINOThermodynamicLoss) -> dict:
    """Validation pass (no weight updates)."""
    model.eval()
    total_loss, total_score, total_pino = 0., 0., 0.
    n_batches = 0

    for data in loader:
        data = data.to(DEVICE)
        if data.pos is None:
            continue

        B = data.batch.max().item() + 1
        t = torch.randint(1, schedule.T, (B,), device=DEVICE)
        t_per_atom = t[data.batch]

        x0 = data.pos
        x_t, eps_true = schedule.q_sample(x0, t_per_atom)
        eps_pred, g_pred = model(data.x, x_t, data.edge_index,
                                 data.edge_attr, data.batch, t)

        g_target = data.y[:, GIBBS_IDX]
        loss, l_score, l_pino = criterion(eps_pred, eps_true, g_pred, g_target)

        total_loss  += loss.item()
        total_score += l_score.item()
        total_pino  += l_pino.item()
        n_batches   += 1

    return {
        "val_loss":       total_loss  / max(n_batches, 1),
        "val_loss_score": total_score / max(n_batches, 1),
        "val_loss_pino":  total_pino  / max(n_batches, 1),
    }


def train(model:     nn.Module,
          train_ld:  DataLoader,
          val_ld:    DataLoader,
          schedule:  DiffusionSchedule,
          criterion: PINOThermodynamicLoss,
          epochs:    int = EPOCHS) -> nn.Module:
    """Full training loop with cosine LR scheduling."""
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val  = float("inf")
    ckpt_path = "./pgmd_checkpoint.pt"

    print("\n" + "="*65)
    print("  STAGE 4 — Training  (score matching + PINO thermodynamic loss)")
    print("="*65)

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, train_ld, schedule, criterion, optimizer)
        val_metrics   = validate_epoch(model, val_ld, schedule, criterion)
        scheduler.step()

        log = (f"Epoch {epoch:03d}/{epochs}  "
               f"Loss: {train_metrics['loss']:.4f}  "
               f"(DSM={train_metrics['loss_score']:.4f} "
               f"PINO={train_metrics['loss_pino']:.4f})  |  "
               f"Val: {val_metrics['val_loss']:.4f}")
        print(log)

        # Save best model
        if val_metrics["val_loss"] < best_val:
            best_val = val_metrics["val_loss"]
            torch.save(model.state_dict(), ckpt_path)

    print(f"\n[Train] Best val loss: {best_val:.4f}  |  Checkpoint: {ckpt_path}")
    # Reload best weights
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return model


# ===========================================================================
#  STAGE 4B — SAMPLING & CHEMICAL VALIDITY (RDKit)
# ===========================================================================

# Map QM9 node-feature index → atomic number
# QM9 one-hot features: H(0) C(1) N(2) O(3) F(4), followed by other features
FEAT_TO_ATOMIC_NUM = {0: 1, 1: 6, 2: 7, 3: 8, 4: 9}   # index → atomic number
ATOMIC_NUM_TO_SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}

# Standard valences for validity checking
VALENCE_MAP = {1: 1, 6: 4, 7: 3, 8: 2, 9: 1}

# Covalent radii in Ångströms (Cordero et al., 2008)
COVALENT_RADII = {
    1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
    15: 1.07, 16: 1.05, 17: 1.02,
}

# Maximum allowed valence per element (for bond pruning)
MAX_VALENCE = {1: 1, 6: 4, 7: 3, 8: 2, 9: 1, 15: 5, 16: 6, 17: 1}

# Mean pairwise distance in real QM9 molecules (computed from dataset)
QM9_MEAN_PAIRWISE_DIST = 3.16  # Ångströms


def node_features_to_atomic_num(x: Tensor) -> list:
    """
    Decode QM9 node feature vectors → atomic numbers.
    The first 5 features in QM9 x are one-hot atom type: [H, C, N, O, F].
    """
    one_hot = x[:, :5]                               # (N, 5)
    atom_type_idx = one_hot.argmax(dim=-1).tolist()
    return [FEAT_TO_ATOMIC_NUM.get(idx, 6) for idx in atom_type_idx]


def rescale_to_qm9(coords_np):
    """
    Rescale generated coordinates so the local bonding geometry matches
    real QM9 molecules.

    Strategy: match the median nearest-neighbor distance to real QM9
    molecules (~1.15 Å, which corresponds to C-H and C-C bond lengths).
    This ensures bonded atom pairs land at realistic distances for
    rdDetermineBonds and covalent-radii bond inference to work.
    """
    from scipy.spatial.distance import pdist, squareform
    centroid = coords_np.mean(axis=0)
    centered = coords_np - centroid
    n = len(centered)
    if n < 2:
        return centered

    # Compute nearest-neighbor distances
    dist_matrix = squareform(pdist(centered))
    np.fill_diagonal(dist_matrix, np.inf)
    nn_dists = dist_matrix.min(axis=1)
    median_nn = np.median(nn_dists)

    if median_nn < 1e-6:
        return centered

    # Real QM9 median nearest-neighbor distance (C-H ~1.09, C-C ~1.54)
    TARGET_NN_DIST = 1.15
    scale = TARGET_NN_DIST / (median_nn + 1e-8)
    return centered * scale


def infer_bonds_from_distance(pos: Tensor,
                               atomic_nums: list,
                               threshold: float = 1.8) -> list:
    """
    Infer bonds from 3-D geometry using element-pair-specific covalent
    radius cutoffs.  Falls back to a global threshold if covalent radii
    are unavailable for a given element.

    Bond cutoff for pair (i, j):
        d_max = cov_radius_i + cov_radius_j + tolerance

    The `threshold` parameter is used as the tolerance added on top of
    the sum of covalent radii.
    """
    tolerance = threshold - 1.2  # offset so default 1.8 → tol=0.6
    tolerance = max(tolerance, 0.3)  # at least 0.3 Å tolerance
    N = pos.shape[0]
    bonds = []
    pos_np = pos.cpu().numpy() if hasattr(pos, 'cpu') else np.asarray(pos)

    for i in range(N):
        ri = COVALENT_RADII.get(atomic_nums[i], 0.76)
        for j in range(i + 1, N):
            rj = COVALENT_RADII.get(atomic_nums[j], 0.76)
            dist = np.linalg.norm(pos_np[i] - pos_np[j])
            cutoff = ri + rj + tolerance
            if dist < cutoff:
                bonds.append((i, j, dist, Chem.rdchem.BondType.SINGLE))
    return bonds


def prune_bonds_for_valence(atomic_nums: list, bonds: list) -> list:
    """
    Remove the longest bonds first whenever an atom exceeds its maximum
    allowed valence.  This prevents SanitizeMol from rejecting molecules
    that have the right connectivity but too many bonds on some atoms.

    Args:
        atomic_nums: list of atomic numbers (length N)
        bonds: list of (i, j, dist, bond_type) tuples — MUST include dist

    Returns:
        pruned list of (i, j, bond_type) tuples (dist removed)
    """
    # Sort ALL bonds longest-first so we remove the weakest first
    sorted_bonds = sorted(bonds, key=lambda b: -b[2])

    # Build adjacency count
    valence_count = {i: 0 for i in range(len(atomic_nums))}
    kept = []

    # Iterate shortest-first to keep the best bonds
    for b in reversed(sorted_bonds):
        i, j = b[0], b[1]
        btype = b[3]
        max_i = MAX_VALENCE.get(atomic_nums[i], 4)
        max_j = MAX_VALENCE.get(atomic_nums[j], 4)

        if valence_count[i] < max_i and valence_count[j] < max_j:
            kept.append((i, j, btype))
            valence_count[i] += 1
            valence_count[j] += 1

    return kept


def graph_to_rdkit_mol(atomic_nums: list,
                        bonds: list) -> Optional[Chem.Mol]:
    """
    Construct an RDKit Mol object from atomic numbers + bond list.
    Accepts bonds as either (i, j, bond_type) or (i, j, dist, bond_type).
    Returns None on failure.
    """
    try:
        mol = Chem.RWMol()
        for an in atomic_nums:
            atom = Chem.Atom(an)
            mol.AddAtom(atom)
        for bond in bonds:
            if len(bond) == 4:
                i, j, _dist, btype = bond
            else:
                i, j, btype = bond
            mol.AddBond(i, j, btype)
        mol = mol.GetMol()
        Chem.SanitizeMol(mol)          # triggers valence check
        return mol
    except Exception:
        return None


def check_chemical_validity(mol: Optional[Chem.Mol]) -> bool:
    """
    Checks whether an RDKit molecule passes:
    1. RDKit sanitisation (aromaticity, valence)
    2. SMILES round-trip (parseable)
    """
    if mol is None:
        return False
    try:
        smi = Chem.MolToSmiles(mol)
        check = Chem.MolFromSmiles(smi)
        return check is not None and len(smi) > 0
    except Exception:
        return False


@torch.no_grad()
def sample_molecule(model:     nn.Module,
                    schedule:  DiffusionSchedule,
                    ref_data:  Data,
                    num_steps: int = 200) -> str:
    """
    Samples a *new* molecular geometry using the reverse diffusion process
    and converts it to a SMILES string via RDKit.

    Sampling workflow
    -----------------
    1. Take a reference graph (topology) from QM9 as scaffold
       (atom types + bond connectivity from data; positions re-initialised
       to pure Gaussian noise).
    2. Run the DDPM reverse sampler for `num_steps` denoising steps.
    3. Decode generated 3-D positions → atom types → infer bonds → SMILES.
    4. Validate with RDKit.

    Note: In a full generative model you would also learn the graph topology
    (atom types, bond types) jointly with positions.  Here we keep the
    topology fixed and only generate novel geometries to focus on the
    physics-guided diffusion aspect.
    """
    model.eval()
    ref_data = ref_data.to(DEVICE)

    # Generate new atomic positions via reverse diffusion
    gen_pos = schedule.p_sample(model, ref_data, num_steps=num_steps)  # (N, 3)

    # Decode atom types from node features
    atomic_nums = node_features_to_atomic_num(ref_data.x.cpu())

    # Infer bonds from generated 3-D geometry
    bonds = infer_bonds_from_distance(gen_pos.cpu(), atomic_nums)

    # Build RDKit molecule
    mol = graph_to_rdkit_mol(atomic_nums, bonds)

    # Validate and return SMILES
    if check_chemical_validity(mol):
        smi = Chem.MolToSmiles(mol)
        print(f"\n[Sample] ✓ Chemically valid SMILES: {smi}")
        return smi
    else:
        print("[Sample] ✗ Generated molecule failed chemical validity check.")
        print("          (Normal early in training; validity improves as the")
        print("           PINO loss guides the model toward feasible geometries.)")
        return ""


# ===========================================================================
#  MAIN — Orchestrate all stages
# ===========================================================================

def main():
    print("\n" + "="*65)
    print("  Physics-Guided Molecular Diffusion (PGMD) for Catalyst Design")
    print("="*65 + "\n")

    # ------------------------------------------------------------------
    # Stage 1: Load QM9
    # ------------------------------------------------------------------
    # Use max_samples=5000 for a quick demo. Remove / increase for full run.
    train_loader, val_loader, g_mean, g_std = load_qm9(max_samples=5000)

    # ------------------------------------------------------------------
    # Stage 2: Instantiate model + diffusion schedule
    # ------------------------------------------------------------------
    print("\n[Stage 2] Building GNN Score Network ...")
    model = GNNScoreNetwork(
        node_feat_dim = NUM_ATOM_FEAT,
        edge_feat_dim = 4,
        hidden_dim    = HIDDEN_DIM,
        num_layers    = NUM_LAYERS,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Stage 2] Model parameters: {n_params:,}")

    schedule = DiffusionSchedule(T=T_MAX)

    # ------------------------------------------------------------------
    # Stage 3: PINO loss
    # ------------------------------------------------------------------
    print(f"\n[Stage 3] PINO thermodynamic loss  (λ = {PINO_WEIGHT})")
    criterion = PINOThermodynamicLoss(pino_weight=PINO_WEIGHT)

    # ------------------------------------------------------------------
    # Stage 4A: Train
    # ------------------------------------------------------------------
    model = train(model, train_loader, val_loader, schedule, criterion, epochs=EPOCHS)

    # ------------------------------------------------------------------
    # Stage 4B: Sample & validate
    # ------------------------------------------------------------------
    print("\n[Stage 4B] Sampling a molecule with the trained diffusion model ...")
    # Pick one reference graph from the validation set
    sample_batch = next(iter(val_loader))
    # Take the first individual graph (un-batch it)
    ref_graph = sample_batch.get_example(0)

    smiles = sample_molecule(model, schedule, ref_graph, num_steps=200)

    if smiles:
        print(f"\n[Result] Generated SMILES : {smiles}")
        # Compute a basic physicochemical property with RDKit
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mw = Descriptors.MolWt(mol)
            print(f"[Result] Molecular weight : {mw:.2f} g/mol")
    else:
        print("\n[Result] No valid molecule generated in this run.")
        print("         Increase EPOCHS and MAX_SAMPLES for better results.")

    print("\n[Done] PGMD training and sampling complete.\n")


if __name__ == "__main__":
    main()
