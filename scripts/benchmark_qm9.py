"""
QM9 Benchmark Evaluation — PGMD Diffusion Model
=================================================
Evaluates the trained EGNN model against the real QM9 dataset using the
six standard metrics reported in molecular generation papers:

  1. Atom Stability   — % of atoms with correct valence
  2. Molecule Stability — % of molecules where every atom is stable
  3. Validity         — % passing RDKit sanitisation
  4. Uniqueness       — % of valid molecules that are structurally distinct
  5. Novelty          — % not seen in the QM9 training split
  6. Gibbs MAE        — Mean Absolute Error vs DFT-computed G from QM9

Results are printed as a comparison table against EDM, GDSS, and GeoDiff
baselines from the literature and saved to benchmark_results.json.

Usage (run from the project root):
    python scripts/benchmark_qm9.py                         # default: 1000 samples, 200 steps
    python scripts/benchmark_qm9.py --n_samples 100 --steps 50   # quick smoke test
    python scripts/benchmark_qm9.py --n_samples 1000 --steps 200 --output results.json
"""

import os
import sys
import json
import time
import math
import random
import argparse
import warnings

warnings.filterwarnings("ignore")

# ── Project root on sys.path so imports work when run directly ────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

from scripts.model_arch import EGNNScoreNetwork
from scripts.physics_guided_molecular_diffusion import (
    load_qm9, DiffusionSchedule,
    NUM_ATOM_FEAT, HIDDEN_DIM, NUM_LAYERS, T_MAX,
    node_features_to_atomic_num,
    infer_bonds_from_distance,
    graph_to_rdkit_mol,
    check_chemical_validity,
)

# ── Constants ─────────────────────────────────────────────────────────────────

# QM9 target index 10 = Gibbs Free Energy (eV)
GIBBS_IDX = 10

# Expected valence per element for atom-stability check (EDM paper definition)
EXPECTED_VALENCE = {1: 1, 6: 4, 7: 3, 8: 2, 9: 1}   # H C N O F

# Published baselines for the comparison table
BASELINES = {
    "EDM  (Hoogeboom 2022)": {"atom_stab": 98.7, "mol_stab": 82.0, "valid": 91.9, "unique": 90.7, "novel": 100.0},
    "GDSS (Jo et al. 2022)": {"atom_stab": 95.7, "mol_stab": 63.2, "valid": 90.2, "unique": 95.0, "novel": 100.0},
    "GeoDiff (Xu 2022)    ": {"atom_stab": None,  "mol_stab": None,  "valid": 99.0, "unique": None,  "novel": None},
}


# ── Metric helpers ────────────────────────────────────────────────────────────

def atom_stability(mol):
    """
    Returns (fraction_stable, n_total_atoms, n_stable_atoms).
    A stable atom has total_valence == expected valence for its element.
    If mol is None every atom is counted as unstable.
    """
    if mol is None:
        return 0.0, 0, 0
    n_stable = 0
    for atom in mol.GetAtoms():
        an = atom.GetAtomicNum()
        expected = EXPECTED_VALENCE.get(an)
        if expected is not None and atom.GetTotalValence() == expected:
            n_stable += 1
    total = mol.GetNumAtoms()
    return (n_stable / total if total else 0.0), total, n_stable


def molecule_stability(mol):
    """True if every atom in the molecule is valence-stable."""
    if mol is None:
        return False
    for atom in mol.GetAtoms():
        an = atom.GetAtomicNum()
        expected = EXPECTED_VALENCE.get(an)
        if expected is not None and atom.GetTotalValence() != expected:
            return False
    return True


# ── Diffusion sampler ─────────────────────────────────────────────────────────

@torch.no_grad()
def _ddpm_core(model, ref_graph, schedule, device, num_steps, noise_scale, pino_weight, guidance):
    """
    Single reverse-diffusion pass (no_grad for speed; PINO grad computed separately).
    Returns raw generated coordinates (N,3) and model-predicted G scalar.
    """
    ref_graph  = ref_graph.to(device)
    n_atoms    = ref_graph.x.size(0)
    batch_vec  = torch.zeros(n_atoms, dtype=torch.long, device=device)
    x_feat     = ref_graph.x
    edge_index = ref_graph.edge_index
    edge_attr  = ref_graph.edge_attr

    pos      = torch.randn(n_atoms, 3, device=device) * noise_scale
    step_ids = torch.linspace(schedule.T - 1, 0, num_steps).long().to(device)
    g_pred_out = 0.0

    for t_val in step_ids:
        # Need grad for PINO guidance step
        pos_g = pos.detach().requires_grad_(True)

        t_batch = t_val.expand(1)
        ab      = schedule.alpha_bar[t_val]
        ab_prev = schedule.alpha_bar[t_val - 1] if t_val > 0 else torch.tensor(1.0, device=device)

        score, g_pred = model(x_feat, pos_g, edge_index, edge_attr, batch_vec, t_batch)
        g_pred_out    = g_pred[0, GIBBS_IDX % g_pred.shape[-1]].item()

        # PINO guidance gradient
        grad_g = torch.autograd.grad(g_pred.sum(), pos_g)[0]

        score = score.detach() * guidance

        beta      = 1 - ab / ab_prev
        coeff     = beta / torch.clamp(1 - ab, min=1e-5).sqrt()
        pos_prev  = (pos_g - coeff * score - pino_weight * grad_g.detach()) / \
                    torch.clamp(1 - beta, min=1e-5).sqrt()

        if t_val > 0:
            pos_prev = pos_prev + beta.sqrt() * torch.randn_like(pos_prev) * noise_scale

        pos = pos_prev.detach()

    return pos.cpu().numpy(), g_pred_out


def sample_one(model, ref_graph, schedule, device, num_steps, noise_scale, pino_weight, guidance):
    """
    Generate one molecule.  Returns (gen_pos_np, atomic_nums, g_pred_scalar).
    """
    gen_pos, g_pred = _ddpm_core(
        model, ref_graph, schedule, device,
        num_steps, noise_scale, pino_weight, guidance
    )
    anums = node_features_to_atomic_num(ref_graph.x.cpu())

    # Normalise coordinates (same as server.py)
    centroid  = gen_pos.mean(axis=0)
    gen_pos  -= centroid
    max_span  = np.abs(gen_pos).max() + 1e-8
    if max_span > 4.0:
        gen_pos = gen_pos * (4.0 / max_span)

    return gen_pos, anums, g_pred


def coords_to_mol(gen_pos, anums):
    """
    Try distance-threshold bond inference with cascading thresholds.
    Returns an RDKit Mol on success, else None.
    """
    import torch as _torch
    pos_t = _torch.tensor(gen_pos)
    for thresh in [1.8, 2.0, 2.3, 2.6]:
        bonds = infer_bonds_from_distance(pos_t, anums, threshold=thresh)
        mol   = graph_to_rdkit_mol(anums, bonds)
        if check_chemical_validity(mol):
            return mol
    return None


# ── Training SMILES builder ───────────────────────────────────────────────────

def build_train_smiles(train_loader, max_batches=None):
    """
    Extract SMILES from the QM9 training split to use as the novelty reference.
    Uses distance-threshold bonding on the ground-truth QM9 coordinates.
    """
    smiles_set = set()
    for batch_idx, batch in enumerate(train_loader):
        if max_batches and batch_idx >= max_batches:
            break
        for i in range(batch.num_graphs):
            try:
                graph = batch.get_example(i)
                if graph.pos is None:
                    continue
                anums = node_features_to_atomic_num(graph.x)
                mol   = coords_to_mol(graph.pos.numpy(), anums)
                if mol is not None:
                    smi = Chem.MolToSmiles(mol)
                    if smi:
                        smiles_set.add(smi)
            except Exception:
                pass
    return smiles_set


# ── Main benchmark ────────────────────────────────────────────────────────────

def run_benchmark(n_samples=1000, num_steps=200, seed=42, output_path=None,
                  noise_scale=1.0, pino_weight=0.1, guidance=1.0):
    """
    Full QM9 benchmark.  Steps:
      1. Load model
      2. Load QM9 (train split for novelty, val split for generation scaffolds)
      3. Build training SMILES reference set
      4. Generate n_samples molecules, compute per-molecule metrics
      5. Aggregate, print comparison table, save JSON
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    SEP = "=" * 65
    print(SEP)
    print("  PGMD  QM9 Benchmark Evaluation")
    print(SEP)
    print(f"  Device     : {device}")
    print(f"  Samples    : {n_samples}")
    print(f"  DDPM steps : {num_steps}")
    print(f"  Seed       : {seed}")
    print(SEP)

    # ── 1. Load model ────────────────────────────────────────────────────────
    print("\n[1/4] Loading EGNN model ...")
    model_path = os.path.join(PROJECT_ROOT, "models", "pgmd_v3_full.pt")
    if not os.path.exists(model_path):
        sys.exit(f"[ERROR] Model not found: {model_path}")

    model = EGNNScoreNetwork(NUM_ATOM_FEAT, 4, HIDDEN_DIM, NUM_LAYERS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      Loaded  |  Parameters: {n_params:,}")

    # ── 2. Load QM9 ──────────────────────────────────────────────────────────
    print("\n[2/4] Loading QM9 dataset ...")
    data_root = os.path.join(PROJECT_ROOT, "data")
    train_loader, val_loader, g_mean, g_std = load_qm9(root=data_root, max_samples=5000)
    print(f"      g_mean = {g_mean:.4f} eV  |  g_std = {g_std:.4f} eV")

    schedule = DiffusionSchedule(T=T_MAX)

    # ── 3. Build training SMILES (novelty reference) ──────────────────────────
    print("\n[3/4] Building training SMILES set for novelty check ...")
    train_smiles = build_train_smiles(train_loader)
    print(f"      Training SMILES collected: {len(train_smiles):,}")

    # ── 4. Generate & evaluate ────────────────────────────────────────────────
    print(f"\n[4/4] Generating {n_samples} molecules ...")
    print(f"      (steps={num_steps}, device={device}; CPU will be slow — reduce --n_samples for quick test)")

    atom_stab_fracs   = []   # fraction of stable atoms per molecule
    mol_stable_list   = []   # bool per molecule
    valid_smiles_list = []   # SMILES strings that passed validity
    g_pred_list       = []   # model-predicted Gibbs (normalised)
    g_true_list       = []   # QM9 true Gibbs (normalised)

    val_iter  = iter(val_loader)
    val_batch = next(val_iter)

    t_start = time.perf_counter()

    for i in range(n_samples):
        # Cycle through the val loader to get reference topologies
        graph_idx = i % val_batch.num_graphs
        if graph_idx == 0 and i > 0:
            try:
                val_batch = next(val_iter)
            except StopIteration:
                val_iter  = iter(val_loader)
                val_batch = next(val_iter)

        try:
            ref = val_batch.get_example(graph_idx)
        except Exception:
            continue

        # True Gibbs from QM9 dataset (already normalised by load_qm9)
        if ref.y is not None:
            y = ref.y
            g_true = y[0, GIBBS_IDX].item() if y.dim() == 2 else y[GIBBS_IDX].item()
            g_true_list.append(g_true)

        # Generate
        try:
            gen_pos, anums, g_pred = sample_one(
                model, ref, schedule, device,
                num_steps, noise_scale, pino_weight, guidance
            )
            g_pred_list.append(g_pred)
        except Exception:
            atom_stab_fracs.append(0.0)
            mol_stable_list.append(False)
            continue

        # Build RDKit mol
        mol = coords_to_mol(gen_pos, anums)

        # Per-molecule metrics
        frac, _, _ = atom_stability(mol)
        atom_stab_fracs.append(frac)
        mol_stable_list.append(molecule_stability(mol))

        # Validity + SMILES
        if mol is not None:
            try:
                smi = Chem.MolToSmiles(mol)
                if smi and Chem.MolFromSmiles(smi) is not None:
                    valid_smiles_list.append(smi)
            except Exception:
                pass

        # Progress report every 10 %
        done = i + 1
        if done % max(n_samples // 10, 1) == 0:
            elapsed = time.perf_counter() - t_start
            eta     = elapsed / done * (n_samples - done)
            pct_v   = len(valid_smiles_list) / done * 100
            print(f"      [{done:>5}/{n_samples}]  Valid: {pct_v:5.1f}%  |  "
                  f"Elapsed: {elapsed:6.0f}s  |  ETA: {eta:6.0f}s")

    total_time = time.perf_counter() - t_start

    # ── 5. Aggregate metrics ──────────────────────────────────────────────────
    n_gen          = n_samples
    unique_smiles  = list(set(valid_smiles_list))
    novel_smiles   = [s for s in unique_smiles if s not in train_smiles]

    atom_stab_pct  = float(np.mean(atom_stab_fracs)) * 100 if atom_stab_fracs else 0.0
    mol_stab_pct   = sum(mol_stable_list) / n_gen * 100
    valid_pct      = len(valid_smiles_list) / n_gen * 100
    unique_pct     = len(unique_smiles) / len(valid_smiles_list) * 100 if valid_smiles_list else 0.0
    novel_pct      = len(novel_smiles) / len(unique_smiles) * 100     if unique_smiles   else 0.0

    gibbs_mae_norm = gibbs_mae_ev = None
    if g_pred_list and g_true_list:
        n = min(len(g_pred_list), len(g_true_list))
        gibbs_mae_norm = float(np.mean(np.abs(np.array(g_pred_list[:n]) - np.array(g_true_list[:n]))))
        gibbs_mae_ev   = gibbs_mae_norm * g_std

    # ── 6. Print results table ────────────────────────────────────────────────
    def fmt(v):
        return f"{v:6.1f}%" if v is not None else "   — "

    print(f"\n{SEP}")
    print("  PGMD vs Published Baselines on QM9")
    print(SEP)
    hdr = f"  {'Method':<30} {'AtomStab':>9} {'MolStab':>8} {'Valid':>7} {'Unique':>8} {'Novel':>7}"
    print(hdr)
    print("  " + "-" * 63)
    for name, b in BASELINES.items():
        print(f"  {name:<30} {fmt(b['atom_stab']):>9} {fmt(b['mol_stab']):>8} "
              f"{fmt(b['valid']):>7} {fmt(b['unique']):>8} {fmt(b['novel']):>7}")
    print("  " + "-" * 63)
    print(f"  {'PGMD (ours)':<30} {fmt(atom_stab_pct):>9} {fmt(mol_stab_pct):>8} "
          f"{fmt(valid_pct):>7} {fmt(unique_pct):>8} {fmt(novel_pct):>7}")
    print(SEP)

    if gibbs_mae_ev is not None:
        print(f"\n  Gibbs MAE : {gibbs_mae_norm:.4f} (normalised)  =  {gibbs_mae_ev:.4f} eV")

    print(f"\n  Generated : {n_gen}")
    print(f"  Valid     : {len(valid_smiles_list)}  ({valid_pct:.1f}%)")
    print(f"  Unique    : {len(unique_smiles)}  ({unique_pct:.1f}% of valid)")
    print(f"  Novel     : {len(novel_smiles)}  ({novel_pct:.1f}% of unique)")
    print(f"  Time      : {total_time:.1f}s  ({total_time/n_gen:.2f}s per molecule)")

    # ── 7. Save JSON ──────────────────────────────────────────────────────────
    results = {
        "config": {
            "n_samples": n_samples, "num_steps": num_steps, "seed": seed,
            "device": str(device), "model": "pgmd_v3_full.pt",
            "noise_scale": noise_scale, "pino_weight": pino_weight, "guidance": guidance,
        },
        "metrics": {
            "atom_stability_pct":   round(atom_stab_pct, 4),
            "mol_stability_pct":    round(mol_stab_pct, 4),
            "validity_pct":         round(valid_pct, 4),
            "uniqueness_pct":       round(unique_pct, 4),
            "novelty_pct":          round(novel_pct, 4),
            "gibbs_mae_normalised": round(gibbs_mae_norm, 6) if gibbs_mae_norm else None,
            "gibbs_mae_ev":         round(gibbs_mae_ev,   6) if gibbs_mae_ev   else None,
        },
        "counts": {
            "generated": n_gen,
            "valid":     len(valid_smiles_list),
            "unique":    len(unique_smiles),
            "novel":     len(novel_smiles),
        },
        "timing": {
            "total_seconds":      round(total_time, 2),
            "seconds_per_sample": round(total_time / n_gen, 3),
        },
        "baselines": BASELINES,
        "novel_smiles_sample": novel_smiles[:20],
    }

    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, "benchmark_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved -> {output_path}")
    print(SEP + "\n")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate PGMD on QM9 with standard molecular generation metrics."
    )
    parser.add_argument("--n_samples",   type=int,   default=1000,
                        help="Molecules to generate (default: 1000)")
    parser.add_argument("--steps",       type=int,   default=200,
                        help="DDPM denoising steps (default: 200)")
    parser.add_argument("--seed",        type=int,   default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output",      type=str,   default=None,
                        help="Output JSON path (default: benchmark_results.json)")
    parser.add_argument("--noise_scale", type=float, default=1.0,
                        help="Initial noise magnitude (default: 1.0)")
    parser.add_argument("--pino_weight", type=float, default=0.1,
                        help="PINO guidance strength (default: 0.1)")
    parser.add_argument("--guidance",    type=float, default=1.0,
                        help="Classifier-free guidance scale (default: 1.0)")
    args = parser.parse_args()

    run_benchmark(
        n_samples   = args.n_samples,
        num_steps   = args.steps,
        seed        = args.seed,
        output_path = args.output,
        noise_scale = args.noise_scale,
        pino_weight = args.pino_weight,
        guidance    = args.guidance,
    )
