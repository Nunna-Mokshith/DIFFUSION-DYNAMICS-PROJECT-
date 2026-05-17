# -*- coding: utf-8 -*-
"""
Version A vs Version B — Inference Comparison
=============================================
Version A : Full EGNN + Gibbs head + PINO loss (trained checkpoint)
Version B : Baseline EGNN, score-matching only, random weights (no training)

Runs N molecules through both, collects:
  - validity %, connectivity %, atom stability %, mol stability %
  - predicted Gibbs energy (Version A only)
  - per-stage bond inference breakdown
  - SMILES diversity (unique count)

Saves results/comparison_AB.json for the HTML report generator.
"""

import os, sys, json, time, random
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
RDLogger.DisableLog("rdApp.*")

from scripts.model_arch import EGNNScoreNetwork
from scripts.baseline_model import BaselineScoreNetwork
from scripts.physics_guided_molecular_diffusion import (
    DiffusionSchedule, load_qm9,
    NUM_ATOM_FEAT, HIDDEN_DIM, NUM_LAYERS, T_MAX,
    node_features_to_atomic_num, infer_bonds_from_distance,
    graph_to_rdkit_mol, check_chemical_validity,
    rescale_to_qm9, prune_bonds_for_valence,
)
from scripts.Advanced_Molecular_Design import xyz_to_rdkit_mol, EXTENDED_SYM_MAP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLES  = 40          # molecules per model
NUM_STEPS  = 80          # denoising steps
SEEDS_BASE = 2000

EXPECTED_VALENCE = {1:1, 6:4, 7:3, 8:2, 9:1, 15:3, 16:2, 17:1}


def load_version_a():
    model = EGNNScoreNetwork(NUM_ATOM_FEAT, 4, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
    ckpt  = os.path.join(ROOT, "models", "pgmd_v3_full.pt")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def load_version_b():
    model = BaselineScoreNetwork(NUM_ATOM_FEAT, 4, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
    # No checkpoint — random weights simulate an untrained baseline
    torch.manual_seed(99)
    model.eval()
    return model


def run_reverse_diffusion(model, ref, schedule, num_steps, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    ref = ref.to(DEVICE)
    n_atoms  = ref.x.size(0)
    batch    = torch.zeros(n_atoms, dtype=torch.long, device=DEVICE)
    pos      = torch.randn(n_atoms, 3, device=DEVICE)
    step_ids = torch.linspace(schedule.T-1, 0, num_steps).long().to(DEVICE)
    x_feat   = ref.x.to(DEVICE)
    ei       = ref.edge_index.to(DEVICE)
    ea       = ref.edge_attr.to(DEVICE)

    g_preds = []
    with torch.no_grad():
        for t_val in step_ids:
            t_batch = t_val.expand(1)
            ab      = schedule.alpha_bar[t_val]
            ab_prev = schedule.alpha_bar[t_val-1] if t_val > 0 else torch.tensor(1.0, device=DEVICE)
            score, g = model(x_feat, pos, ei, ea, batch, t_batch)
            if g is not None:
                # prop_head outputs (B, 12); index 0 = Gibbs proxy
                g_scalar = g[0, 0].item() if g.dim() == 2 else g.mean().item()
                g_preds.append(g_scalar)
            beta  = 1 - ab / ab_prev
            coeff = beta / torch.clamp((1-ab), min=1e-5).sqrt()
            pos   = (pos - coeff * score) / torch.clamp((1-beta), min=1e-5).sqrt()
            if t_val > 0:
                pos = pos + beta.sqrt() * torch.randn_like(pos)

    gen_pos_np = pos.detach().cpu().numpy()
    gen_pos_np = rescale_to_qm9(gen_pos_np)
    anums      = node_features_to_atomic_num(ref.x.cpu())
    avg_g      = float(np.mean(g_preds)) if g_preds else None
    return gen_pos_np, anums, avg_g


def build_xyz(anums, coords):
    lines = [str(len(anums)), "gen"]
    for i, an in enumerate(anums):
        sym = EXTENDED_SYM_MAP.get(an, "C")
        lines.append(f"{sym} {coords[i,0]:.5f} {coords[i,1]:.5f} {coords[i,2]:.5f}")
    return "\n".join(lines) + "\n"


def bond_inference(anums, coords, xyz_str):
    """3-stage pipeline. Returns (mol, smi, stage)."""
    # Stage 1
    mol = xyz_to_rdkit_mol(xyz_str)
    if mol:
        smi = Chem.MolToSmiles(mol)
        if smi and "." not in smi:
            return mol, smi, 1

    # Stage 2
    pos_t = torch.tensor(coords, dtype=torch.float32)
    for thresh in [1.8, 2.0, 2.3, 2.6]:
        raw  = infer_bonds_from_distance(pos_t, anums, threshold=thresh)
        prn  = prune_bonds_for_valence(anums, raw)
        mol2 = graph_to_rdkit_mol(anums, prn)
        if check_chemical_validity(mol2):
            smi2 = Chem.MolToSmiles(mol2)
            if smi2 and "." not in smi2:
                return mol2, smi2, 2

    # Stage 3
    try:
        rw = Chem.RWMol()
        heavy = [a for a in anums if a != 1]
        for an in heavy:
            rw.AddAtom(Chem.Atom(an))
        for i in range(len(heavy)-1):
            rw.AddBond(i, i+1, Chem.rdchem.BondType.SINGLE)
        m = Chem.RWMol(rw.GetMol())
        Chem.SanitizeMol(m)
        m = Chem.AddHs(m)
        p = AllChem.ETKDGv3(); p.randomSeed = 42
        if AllChem.EmbedMolecule(m, p) >= 0:
            AllChem.MMFFOptimizeMolecule(m)
            m_noh = Chem.RemoveHs(m)
            smi3 = Chem.MolToSmiles(m_noh)
            if smi3 and "." not in smi3:
                return m_noh, smi3, 3
    except Exception:
        pass

    return None, None, None


def atom_stability(mol):
    if mol is None:
        return 0.0, False
    n_stab, all_ok = 0, True
    for atom in mol.GetAtoms():
        exp = EXPECTED_VALENCE.get(atom.GetAtomicNum())
        if exp is not None:
            if atom.GetTotalValence() == exp:
                n_stab += 1
            else:
                all_ok = False
        else:
            n_stab += 1
    return round(n_stab / max(mol.GetNumAtoms(), 1) * 100, 1), all_ok


def run_model(label, model, ref_graphs, schedule, n_samples, num_steps):
    print(f"\n  Running {label} ({n_samples} molecules, {num_steps} steps) ...")
    results = []
    t0 = time.perf_counter()
    for i in range(n_samples):
        seed = SEEDS_BASE + i * 11
        ref  = ref_graphs[i % len(ref_graphs)]
        coords, anums, avg_g = run_reverse_diffusion(model, ref, schedule, num_steps, seed)
        xyz_str = build_xyz(anums, coords)
        mol, smi, stage = bond_inference(anums, coords, xyz_str)
        as_pct, ms = atom_stability(mol)
        mw = round(Descriptors.MolWt(mol), 2) if mol else 0.0
        results.append({
            "seed": seed,
            "valid": mol is not None,
            "connected": mol is not None and smi and "." not in smi,
            "smiles": smi or "",
            "stage": stage,
            "atom_stab": as_pct,
            "mol_stable": ms,
            "mol_wt": mw,
            "avg_gibbs": avg_g,
            "n_atoms": len(anums),
        })
        sym = "OK" if mol else "X"
        g_str = f"{avg_g:.4f}" if avg_g is not None else "N/A"
        print(f"    [{i+1:>3}/{n_samples}] {sym}  stage={stage}  stab={as_pct}%  g={g_str}")
    elapsed = round(time.perf_counter() - t0, 2)

    n = n_samples
    n_valid   = sum(1 for r in results if r["valid"])
    n_conn    = sum(1 for r in results if r["connected"])
    n_mstab   = sum(1 for r in results if r["mol_stable"])
    as_vals   = [r["atom_stab"] for r in results if r["valid"]]
    avg_stab  = round(sum(as_vals)/max(len(as_vals),1), 1)
    smiles    = [r["smiles"] for r in results if r["smiles"]]
    n_unique  = len(set(smiles))
    g_vals    = [r["avg_gibbs"] for r in results if r["avg_gibbs"] is not None]
    avg_gibbs = round(sum(g_vals)/max(len(g_vals),1), 4) if g_vals else None
    stages    = {1:0, 2:0, 3:0, None:0}
    for r in results:
        stages[r["stage"]] = stages.get(r["stage"], 0) + 1
    avg_mw    = round(sum(r["mol_wt"] for r in results)/n, 2)

    summary = {
        "label": label,
        "n_samples": n,
        "num_steps": num_steps,
        "elapsed_s": elapsed,
        "validity_pct":      round(n_valid/n*100, 1),
        "connectivity_pct":  round(n_conn/n*100, 1),
        "mol_stability_pct": round(n_mstab/n*100, 1),
        "avg_atom_stab_pct": avg_stab,
        "unique_smiles":     n_unique,
        "unique_pct":        round(n_unique/max(len(smiles),1)*100, 1),
        "avg_gibbs":         avg_gibbs,
        "avg_mol_wt":        avg_mw,
        "stage_counts":      stages,
        "per_molecule":      results,
    }
    print(f"    Done in {elapsed}s  |  Validity={summary['validity_pct']}%  AvgStab={avg_stab}%")
    return summary


def main():
    print("=" * 62)
    print("  VERSION A vs VERSION B — INFERENCE COMPARISON")
    print("=" * 62)

    print("\n[1/4] Loading QM9 reference graphs ...")
    _, val_loader, _, _ = load_qm9(root=os.path.join(ROOT, "data"), max_samples=200)
    ref_graphs = []
    for batch in val_loader:
        for i in range(batch.num_graphs):
            ref_graphs.append(batch.get_example(i))
    print(f"      {len(ref_graphs)} reference graphs loaded.")

    schedule = DiffusionSchedule(T=T_MAX)

    print("\n[2/4] Loading Version A (trained EGNN + Gibbs head + PINO) ...")
    model_a = load_version_a()
    print("      OK.")

    print("\n[3/4] Loading Version B (baseline EGNN, random weights, no Gibbs head) ...")
    model_b = load_version_b()
    print("      OK (random weights — simulates untrained baseline).")

    print("\n[4/4] Running inference ...")
    summary_a = run_model("Version A (PINO + Gibbs)", model_a, ref_graphs, schedule, N_SAMPLES, NUM_STEPS)
    summary_b = run_model("Version B (Baseline)",     model_b, ref_graphs, schedule, N_SAMPLES, NUM_STEPS)

    out = {
        "version_a": summary_a,
        "version_b": summary_b,
        "n_samples": N_SAMPLES,
        "num_steps": NUM_STEPS,
    }
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json_path = os.path.join(ROOT, "results", "comparison_AB.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved: {json_path}")
    print("=" * 62)


if __name__ == "__main__":
    main()
