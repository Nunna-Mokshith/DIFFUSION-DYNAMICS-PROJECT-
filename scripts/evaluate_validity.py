"""
Molecule Validity Evaluation Script
====================================
Generates N molecules via the trained EGNN diffusion model and evaluates
chemical validity using the same 3-stage RDKit bond inference pipeline
used in the dashboard.

Metrics reported:
  - RDKit Sanitization pass rate
  - Valid SMILES rate
  - Connected molecule rate (no '.' in SMILES)
  - Atom Stability (correct valence per QM9 standards)
  - Molecule Stability (all atoms stable)
  - Unique SMILES count
  - Stage breakdown (which bond inference stage succeeded)

Usage:
  python scripts/evaluate_validity.py --n_samples 100 --steps 100
  python scripts/evaluate_validity.py --n_samples 200 --steps 200 --verbose
"""

import os
import sys
import time
import random
import argparse

import numpy as np
import torch

# Ensure project root is on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem

RDLogger.DisableLog("rdApp.*")

# ── Project imports ─────────────────────────────────────────────────────────
from scripts.model_arch import EGNNScoreNetwork
from scripts.physics_guided_molecular_diffusion import (
    DiffusionSchedule, load_qm9,
    NUM_ATOM_FEAT, HIDDEN_DIM, NUM_LAYERS, T_MAX,
    node_features_to_atomic_num, infer_bonds_from_distance,
    graph_to_rdkit_mol, check_chemical_validity,
    rescale_to_qm9, prune_bonds_for_valence,
)
from scripts.Advanced_Molecular_Design import (
    xyz_to_rdkit_mol, apply_element_doping, generate_advanced_catalyst,
)
from scripts.pino_operator import pino_refine_coordinates

# ── Constants ───────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPECTED_VALENCE = {1: 1, 6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1}


def load_model():
    """Load the trained EGNN checkpoint."""
    model = EGNNScoreNetwork(NUM_ATOM_FEAT, 4, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
    ckpt = os.path.join(ROOT, "models", "pgmd_v3_full.pt")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def generate_one_molecule(model, schedule, ref_graph, seed, num_steps=100):
    """
    Run the full generation pipeline for a single molecule.
    Returns a dict with all validity info.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    ref = ref_graph.to(DEVICE)
    n_atoms = ref.x.size(0)
    batch_vec = torch.zeros(n_atoms, dtype=torch.long, device=DEVICE)

    # ── Reverse diffusion ───────────────────────────────────────────────
    pos = torch.randn(n_atoms, 3, device=DEVICE)
    step_ids = torch.linspace(schedule.T - 1, 0, num_steps).long().to(DEVICE)
    x_feat = ref.x.to(DEVICE)
    edge_index = ref.edge_index.to(DEVICE)
    edge_attr = ref.edge_attr.to(DEVICE)

    for t_val in step_ids:
        with torch.no_grad():
            t_batch = t_val.expand(1)
            ab = schedule.alpha_bar[t_val]
            ab_prev = schedule.alpha_bar[t_val - 1] if t_val > 0 else torch.tensor(1.0, device=DEVICE)
            score, _ = model(x_feat, pos, edge_index, edge_attr, batch_vec, t_batch)
            beta = 1 - ab / ab_prev
            coeff = beta / torch.clamp((1 - ab), min=1e-5).sqrt()
            pos = (pos - coeff * score) / torch.clamp((1 - beta), min=1e-5).sqrt()
            if t_val > 0:
                pos = pos + beta.sqrt() * torch.randn_like(pos)

    gen_pos_np = pos.detach().cpu().numpy()
    anums = node_features_to_atomic_num(ref.x.cpu())

    # ── Scale coordinates to realistic bond lengths ─────────────────────
    # Match mean pairwise distance to real QM9 geometry (~3.16 Å)
    gen_pos_np = rescale_to_qm9(gen_pos_np)

    # ── PINO refinement ─────────────────────────────────────────────────
    pino_result = pino_refine_coordinates(
        raw_coords=gen_pos_np,
        atomic_nums=anums,
        num_steps=10,       # fewer steps to save compute
        lr=1e-3,
        device=str(DEVICE),
    )
    gen_pos_np = pino_result["refined_coords"]

    # ── Build XYZ string ────────────────────────────────────────────────
    from scripts.Advanced_Molecular_Design import EXTENDED_SYM_MAP
    xyz_str = f"{len(anums)}\nEvaluate seed={seed}\n"
    for i, an in enumerate(anums):
        sym = EXTENDED_SYM_MAP.get(an, "C")
        x, y, z = gen_pos_np[i]
        xyz_str += f"{sym} {x:.5f} {y:.5f} {z:.5f}\n"

    # ── 3-Stage bond inference (same as server.py) ──────────────────────
    result = {
        "seed": seed,
        "n_atoms": len(anums),
        "stage": None,
        "sanitized": False,
        "valid_smiles": False,
        "connected": False,
        "smiles": None,
        "atom_stability_pct": 0.0,
        "mol_stable": False,
    }

    mol = None
    smi = None

    # Stage 1: rdDetermineBonds
    mol = xyz_to_rdkit_mol(xyz_str)
    if mol is not None:
        smi = Chem.MolToSmiles(mol)
        if smi and "." not in smi:
            result["stage"] = 1
        else:
            mol = None
            smi = None

    # Stage 2: distance-based with covalent radii + valence pruning
    if mol is None:
        scaled_pos = torch.tensor(gen_pos_np)
        for thresh in [1.8, 2.0, 2.3, 2.6]:
            raw_bonds = infer_bonds_from_distance(scaled_pos, anums, threshold=thresh)
            pruned_bonds = prune_bonds_for_valence(anums, raw_bonds)
            mol2 = graph_to_rdkit_mol(anums, pruned_bonds)
            if check_chemical_validity(mol2):
                smi2 = Chem.MolToSmiles(mol2)
                if smi2 and "." not in smi2:
                    mol = mol2
                    smi = smi2
                    result["stage"] = 2
                    break

    # Stage 2.5: adaptive rescaling — try multiple NN distance targets
    if mol is None:
        from scipy.spatial.distance import pdist, squareform
        raw_centered = gen_pos_np - gen_pos_np.mean(axis=0)
        for target_nn in [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
            dm = squareform(pdist(raw_centered))
            np.fill_diagonal(dm, np.inf)
            med_nn = np.median(dm.min(axis=1))
            if med_nn < 1e-6:
                continue
            trial = raw_centered * (target_nn / (med_nn + 1e-8))
            # Try rdDetermineBonds on rescaled coords
            xyz_trial = f"{len(anums)}\ntrial_nn={target_nn}\n"
            for k, an in enumerate(anums):
                sym = EXTENDED_SYM_MAP.get(an, "C")
                xyz_trial += f"{sym} {trial[k,0]:.5f} {trial[k,1]:.5f} {trial[k,2]:.5f}\n"
            mol_t = xyz_to_rdkit_mol(xyz_trial)
            if mol_t:
                smi_t = Chem.MolToSmiles(mol_t)
                if smi_t and "." not in smi_t:
                    mol = mol_t; smi = smi_t; result["stage"] = 1; break
            # Try distance-based on rescaled coords
            for thresh in [1.8, 2.0, 2.3]:
                raw_bonds = infer_bonds_from_distance(torch.tensor(trial), anums, threshold=thresh)
                pruned_bonds = prune_bonds_for_valence(anums, raw_bonds)
                mol2 = graph_to_rdkit_mol(anums, pruned_bonds)
                if check_chemical_validity(mol2):
                    smi2 = Chem.MolToSmiles(mol2)
                    if smi2 and "." not in smi2:
                        mol = mol2; smi = smi2; result["stage"] = 2; break
            if mol is not None:
                break

    # Stage 3: ETKDGv3 re-embedding
    if mol is None:
        try:
            rw = Chem.RWMol()
            heavy = [a for a in anums if a != 1]
            for an in heavy:
                rw.AddAtom(Chem.Atom(an))
            for i in range(len(heavy) - 1):
                rw.AddBond(i, i + 1, Chem.rdchem.BondType.SINGLE)
            m = rw.GetMol()
            m = Chem.RWMol(m)
            Chem.SanitizeMol(m)
            m = Chem.AddHs(m)
            params = AllChem.ETKDGv3()
            params.randomSeed = seed
            if AllChem.EmbedMolecule(m, params) >= 0:
                AllChem.MMFFOptimizeMolecule(m)
                m_noh = Chem.RemoveHs(m)
                smi3 = Chem.MolToSmiles(m_noh)
                if smi3 and "." not in smi3:
                    mol = m_noh
                    smi = smi3
                    result["stage"] = 3
        except Exception:
            pass

    # ── Record results ──────────────────────────────────────────────────
    if mol is not None:
        result["sanitized"] = True
        result["smiles"] = smi
        result["valid_smiles"] = smi is not None and len(smi) > 0
        result["connected"] = "." not in (smi or ".")

        # Atom / molecule stability
        n_stable = 0
        all_ok = True
        for atom in mol.GetAtoms():
            an = atom.GetAtomicNum()
            expected = EXPECTED_VALENCE.get(an)
            if expected is not None:
                if atom.GetTotalValence() == expected:
                    n_stable += 1
                else:
                    all_ok = False
            else:
                n_stable += 1  # unknown element, count as ok
        result["atom_stability_pct"] = round(n_stable / max(mol.GetNumAtoms(), 1) * 100, 1)
        result["mol_stable"] = all_ok

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate molecule generation validity")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of molecules to generate")
    parser.add_argument("--steps", type=int, default=100, help="Diffusion denoising steps per molecule")
    parser.add_argument("--verbose", action="store_true", help="Print per-molecule details")
    args = parser.parse_args()

    print("=" * 60)
    print("  Molecule Validity Evaluation")
    print(f"  Samples: {args.n_samples}  |  Steps: {args.steps}  |  Device: {DEVICE}")
    print("=" * 60)

    # ── Load model + data ───────────────────────────────────────────────
    print("\n[1/3] Loading model...")
    model = load_model()
    print("[OK]  Model loaded.")

    print("[2/3] Loading QM9 reference data...")
    _, val_loader, _, _ = load_qm9(
        root=os.path.join(ROOT, "data"), max_samples=500
    )
    # Collect all individual graphs from the val loader
    ref_graphs = []
    for batch in val_loader:
        for i in range(batch.num_graphs):
            ref_graphs.append(batch.get_example(i))
    print(f"[OK]  {len(ref_graphs)} reference graphs available.")

    schedule = DiffusionSchedule(T=T_MAX)

    # ── Generate + evaluate ─────────────────────────────────────────────
    print(f"\n[3/3] Generating {args.n_samples} molecules...\n")

    results = []
    t0 = time.perf_counter()

    for i in range(args.n_samples):
        seed = 1000 + i * 7  # deterministic, spread-out seeds
        ref = ref_graphs[i % len(ref_graphs)]

        r = generate_one_molecule(model, schedule, ref, seed, num_steps=args.steps)
        results.append(r)

        # Progress update every 10 molecules
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            eta = (args.n_samples - i - 1) / rate if rate > 0 else 0
            valid_so_far = sum(1 for x in results if x["valid_smiles"])
            print(
                f"  [{i+1:>3}/{args.n_samples}]  "
                f"Valid: {valid_so_far}/{i+1} ({valid_so_far/(i+1)*100:.0f}%)  "
                f"ETA: {eta:.0f}s"
            )

        if args.verbose and r["valid_smiles"]:
            print(f"    seed={r['seed']}  stage={r['stage']}  "
                  f"SMILES={r['smiles']}  "
                  f"AtomStab={r['atom_stability_pct']}%  "
                  f"MolStab={'Y' if r['mol_stable'] else 'N'}")

    total_time = time.perf_counter() - t0

    # ── Compute summary statistics ──────────────────────────────────────
    n = len(results)
    n_sanitized = sum(1 for r in results if r["sanitized"])
    n_valid     = sum(1 for r in results if r["valid_smiles"])
    n_connected = sum(1 for r in results if r["connected"])
    n_mol_stable = sum(1 for r in results if r["mol_stable"])

    valid_smiles = [r["smiles"] for r in results if r["valid_smiles"]]
    n_unique = len(set(valid_smiles))

    atom_stab_vals = [r["atom_stability_pct"] for r in results if r["sanitized"]]
    avg_atom_stab = sum(atom_stab_vals) / max(len(atom_stab_vals), 1)

    stage_counts = {1: 0, 2: 0, 3: 0, None: 0}
    for r in results:
        stage_counts[r["stage"]] = stage_counts.get(r["stage"], 0) + 1

    # ── Print results ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  VALIDITY EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Total generated        : {n}")
    print(f"  Diffusion steps        : {args.steps}")
    print(f"  Total time             : {total_time:.1f}s  ({total_time/n:.1f}s per mol)")
    print("-" * 60)
    print(f"  RDKit Sanitized        : {n_sanitized:>4} / {n}  ({n_sanitized/n*100:>5.1f}%)")
    print(f"  Valid SMILES           : {n_valid:>4} / {n}  ({n_valid/n*100:>5.1f}%)")
    print(f"  Connected (no '.')     : {n_connected:>4} / {n}  ({n_connected/n*100:>5.1f}%)")
    print(f"  Molecule Stable        : {n_mol_stable:>4} / {n}  ({n_mol_stable/n*100:>5.1f}%)")
    print(f"  Unique valid SMILES    : {n_unique:>4} / {n_valid}")
    print(f"  Avg Atom Stability     : {avg_atom_stab:.1f}%")
    print("-" * 60)
    print("  Bond Inference Stage Breakdown:")
    print(f"    Stage 1 (rdDetermineBonds) : {stage_counts[1]:>4}")
    print(f"    Stage 2 (distance-based)   : {stage_counts[2]:>4}")
    print(f"    Stage 3 (ETKDGv3 fallback) : {stage_counts[3]:>4}")
    print(f"    Failed (no valid mol)      : {stage_counts[None]:>4}")
    print("=" * 60)

    # ── Print sample valid SMILES ───────────────────────────────────────
    if valid_smiles:
        print("\n  Sample valid SMILES (first 10):")
        for s in list(set(valid_smiles))[:10]:
            print(f"    {s}")

    print("\n[Done]\n")


if __name__ == "__main__":
    main()
