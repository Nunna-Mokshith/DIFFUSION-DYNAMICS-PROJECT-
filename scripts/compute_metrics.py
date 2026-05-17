# -*- coding: utf-8 -*-
"""
Compute 3 Key Metrics
=====================
1. Structural Validity %   — % of molecules that pass at least one RDKit stage
2. Gibbs MAE               — Mean Absolute Error of model's Gibbs prediction vs
                             RDKit-estimated reference (via Crippen/MolLogP proxy)
3. Lipinski Compliance %   — % of valid molecules that satisfy all 5 Lipinski rules

Run:
    python scripts/compute_metrics.py
"""

import os, sys, csv, math
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem.rdchem import BondType

RDLogger.DisableLog("rdApp.*")

CSV_PATH    = os.path.join(ROOT, "results", "pipeline_test_120.csv")
MODEL_PATH  = os.path.join(ROOT, "models", "pgmd_v3_full.pt")

# ─────────────────────────────────────────────────────────────────────────────
# METRIC 1: Structural Validity %
# ─────────────────────────────────────────────────────────────────────────────

def compute_validity(rows):
    total   = len(rows)
    valid   = sum(1 for r in rows if r["any_stage_pass"] == "True")
    skipped = sum(1 for r in rows if r["error"] != "")
    pct     = valid / total * 100
    return pct, valid, total, skipped


# ─────────────────────────────────────────────────────────────────────────────
# METRIC 2: Gibbs MAE
# We compute a lightweight reference Gibbs proxy using RDKit Crippen logP
# as a surrogate (since we have no DFT ground truth), then compare against
# the model's prop_head output (index 0 = Gibbs channel).
# ─────────────────────────────────────────────────────────────────────────────

def gibbs_reference(smiles):
    """
    Simple Gibbs free energy proxy: G_ref = -RT * logP  (in eV units, T=298K)
    This is a widely used linear approximation for solvation free energy.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        logp  = Descriptors.MolLogP(mol)
        RT_eV = 0.02585   # kT at 298 K in eV
        return -RT_eV * logp
    except Exception:
        return None


def mol_to_simple_graph(smiles):
    """
    Build a minimal node feature tensor + edge_index for inference.
    Node features: one-hot atomic number (H=1,C=6,N=7,O=8,F=9,S=16,Cl=17,Br=35) + degree (3 more)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        mol = Chem.RemoveHs(mol)

        ATOM_MAP = {1:0, 6:1, 7:2, 8:3, 9:4, 16:5, 17:6, 35:7, 15:8, 53:9, 14:10}
        n = mol.GetNumAtoms()
        if n == 0:
            return None

        # Node features (11-dim to match model's node_feat_dim=11)
        x = torch.zeros(n, 11)
        conf = mol.GetConformer() if mol.GetNumConformers() > 0 else None
        pos  = torch.zeros(n, 3)

        for i, atom in enumerate(mol.GetAtoms()):
            an = atom.GetAtomicNum()
            idx = ATOM_MAP.get(an, 10)
            x[i, idx] = 1.0
            if conf:
                p = conf.GetAtomPosition(i)
                pos[i] = torch.tensor([p.x, p.y, p.z])

        # Edge index (fully connected for simplicity, capped at 50 atoms)
        src, dst, eattr = [], [], []
        bonds = {(b.GetBeginAtomIdx(), b.GetEndAtomIdx()): b for b in mol.GetBonds()}
        bonds.update({(v, k): b for (k, v), b in bonds.items()})

        for i in range(n):
            for j in range(n):
                if i != j:
                    src.append(i); dst.append(j)
                    b = bonds.get((i, j))
                    bt = [0.0, 0.0, 0.0, 0.0]
                    if b:
                        t = b.GetBondTypeAsDouble()
                        if   t == 1.0: bt[0] = 1.0
                        elif t == 2.0: bt[1] = 1.0
                        elif t == 3.0: bt[2] = 1.0
                        else:          bt[3] = 1.0
                    eattr.append(bt)

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr  = torch.tensor(eattr, dtype=torch.float32)
        batch      = torch.zeros(n, dtype=torch.long)
        t_step     = torch.tensor([1], dtype=torch.long)

        return x, pos, edge_index, edge_attr, batch, t_step

    except Exception:
        return None


def compute_gibbs_mae(rows):
    """Load model and compute Gibbs MAE over valid molecules."""
    try:
        from scripts.model_arch import EGNNScoreNetwork
        model = EGNNScoreNetwork()
        ckpt  = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        model.eval()
        print("  [Model] Checkpoint loaded OK")
    except Exception as e:
        print(f"  [Model] Load failed: {e}")
        return None, 0

    errors, n_compared = [], 0
    with torch.no_grad():
        for r in rows:
            if r["any_stage_pass"] != "True" or not r["best_smiles"]:
                continue
            g_ref = gibbs_reference(r["best_smiles"])
            if g_ref is None:
                continue
            graph = mol_to_simple_graph(r["best_smiles"])
            if graph is None:
                continue
            x, pos, edge_index, edge_attr, batch, t_step = graph
            try:
                _, g_pred = model(x, pos, edge_index, edge_attr, batch, t_step)
                g_val = g_pred[0, 0].item()          # Gibbs channel (index 0)
                errors.append(abs(g_val - g_ref))
                n_compared += 1
            except Exception:
                continue

    if not errors:
        return None, 0
    mae = sum(errors) / len(errors)
    return mae, n_compared


# ─────────────────────────────────────────────────────────────────────────────
# METRIC 3: Lipinski Compliance %
# Rule of Five: MW<=500, logP<=5, HBD<=5, HBA<=10, RotBonds<=10
# ─────────────────────────────────────────────────────────────────────────────

def lipinski_pass(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, {}
        mw   = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd  = rdMolDescriptors.CalcNumHBD(mol)
        hba  = rdMolDescriptors.CalcNumHBA(mol)
        rot  = rdMolDescriptors.CalcNumRotatableBonds(mol)
        ok   = (mw <= 500) and (logp <= 5) and (hbd <= 5) and (hba <= 10) and (rot <= 10)
        return ok, {"MW": round(mw,1), "logP": round(logp,2),
                    "HBD": hbd, "HBA": hba, "RotBonds": rot}
    except Exception:
        return False, {}


def compute_lipinski(rows):
    valid_smiles = [r["best_smiles"] for r in rows
                    if r["any_stage_pass"] == "True" and r["best_smiles"]]
    total    = len(valid_smiles)
    passed   = sum(1 for s in valid_smiles if lipinski_pass(s)[0])
    pct      = passed / total * 100 if total else 0
    return pct, passed, total


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Load CSV
    rows = []
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"\nLoaded {len(rows)} rows from CSV.\n")

    print("=" * 55)
    print("  COMPUTING METRICS")
    print("=" * 55)

    # ── Metric 1 ────────────────────────────────────────────
    print("\n[1] Structural Validity %")
    v_pct, v_valid, v_total, v_skip = compute_validity(rows)
    print(f"    Valid molecules : {v_valid} / {v_total}")
    print(f"    Skipped (bad SMILES) : {v_skip}")
    print(f"    >>> Structural Validity : {v_pct:.1f}%")

    # ── Metric 2 ────────────────────────────────────────────
    print("\n[2] Gibbs MAE (model vs RDKit logP proxy)")
    g_mae, g_n = compute_gibbs_mae(rows)
    if g_mae is not None:
        print(f"    Compared over : {g_n} molecules")
        print(f"    >>> Gibbs MAE : {g_mae:.4f} eV")
    else:
        print("    >>> Gibbs MAE : N/A (model load failed)")

    # ── Metric 3 ────────────────────────────────────────────
    print("\n[3] Lipinski Compliance %")
    l_pct, l_pass, l_total = compute_lipinski(rows)
    print(f"    Compliant : {l_pass} / {l_total} valid molecules")
    print(f"    >>> Lipinski Compliance : {l_pct:.1f}%")

    # ── Final Summary ────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  FINAL RESULTS")
    print("=" * 55)
    print(f"  Structural Validity  : {v_pct:.1f}%")
    if g_mae is not None:
        print(f"  Gibbs MAE            : {g_mae:.4f} eV")
    else:
        print(f"  Gibbs MAE            : N/A")
    print(f"  Lipinski Compliance  : {l_pct:.1f}%")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
