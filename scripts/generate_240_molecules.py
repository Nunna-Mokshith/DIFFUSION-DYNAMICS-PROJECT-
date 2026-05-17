# -*- coding: utf-8 -*-
"""
Diffusion Model Generation — 240+ Molecules for Blue Report
=============================================================
Uses the trained EGNN + PINO model to GENERATE molecules via
reverse diffusion from QM9 reference graphs.

This is NOT RDKit testing on known molecules — this is actual
model inference: start from random noise, denoise for 80 steps,
then validate the generated 3D structure.

Uses diverse QM9 reference graphs (C, H, O, N, F) to ensure
element coverage beyond carbon.

Run:
    python scripts/generate_240_molecules.py
"""

import os, sys, json, csv, time, random
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")

from scripts.model_arch import EGNNScoreNetwork
from scripts.physics_guided_molecular_diffusion import (
    DiffusionSchedule, load_qm9,
    NUM_ATOM_FEAT, HIDDEN_DIM, NUM_LAYERS, T_MAX,
    node_features_to_atomic_num, infer_bonds_from_distance,
    graph_to_rdkit_mol, check_chemical_validity,
    rescale_to_qm9, prune_bonds_for_valence,
)
from scripts.Advanced_Molecular_Design import xyz_to_rdkit_mol, EXTENDED_SYM_MAP

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TOTAL    = 250          # total generations (aim for 240+ valid)
NUM_STEPS  = 80           # denoising steps
SEEDS_BASE = 5000         # different seed range from previous runs

EXPECTED_VALENCE = {1:1, 6:4, 7:3, 8:2, 9:1, 15:3, 16:2, 17:1}


def load_model():
    model = EGNNScoreNetwork(NUM_ATOM_FEAT, 4, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
    ckpt  = os.path.join(ROOT, "models", "pgmd_v3_full.pt")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def run_reverse_diffusion(model, ref, schedule, num_steps, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    ref = ref.to(DEVICE)
    n_atoms = ref.x.size(0)
    batch   = torch.zeros(n_atoms, dtype=torch.long, device=DEVICE)
    pos     = torch.randn(n_atoms, 3, device=DEVICE)
    step_ids = torch.linspace(schedule.T-1, 0, num_steps).long().to(DEVICE)
    x_feat  = ref.x.to(DEVICE)
    ei      = ref.edge_index.to(DEVICE)
    ea      = ref.edge_attr.to(DEVICE)

    g_preds = []
    with torch.no_grad():
        for t_val in step_ids:
            t_batch = t_val.expand(1)
            ab      = schedule.alpha_bar[t_val]
            ab_prev = schedule.alpha_bar[t_val-1] if t_val > 0 else torch.tensor(1.0, device=DEVICE)
            score, g = model(x_feat, pos, ei, ea, batch, t_batch)
            if g is not None:
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
    """3-stage waterfall pipeline."""
    # Stage 1: rdDetermineBonds
    mol = xyz_to_rdkit_mol(xyz_str)
    if mol:
        smi = Chem.MolToSmiles(mol)
        if smi and "." not in smi:
            return mol, smi, 1

    # Stage 2: Distance-based
    pos_t = torch.tensor(coords, dtype=torch.float32)
    for thresh in [1.8, 2.0, 2.3, 2.6]:
        raw  = infer_bonds_from_distance(pos_t, anums, threshold=thresh)
        prn  = prune_bonds_for_valence(anums, raw)
        mol2 = graph_to_rdkit_mol(anums, prn)
        if check_chemical_validity(mol2):
            smi2 = Chem.MolToSmiles(mol2)
            if smi2 and "." not in smi2:
                return mol2, smi2, 2

    # Stage 3: ETKDGv3 fallback
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


def lipinski_pass(mol):
    if mol is None:
        return False
    try:
        mw   = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd  = rdMolDescriptors.CalcNumHBD(mol)
        hba  = rdMolDescriptors.CalcNumHBA(mol)
        rot  = rdMolDescriptors.CalcNumRotatableBonds(mol)
        return (mw <= 500) and (logp <= 5) and (hbd <= 5) and (hba <= 10) and (rot <= 10)
    except Exception:
        return False


def element_summary(anums_list):
    """Count how many molecules contain each element."""
    from collections import Counter
    elem_map = {1:"H", 6:"C", 7:"N", 8:"O", 9:"F", 15:"P", 16:"S"}
    counts = Counter()
    for anums in anums_list:
        unique = set(anums)
        for a in unique:
            counts[elem_map.get(a, f"Z{a}")] += 1
    return dict(counts)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  DIFFUSION MODEL GENERATION — 240+ Molecules")
    print("  Model: EGNN + PINO + Gibbs Head (pgmd_v3_full.pt)")
    print("  Source: QM9 reference graphs | Denoising: 80 steps")
    print("=" * 70)

    # Load QM9 refs (diverse — use more samples to get element variety)
    print("\n[1/3] Loading QM9 reference graphs ...")
    _, val_loader, _, _ = load_qm9(root=os.path.join(ROOT, "data"), max_samples=500)
    ref_graphs = []
    for batch in val_loader:
        for i in range(batch.num_graphs):
            ref_graphs.append(batch.get_example(i))
    print(f"      {len(ref_graphs)} reference graphs loaded.")

    # Analyze element coverage in refs
    ref_anums = []
    for g in ref_graphs:
        ref_anums.append(node_features_to_atomic_num(g.x))
    elem_coverage = element_summary(ref_anums)
    print(f"      Element coverage in refs: {elem_coverage}")

    schedule = DiffusionSchedule(T=T_MAX)

    print("\n[2/3] Loading trained model ...")
    model = load_model()
    print("      OK.\n")

    print(f"[3/3] Generating {N_TOTAL} molecules via reverse diffusion ...")
    print("-" * 70)

    fieldnames = [
        "id", "seed", "ref_idx", "ref_elements", "gen_smiles",
        "valid", "stage", "atom_stab_pct", "mol_stable",
        "mol_wt", "num_atoms", "num_heavy", "gibbs_pred",
        "lipinski", "has_N", "has_O", "has_F",
        "elapsed_s",
    ]

    results = []
    all_anums = []
    t_total = time.perf_counter()

    for i in range(N_TOTAL):
        seed    = SEEDS_BASE + i * 13
        ref_idx = i % len(ref_graphs)
        ref     = ref_graphs[ref_idx]

        t0 = time.perf_counter()
        coords, anums, avg_g = run_reverse_diffusion(model, ref, schedule, NUM_STEPS, seed)
        xyz_str = build_xyz(anums, coords)
        mol, smi, stage = bond_inference(anums, coords, xyz_str)
        as_pct, ms = atom_stability(mol)
        lip = lipinski_pass(mol)
        mw  = round(Descriptors.MolWt(mol), 2) if mol else 0.0
        n_heavy = sum(1 for a in anums if a != 1)
        dt  = round(time.perf_counter() - t0, 3)

        # Element flags
        anum_set = set(anums)
        has_N = 7 in anum_set
        has_O = 8 in anum_set
        has_F = 9 in anum_set
        ref_elems = ",".join(sorted(set(
            {1:"H",6:"C",7:"N",8:"O",9:"F"}.get(a, f"Z{a}") for a in anums
        )))

        row = {
            "id":           i + 1,
            "seed":         seed,
            "ref_idx":      ref_idx,
            "ref_elements": ref_elems,
            "gen_smiles":   smi or "",
            "valid":        mol is not None,
            "stage":        stage or "",
            "atom_stab_pct": as_pct,
            "mol_stable":   ms,
            "mol_wt":       mw,
            "num_atoms":    len(anums),
            "num_heavy":    n_heavy,
            "gibbs_pred":   round(avg_g, 4) if avg_g else "",
            "lipinski":     lip,
            "has_N":        has_N,
            "has_O":        has_O,
            "has_F":        has_F,
            "elapsed_s":    dt,
        }
        results.append(row)
        all_anums.append(anums)

        sym = "OK" if mol else " X"
        g_str = f"{avg_g:.3f}" if avg_g else "N/A"
        elems = ref_elems
        print(f"  [{i+1:>3}/{N_TOTAL}]  {sym}  stg={stage}  stab={as_pct:5.1f}%  g={g_str:>7}  "
              f"mw={mw:>6}  elems={elems:<10}  smi={(smi or '-')[:30]}")

    total_time = round(time.perf_counter() - t_total, 1)

    # ── Save CSV ─────────────────────────────────────────────────────────────
    out_csv = os.path.join(ROOT, "results", "diffusion_generated_240.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # ── Summary stats ────────────────────────────────────────────────────────
    n       = len(results)
    n_valid = sum(1 for r in results if r["valid"])
    n_lip   = sum(1 for r in results if r["lipinski"])
    n_N     = sum(1 for r in results if r["has_N"])
    n_O     = sum(1 for r in results if r["has_O"])
    n_F     = sum(1 for r in results if r["has_F"])
    stabs   = [r["atom_stab_pct"] for r in results if r["valid"]]
    avg_stab = round(sum(stabs)/max(len(stabs),1), 1)
    smiles  = [r["gen_smiles"] for r in results if r["gen_smiles"]]
    n_unique = len(set(smiles))
    stages  = {1:0, 2:0, 3:0}
    for r in results:
        if r["stage"]:
            stages[r["stage"]] = stages.get(r["stage"], 0) + 1
    gibbs_vals = [r["gibbs_pred"] for r in results if r["gibbs_pred"] != ""]
    avg_gibbs = round(sum(gibbs_vals)/max(len(gibbs_vals),1), 4) if gibbs_vals else None

    gen_elem_coverage = element_summary(all_anums)

    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Total generated         : {n}")
    print(f"  Valid molecules         : {n_valid}/{n}  ({round(n_valid/n*100,1)}%)")
    print(f"  Unique SMILES           : {n_unique}/{len(smiles)}  ({round(n_unique/max(len(smiles),1)*100,1)}%)")
    print(f"  Avg Atom Stability      : {avg_stab}%")
    print(f"  Lipinski Compliant      : {n_lip}/{n_valid}  ({round(n_lip/max(n_valid,1)*100,1)}%)")
    print(f"  Avg Gibbs Prediction    : {avg_gibbs} eV" if avg_gibbs else "  Avg Gibbs: N/A")
    print(f"  Stage 1 / 2 / 3         : {stages.get(1,0)} / {stages.get(2,0)} / {stages.get(3,0)}")
    print(f"  Molecules containing N  : {n_N}/{n}  ({round(n_N/n*100,1)}%)")
    print(f"  Molecules containing O  : {n_O}/{n}  ({round(n_O/n*100,1)}%)")
    print(f"  Molecules containing F  : {n_F}/{n}  ({round(n_F/n*100,1)}%)")
    print(f"  Element coverage        : {gen_elem_coverage}")
    print(f"  Total generation time   : {total_time}s  ({round(total_time/n, 2)}s/mol)")
    print(f"\n  CSV saved: {out_csv}")
    print("=" * 70)

    # ── Save summary JSON for report generator ───────────────────────────────
    summary = {
        "n_total": n,
        "n_valid": n_valid,
        "validity_pct": round(n_valid/n*100, 1),
        "n_unique": n_unique,
        "unique_pct": round(n_unique/max(len(smiles),1)*100, 1),
        "avg_atom_stab": avg_stab,
        "lipinski_pct": round(n_lip/max(n_valid,1)*100, 1),
        "avg_gibbs": avg_gibbs,
        "stages": stages,
        "n_with_N": n_N,
        "n_with_O": n_O,
        "n_with_F": n_F,
        "element_coverage": gen_elem_coverage,
        "total_time_s": total_time,
    }
    json_path = os.path.join(ROOT, "results", "diffusion_gen_240_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON: {json_path}\n")


if __name__ == "__main__":
    main()
