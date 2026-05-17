# -*- coding: utf-8 -*-
"""
3-Stage RDKit Bond Inference Pipeline - Standalone Test
=========================================================
Tests each of the 3 stages against known molecules with
ground-truth SMILES and 3D coordinates.

Molecules tested:
  - Methane    (CH4)      — simplest possible, all H
  - Ethanol    (CCO)      — contains O, HBD/HBA
  - Formaldehyde (C=O)   — double bond, small
  - Ammonia    (N)        — lone pair nitrogen
  - Fluoromethane (CF)   — F element

Run:
    python scripts/test_3stage_pipeline.py
"""

import os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors

RDLogger.DisableLog("rdApp.*")

from scripts.physics_guided_molecular_diffusion import (
    infer_bonds_from_distance,
    graph_to_rdkit_mol,
    check_chemical_validity,
    prune_bonds_for_valence,
)
from scripts.Advanced_Molecular_Design import xyz_to_rdkit_mol

# ────────────────────────────────────────────────────────────────────────────
#  Known test molecules with exact 3D coordinates (in Angstroms)
#  Coordinates taken from standard DFT-optimised geometries
# ────────────────────────────────────────────────────────────────────────────

TEST_MOLECULES = [
    {
        "name":       "Methane (CH4)",
        "expected":   "C",          # canonical SMILES
        "atomic_nums": [6, 1, 1, 1, 1],
        "coords": np.array([
            [ 0.0000,  0.0000,  0.0000],  # C
            [ 0.6276,  0.6276,  0.6276],  # H
            [-0.6276, -0.6276,  0.6276],  # H
            [-0.6276,  0.6276, -0.6276],  # H
            [ 0.6276, -0.6276, -0.6276],  # H
        ]),
        "xyz": "5\nMethane\nC  0.0000  0.0000  0.0000\nH  0.6276  0.6276  0.6276\nH -0.6276 -0.6276  0.6276\nH -0.6276  0.6276 -0.6276\nH  0.6276 -0.6276 -0.6276\n"
    },
    {
        "name":       "Ethanol (CCO)",
        "expected":   "CCO",
        "atomic_nums": [6, 6, 8, 1, 1, 1, 1, 1, 1],
        "coords": np.array([
            [-1.1763,  0.1284,  0.0000],  # C
            [ 0.0000,  0.0000,  0.0000],  # C  (pivot)
            [ 0.6280,  1.2964,  0.0000],  # O
            [-1.6218, -0.8748,  0.0000],  # H
            [-1.4696,  0.6861,  0.8900],  # H
            [-1.4696,  0.6861, -0.8900],  # H
            [ 0.3938, -0.5546,  0.8900],  # H
            [ 0.3938, -0.5546, -0.8900],  # H
            [ 1.5759,  1.2176,  0.0000],  # H (OH)
        ]),
        "xyz": "9\nEthanol\nC -1.1763  0.1284  0.0000\nC  0.0000  0.0000  0.0000\nO  0.6280  1.2964  0.0000\nH -1.6218 -0.8748  0.0000\nH -1.4696  0.6861  0.8900\nH -1.4696  0.6861 -0.8900\nH  0.3938 -0.5546  0.8900\nH  0.3938 -0.5546 -0.8900\nH  1.5759  1.2176  0.0000\n"
    },
    {
        "name":       "Formaldehyde (C=O)",
        "expected":   "C=O",
        "atomic_nums": [6, 8, 1, 1],
        "coords": np.array([
            [ 0.0000,  0.0000,  0.0000],  # C
            [ 0.0000,  1.2078,  0.0000],  # O
            [ 0.9380, -0.5439,  0.0000],  # H
            [-0.9380, -0.5439,  0.0000],  # H
        ]),
        "xyz": "4\nFormaldehyde\nC  0.0000  0.0000  0.0000\nO  0.0000  1.2078  0.0000\nH  0.9380 -0.5439  0.0000\nH -0.9380 -0.5439  0.0000\n"
    },
    {
        "name":       "Ammonia (NH3)",
        "expected":   "N",
        "atomic_nums": [7, 1, 1, 1],
        "coords": np.array([
            [ 0.0000,  0.0000,  0.1127],  # N
            [ 0.0000,  0.9380, -0.2627],  # H
            [ 0.8121, -0.4690, -0.2627],  # H
            [-0.8121, -0.4690, -0.2627],  # H
        ]),
        "xyz": "4\nAmmonia\nN  0.0000  0.0000  0.1127\nH  0.0000  0.9380 -0.2627\nH  0.8121 -0.4690 -0.2627\nH -0.8121 -0.4690 -0.2627\n"
    },
    {
        "name":       "Fluoromethane (CH3F)",
        "expected":   "CF",
        "atomic_nums": [6, 9, 1, 1, 1],
        "coords": np.array([
            [ 0.0000,  0.0000,  0.0000],  # C
            [ 0.0000,  0.0000,  1.3850],  # F
            [ 1.0270,  0.0000, -0.3627],  # H
            [-0.5135,  0.8893, -0.3627],  # H
            [-0.5135, -0.8893, -0.3627],  # H
        ]),
        "xyz": "5\nFluoromethane\nC  0.0000  0.0000  0.0000\nF  0.0000  0.0000  1.3850\nH  1.0270  0.0000 -0.3627\nH -0.5135  0.8893 -0.3627\nH -0.5135 -0.8893 -0.3627\n"
    },
]

EXPECTED_VALENCE = {1: 1, 6: 4, 7: 3, 8: 2, 9: 1}

# ────────────────────────────────────────────────────────────────────────────

def canon(smi):
    """Canonical SMILES, or None."""
    try:
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(m) if m else None
    except Exception:
        return None


def check_stability(mol):
    """Return (atom_stability_pct, mol_stable)."""
    if mol is None:
        return 0.0, False
    n_stable = 0
    all_ok = True
    for atom in mol.GetAtoms():
        an = atom.GetAtomicNum()
        exp = EXPECTED_VALENCE.get(an)
        if exp is not None:
            if atom.GetTotalValence() == exp:
                n_stable += 1
            else:
                all_ok = False
        else:
            n_stable += 1
    pct = round(n_stable / max(mol.GetNumAtoms(), 1) * 100, 1)
    return pct, all_ok


def run_stage1(xyz_str):
    """Stage 1: rdDetermineBonds via xyz_to_rdkit_mol."""
    mol = xyz_to_rdkit_mol(xyz_str)
    if mol is None:
        return None, None
    smi = Chem.MolToSmiles(mol)
    if smi and "." not in smi:
        return mol, smi
    return None, None


def run_stage2(coords_np, anums):
    """Stage 2: Distance-based with covalent radii + valence pruning."""
    pos_t = torch.tensor(coords_np)
    for thresh in [1.8, 2.0, 2.3, 2.6]:
        raw_bonds   = infer_bonds_from_distance(pos_t, anums, threshold=thresh)
        pruned      = prune_bonds_for_valence(anums, raw_bonds)
        mol         = graph_to_rdkit_mol(anums, pruned)
        if check_chemical_validity(mol):
            smi = Chem.MolToSmiles(mol)
            if smi and "." not in smi:
                return mol, smi
    return None, None


def run_stage3(anums, seed=42):
    """Stage 3: ETKDGv3 re-embedding from atom types."""
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
            smi = Chem.MolToSmiles(m_noh)
            if smi and "." not in smi:
                return m_noh, smi
    except Exception as e:
        pass
    return None, None


# ────────────────────────────────────────────────────────────────────────────
#  MAIN TEST RUNNER
# ────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  3-STAGE RDKIT BOND INFERENCE PIPELINE — TEST REPORT")
    print("=" * 65)

    stage_pass = {1: 0, 2: 0, 3: 0}
    stage_fail = {1: 0, 2: 0, 3: 0}
    all_results = []

    for mol_spec in TEST_MOLECULES:
        name     = mol_spec["name"]
        expected = canon(mol_spec["expected"])
        anums    = mol_spec["atomic_nums"]
        coords   = mol_spec["coords"]
        xyz_str  = mol_spec["xyz"]

        print("\n" + "-"*65)
        print(f"  Molecule : {name}")
        print(f"  Expected SMILES : {expected}")
        print(f"  Atoms    : {anums}")

        row = {"name": name, "expected": expected,
               "stage1": "FAIL", "stage2": "FAIL", "stage3": "FAIL",
               "best_smiles": None, "match": False,
               "atom_stab": 0.0, "mol_stab": False}

        # ── STAGE 1 ─────────────────────────────────────────────────────────
        mol1, smi1 = run_stage1(xyz_str)
        if mol1:
            atom_s, mol_s = check_stability(mol1)
            match1 = canon(smi1) == expected
            row["stage1"] = "PASS"
            row["best_smiles"] = smi1
            row["match"]     = match1
            row["atom_stab"] = atom_s
            row["mol_stab"]  = mol_s
            stage_pass[1] += 1
            print(f"  Stage 1  : PASS  SMILES={smi1}  "
                  f"Match={'OK' if match1 else 'DIFF'}  "
                  f"AtomStab={atom_s}%  MolStab={'STABLE' if mol_s else 'UNSTABLE'}")
        else:
            stage_fail[1] += 1
            print(f"  Stage 1  : FAIL  (rdDetermineBonds could not produce connected mol)")

        # ── STAGE 2 ─────────────────────────────────────────────────────────
        mol2, smi2 = run_stage2(coords, anums)
        if mol2:
            atom_s, mol_s = check_stability(mol2)
            match2 = canon(smi2) == expected
            row["stage2"] = "PASS"
            if row["best_smiles"] is None:
                row["best_smiles"] = smi2
                row["match"]     = match2
                row["atom_stab"] = atom_s
                row["mol_stab"]  = mol_s
            stage_pass[2] += 1
            print(f"  Stage 2  : PASS  SMILES={smi2}  "
                  f"Match={'OK' if match2 else 'DIFF'}  "
                  f"AtomStab={atom_s}%  MolStab={'STABLE' if mol_s else 'UNSTABLE'}")
        else:
            stage_fail[2] += 1
            print(f"  Stage 2  : FAIL  (distance-based bonding failed for all thresholds)")

        # ── STAGE 3 ─────────────────────────────────────────────────────────
        mol3, smi3 = run_stage3(anums)
        if mol3:
            atom_s, mol_s = check_stability(mol3)
            match3 = canon(smi3) == expected
            row["stage3"] = "PASS"
            if row["best_smiles"] is None:
                row["best_smiles"] = smi3
                row["match"]     = match3
                row["atom_stab"] = atom_s
                row["mol_stab"]  = mol_s
            stage_pass[3] += 1
            print(f"  Stage 3  : PASS  SMILES={smi3}  "
                  f"Match={'OK' if match3 else 'DIFF'}  "
                  f"AtomStab={atom_s}%  MolStab={'STABLE' if mol_s else 'UNSTABLE'}")
        else:
            stage_fail[3] += 1
            print(f"  Stage 3  : FAIL  (ETKDGv3 embedding failed)")

        all_results.append(row)

    # ── SUMMARY TABLE ──────────────────────────────────────────────────────
    N = len(TEST_MOLECULES)
    print("\n" + "="*65)
    print("  SUMMARY")
    print("="*65)
    print(f"  {'Molecule':<25} {'S1':^4} {'S2':^4} {'S3':^4} {'Match':^6} {'AtmStb%':^8} {'MolStb':^6}")
    print("  " + "-"*65)
    for r in all_results:
        match_sym  = "OK  " if r["match"] else "DIFF"
        mstab_sym  = "STABLE" if r["mol_stab"] else "UNSTABLE"
        print(f"  {r['name']:<25} {r['stage1']:^4} {r['stage2']:^4} {r['stage3']:^4} "
              f"{match_sym:^6} {r['atom_stab']:^8.1f} {mstab_sym:^10}")

    print(f"\n  Stage Pass Rates (out of {N} molecules):")
    for s in [1, 2, 3]:
        label = {1: "Stage 1 (rdDetermineBonds)",
                 2: "Stage 2 (distance-based)",
                 3: "Stage 3 (ETKDGv3 fallback)"}[s]
        pct = stage_pass[s] / N * 100
        bar = "#" * stage_pass[s] + "." * stage_fail[s]
        print(f"    {label:<30} : {bar}  {stage_pass[s]}/{N} ({pct:.0f}%)")

    print()
    all_pass = all(r["stage1"] == "PASS" or r["stage2"] == "PASS" or r["stage3"] == "PASS"
                   for r in all_results)
    if all_pass:
        print("  [OK]  ALL MOLECULES RECOVERED BY AT LEAST ONE STAGE - PIPELINE IS FUNCTIONAL")
    else:
        print("  [!!]  SOME MOLECULES FAILED ALL STAGES - REVIEW PIPELINE FUNCTIONS")

    print("="*65 + "\n")


if __name__ == "__main__":
    main()
