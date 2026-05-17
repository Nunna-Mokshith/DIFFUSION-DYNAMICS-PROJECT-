# -*- coding: utf-8 -*-
"""
100 Additional Diverse Molecules — QM9 Element Coverage
========================================================
Focuses on N, O, F-heavy and multi-heteroatom molecules
to complement the existing carbon-dominated test set.

QM9 elements: C, H, O, N, F (up to 9 heavy atoms)

Run:
    python scripts/test_100_diverse.py
"""

import os, sys, csv, time
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

EXPECTED_VALENCE = {1: 1, 6: 4, 7: 3, 8: 2, 9: 1}

# ── 100 diverse QM9-style SMILES ────────────────────────────────────────────
# Emphasis: nitrogen-rich, oxygen-rich, fluorine-containing, mixed heteroatoms
DIVERSE_SMILES = [
    # ── Nitrogen-rich ────────────────────────────────────────
    "N",                          # Ammonia
    "NN",                         # Hydrazine
    "NNN",                        # Triazanamine
    "N=N",                        # Diazene
    "CN=O",                       # Formaldoxime
    "NC=O",                       # Formamide
    "NC(N)=O",                    # Urea
    "NC(=O)N",                    # Urea (alt)
    "N#N",                        # Dinitrogen
    "C(=O)N",                     # Formamide (alt)
    "[NH3]",                      # Ammonia explicit
    "NCC",                        # Ethylamine
    "NCCN",                       # Ethylenediamine
    "NCCCN",                      # 1,3-diaminopropane
    "NC(=O)CC",                   # Propanamide
    "NC(=O)CCC",                  # Butanamide
    "NC(=O)C(=O)N",               # Oxamide
    "c1nncnn1",                   # Triazine (sym)
    "C1=NN=CN1",                  # 1H-1,2,3-triazole
    "c1cn[nH]c1",                 # Pyrazole

    # ── Oxygen-rich ──────────────────────────────────────────
    "O",                          # Water
    "OO",                         # Hydrogen peroxide
    "O=O",                        # O2
    "CO",                         # Methanol
    "OCO",                        # Formaldehyde hydrate / methylene glycol
    "OCCO",                       # Ethylene glycol
    "OCCCO",                      # 1,3-propanediol
    "OC=O",                       # Formic acid
    "OC(=O)C",                    # Acetic acid
    "OC(=O)CC",                   # Propionic acid
    "OC(=O)O",                    # Carbonic acid
    "CC(=O)O",                    # Acetic acid (alt)
    "OCC(=O)O",                   # Glycolic acid
    "OC(=O)C(=O)O",               # Oxalic acid
    "OC(O)=O",                    # Carbonic acid (alt)
    "O=CO",                       # Formic acid (alt)
    "C(=O)O",                     # Formic acid (alt2)
    "O=CC=O",                     # Glyoxal
    "O=CCC=O",                    # Succinaldehyde
    "COC=O",                      # Methyl formate

    # ── Fluorine-containing ──────────────────────────────────
    "F",                          # HF
    "FF",                         # F2
    "CF",                         # Fluoromethane
    "CCF",                        # Fluoroethane
    "CCCF",                       # 1-fluoropropane
    "FC(F)F",                     # Trifluoromethane
    "FC(F)(F)F",                  # Tetrafluoromethane (CF4)
    "FCC(F)F",                    # 1,1,2-trifluoroethane
    "FC=O",                       # Formyl fluoride
    "FC(=O)F",                    # Carbonyl difluoride
    "FN",                         # Nitrogen fluoride
    "FNF",                        # Difluoroamine
    "FOF",                        # Oxygen difluoride (approx)
    "FC(F)=O",                    # Carbonyl difluoride (alt)
    "FCC",                        # Fluoroethane (alt)
    "FCCF",                       # 1,2-difluoroethane
    "FC(C)F",                     # 1,1-difluoroethane
    "FC(F)(F)C",                  # 1,1,1-trifluoroethane
    "FC(F)(F)CF",                 # 1,1,1,2-tetrafluoroethane
    "FC(F)C(F)F",                 # 1,1,2,2-tetrafluoroethane

    # ── Mixed heteroatoms (N + O) ────────────────────────────
    "ONC",                        # N-methylhydroxylamine
    "ON=O",                       # Nitrous acid
    "NC(=O)O",                    # Carbamic acid
    "OC(=O)N",                    # Carbamic acid (alt)
    "NCC(=O)O",                   # Glycine
    "NCCO",                       # Ethanolamine
    "NCCCO",                      # 3-aminopropanol
    "NC(=O)CO",                   # Glycolamide
    "ONC=O",                      # N-hydroxyformamide
    "OC(=O)NC",                   # N-methylcarbamic acid

    # ── Mixed heteroatoms (N + F) ────────────────────────────
    "NF",                         # Nitrogen monofluoride
    "FNC",                        # N-fluoromethylamine
    "FC(N)=O",                    # Fluoroformamide
    "NCF",                        # Fluoromethylamine
    "NCCF",                       # 2-fluoroethylamine

    # ── Mixed heteroatoms (O + F) ────────────────────────────
    "OC(F)F",                     # Difluoromethanol
    "OC(F)(F)F",                  # Trifluoromethanol
    "FC(=O)O",                    # Fluoroformic acid
    "OCC(F)F",                    # 2,2-difluoroethanol
    "FOC",                        # Methyl hypofluorite

    # ── Mixed (N + O + F) ────────────────────────────────────
    "NC(=O)CF",                   # 2-fluoroacetamide
    "FC(=O)NC",                   # N-methyl fluoroformamide
    "NCC(F)=O",                   # 2-fluoro-1-aminoethanone
    "OC(=O)C(N)F",                # N,F-substituted glycine
    "NC(F)(F)C=O",                # Difluoroalaninal

    # ── Small rings with heteroatoms ─────────────────────────
    "C1CO1",                      # Ethylene oxide (oxirane)
    "C1CN1",                      # Aziridine
    "C1NO1",                      # Oxaziridine
    "C1COC1",                     # Oxetane
    "C1CNC1",                     # Azetidine
    "C1CCNC1",                    # Pyrrolidine (existing but with N)
    "C1CCOC1",                    # Tetrahydrofuran
    "O=C1CC1",                    # Cyclopropanone
    "O=C1CCC1",                   # Cyclobutanone
    "O=C1CCCC1",                  # Cyclopentanone

    # ── Heterocyclic aromatics ───────────────────────────────
    "c1ccoc1",                    # Furan
    "c1cc[nH]c1",                 # Pyrrole
    "c1ccncc1",                   # Pyridine
    "c1cnoc1",                    # Isoxazole
    "c1cscn1",                    # Thiazole
]

# ── Canonical helper ──────────────────────────────────────────────────────────
def canon(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(m) if m else None
    except Exception:
        return None

def check_stability(mol):
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

# ── Stage runners (same as test_120_molecules.py) ────────────────────────────
def run_stage1(mol_3d):
    try:
        xyz_lines = [f"{mol_3d.GetNumAtoms()}", "gen"]
        conf = mol_3d.GetConformer()
        for i, atom in enumerate(mol_3d.GetAtoms()):
            p = conf.GetAtomPosition(i)
            sym = atom.GetSymbol()
            xyz_lines.append(f"{sym}  {p.x:.4f}  {p.y:.4f}  {p.z:.4f}")
        xyz_str = "\n".join(xyz_lines) + "\n"
        mol = xyz_to_rdkit_mol(xyz_str)
        if mol is None:
            return None, None
        smi = Chem.MolToSmiles(mol)
        if smi and "." not in smi:
            return mol, smi
    except Exception:
        pass
    return None, None

def run_stage2(mol_3d, anums):
    try:
        conf = mol_3d.GetConformer()
        coords = np.array([[conf.GetAtomPosition(i).x,
                             conf.GetAtomPosition(i).y,
                             conf.GetAtomPosition(i).z] for i in range(mol_3d.GetNumAtoms())])
        pos_t = torch.tensor(coords)
        for thresh in [1.8, 2.0, 2.3, 2.6]:
            raw_bonds = infer_bonds_from_distance(pos_t, anums, threshold=thresh)
            pruned    = prune_bonds_for_valence(anums, raw_bonds)
            mol       = graph_to_rdkit_mol(anums, pruned)
            if check_chemical_validity(mol):
                smi = Chem.MolToSmiles(mol)
                if smi and "." not in smi:
                    return mol, smi
    except Exception:
        pass
    return None, None

def run_stage3(anums, seed=42):
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
    except Exception:
        pass
    return None, None

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    out_csv = os.path.join(ROOT, "results", "pipeline_test_100_diverse.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    fieldnames = [
        "id", "input_smiles", "expected_smiles",
        "stage1_pass", "stage1_smiles", "stage1_match", "stage1_atom_stab", "stage1_mol_stable",
        "stage2_pass", "stage2_smiles", "stage2_match", "stage2_atom_stab", "stage2_mol_stable",
        "stage3_pass", "stage3_smiles", "stage3_match", "stage3_atom_stab", "stage3_mol_stable",
        "any_stage_pass", "any_stage_match", "best_stage", "best_smiles",
        "mol_wt", "num_atoms", "elapsed_s", "error",
    ]

    results = []
    print(f"Testing {len(DIVERSE_SMILES)} diverse molecules...")
    print("=" * 65)

    for idx, smi in enumerate(DIVERSE_SMILES, 1):
        t0 = time.time()
        row = {k: "" for k in fieldnames}
        row["id"] = idx + 200  # offset IDs to avoid clash with existing 1-141
        row["input_smiles"] = smi

        csmi = canon(smi)
        if csmi is None:
            row["expected_smiles"] = "INVALID"
            row["error"] = "Invalid SMILES"
            row["any_stage_pass"] = False
            row["any_stage_match"] = False
            row["mol_wt"] = 0.0
            row["num_atoms"] = 0
            row["elapsed_s"] = 0.0
            results.append(row)
            print(f"  [{idx:>3}] SKIP  {smi}  (invalid)")
            continue

        row["expected_smiles"] = csmi

        # Build 3D ground truth via RDKit
        ref_mol = Chem.MolFromSmiles(csmi)
        ref_mol = Chem.AddHs(ref_mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        ok = AllChem.EmbedMolecule(ref_mol, params)
        if ok < 0:
            # Fallback: try ETKDG
            ok = AllChem.EmbedMolecule(ref_mol, AllChem.ETKDG())
        if ok < 0:
            row["expected_smiles"] = csmi
            row["error"] = "3D embed failed"
            row["any_stage_pass"] = False
            row["any_stage_match"] = False
            row["mol_wt"] = round(Descriptors.MolWt(Chem.MolFromSmiles(csmi)), 2)
            row["num_atoms"] = Chem.MolFromSmiles(csmi).GetNumAtoms()
            row["elapsed_s"] = round(time.time() - t0, 3)
            results.append(row)
            print(f"  [{idx:>3}] SKIP  {smi}  (3D embed failed)")
            continue

        AllChem.MMFFOptimizeMolecule(ref_mol)

        anums = [a.GetAtomicNum() for a in ref_mol.GetAtoms()]
        row["mol_wt"] = round(Descriptors.MolWt(Chem.RemoveHs(ref_mol)), 2)
        row["num_atoms"] = ref_mol.GetNumAtoms()

        best_mol, best_smi, best_stage = None, None, None

        # Stage 1
        m1, s1 = run_stage1(ref_mol)
        row["stage1_pass"] = m1 is not None
        row["stage1_smiles"] = s1 or ""
        row["stage1_match"] = (canon(s1) == csmi) if s1 else False
        if m1:
            as1, ms1 = check_stability(m1)
            row["stage1_atom_stab"] = as1
            row["stage1_mol_stable"] = ms1
            if best_mol is None:
                best_mol, best_smi, best_stage = m1, s1, 1
        else:
            row["stage1_atom_stab"] = 0.0
            row["stage1_mol_stable"] = False

        # Stage 2
        m2, s2 = run_stage2(ref_mol, anums)
        row["stage2_pass"] = m2 is not None
        row["stage2_smiles"] = s2 or ""
        row["stage2_match"] = (canon(s2) == csmi) if s2 else False
        if m2:
            as2, ms2 = check_stability(m2)
            row["stage2_atom_stab"] = as2
            row["stage2_mol_stable"] = ms2
            if best_mol is None:
                best_mol, best_smi, best_stage = m2, s2, 2
        else:
            row["stage2_atom_stab"] = 0.0
            row["stage2_mol_stable"] = False

        # Stage 3
        m3, s3 = run_stage3(anums)
        row["stage3_pass"] = m3 is not None
        row["stage3_smiles"] = s3 or ""
        row["stage3_match"] = (canon(s3) == csmi) if s3 else False
        if m3:
            as3, ms3 = check_stability(m3)
            row["stage3_atom_stab"] = as3
            row["stage3_mol_stable"] = ms3
            if best_mol is None:
                best_mol, best_smi, best_stage = m3, s3, 3
        else:
            row["stage3_atom_stab"] = 0.0
            row["stage3_mol_stable"] = False

        row["any_stage_pass"]  = best_mol is not None
        row["any_stage_match"] = (canon(best_smi) == csmi) if best_smi else False
        row["best_stage"]      = best_stage or ""
        row["best_smiles"]     = best_smi or ""
        row["elapsed_s"]       = round(time.time() - t0, 3)

        sym = "OK" if row["any_stage_pass"] else "FAIL"
        print(f"  [{idx:>3}] {sym:>4}  {smi:<25}  S1={'P' if row['stage1_pass'] else 'F'}  S2={'P' if row['stage2_pass'] else 'F'}  S3={'P' if row['stage3_pass'] else 'F'}  best={best_stage}")

        results.append(row)

    # Save CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved: {out_csv}")
    print(f"Total: {len(results)} molecules")

    # ── Now merge with existing CSV ──────────────────────────────────────────
    existing_csv = os.path.join(ROOT, "results", "pipeline_test_120.csv")
    merged_csv   = os.path.join(ROOT, "results", "pipeline_test_merged.csv")

    existing_rows = []
    with open(existing_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            existing_rows.append(r)

    # Re-number merged rows sequentially
    all_rows = existing_rows + results
    for i, r in enumerate(all_rows, 1):
        r["id"] = i

    with open(merged_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Merged: {merged_csv}  ({len(all_rows)} total rows)")
    print("=" * 65)


if __name__ == "__main__":
    main()
