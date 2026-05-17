# -*- coding: utf-8 -*-
"""
120-Molecule 3-Stage RDKit Bond Inference Pipeline Test
========================================================
- Builds 120 molecules from canonical SMILES
- Generates realistic 3-D coords via ETKDGv3 (ground-truth geometry)
- Runs all 3 pipeline stages on the coordinates
- Compares each stage output against expected SMILES
- Saves full results to:
      results/pipeline_test_120.csv
      results/pipeline_test_120_summary.txt

Run:
    python scripts/test_120_molecules.py
"""

import os, sys, csv, time, traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
import numpy as np
import torch

RDLogger.DisableLog("rdApp.*")

from scripts.physics_guided_molecular_diffusion import (
    infer_bonds_from_distance,
    graph_to_rdkit_mol,
    check_chemical_validity,
    prune_bonds_for_valence,
)
from scripts.Advanced_Molecular_Design import xyz_to_rdkit_mol

# ---------------------------------------------------------------------------
#  120 molecules — diverse coverage: alkanes, alcohols, acids, amines,
#  aromatics, heterocycles, halides, ketones, esters, sulfur/phosphorus cpds
# ---------------------------------------------------------------------------
SMILES_120 = [
    # Alkanes / cycloalkanes
    "C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC",
    "C1CCCCC1", "C1CCCC1", "C1CCC1", "CC(C)C", "CC(C)(C)C",
    # Alkenes / alkynes
    "C=C", "C=CC", "C=CCC", "CC=CC", "C#C", "C#CC",
    # Aromatics
    "c1ccccc1", "Cc1ccccc1", "c1ccc(C)cc1",
    "c1ccc(O)cc1", "c1ccc(N)cc1", "c1ccc(Cl)cc1",
    "c1ccc(F)cc1", "c1ccc(Br)cc1", "c1ccc(cc1)C(=O)O",
    "c1ccncc1", "c1ccoc1", "c1ccsc1",
    "c1ccc2ccccc2c1", "c1ccc2[nH]ccc2c1",
    # Alcohols
    "CO", "CCO", "CCCO", "CCCCO", "CC(O)C",
    "OCC(O)CO", "c1ccccc1CO",
    # Ketones / aldehydes
    "CC(=O)C", "CCC(=O)C", "CC(=O)CC", "O=CC",
    "O=CCC", "c1ccccc1C=O", "CC(=O)c1ccccc1",
    # Carboxylic acids / esters
    "CC(=O)O", "CCC(=O)O", "c1ccccc1C(=O)O",
    "CC(=O)OC", "CC(=O)OCC", "CCOC(=O)C",
    # Amines
    "CN", "CCN", "CCCN", "CC(N)C", "c1ccccc1N",
    "c1ccc(N)cc1", "CN(C)C", "CCN(CC)CC",
    # Amides
    "CC(=O)N", "CC(=O)NC", "CC(=O)NCC",
    "c1ccccc1C(=O)N", "NC(=O)c1ccccc1",
    # Nitriles
    "CC#N", "CCC#N", "c1ccccc1C#N",
    # Halides
    "CCl", "CBr", "CF", "CI",
    "CCCl", "CCBr", "c1ccccc1Cl", "c1ccccc1Br", "c1ccccc1F",
    "ClCCCl", "BrCCBr",
    # Ethers
    "COC", "CCOCC", "c1cccoc1", "COc1ccccc1",
    # Sulfur compounds
    "CS", "CCS", "CCSC", "c1ccccc1S",
    "CC(=O)S", "CS(=O)C", "CS(=O)(=O)C",
    # Phosphorus
    "CP(=O)(O)O", "COP(=O)(OC)OC",
    # Heterocycles
    "C1CCNCC1", "C1CCOCC1", "C1CCOC1",
    "c1ccnc(N)c1", "c1cnc2ccccc2n1",
    "C1CN1", "C1CNCC1",
    # Amino acids (simple)
    "NCC(=O)O",       # Glycine
    "NC(C)C(=O)O",    # Alanine
    "NC(CS)C(=O)O",   # Cysteine
    "NC(CO)C(=O)O",   # Serine
    # Sugars (simple)
    "OCC(O)C(O)C(O)CO",  # open-chain sugar
    # Drug-like fragments
    "c1ccc(cc1)C(=O)Nc1ccccc1",
    "c1ccc2c(c1)CCCO2",
    "CC(=O)Nc1ccc(O)cc1",   # Paracetamol-like
    "c1ccc(cc1)N=Nc1ccc(N)cc1",
    # Vitamins / cofactor fragments
    "Cc1ncc(COP(=O)(O)O)c(N)n1",  # B6 fragment
    "OC(=O)c1ccc(O)cc1",           # PHBA
    # Nitro compounds
    "c1ccc([N+](=O)[O-])cc1",
    "CC[N+](=O)[O-]",
    # Epoxides / lactones
    "C1CO1",
    "O=C1CCCO1",
    "O=C1CCCCO1",
    # Anhydrides / imides
    "CC(=O)OC(=O)C",
    "O=C1NC(=O)c2ccccc21",
    # More ring systems
    "C1CCCCC1C",
    "C1CC2CCCC2CC1",
    "c1ccc2c(c1)ccc1ccccc12",  # Anthracene
    "c1ccc2cc3ccccc3cc2c1",    # Phenanthrene
    # Fluorinated
    "FC(F)(F)C(=O)O",
    "FC(F)F",
    "C(F)(F)(F)Cl",
    # Boron
    "B(O)(O)c1ccccc1",
    # Silicon
    "C[Si](C)(C)C",
    # Extra miscellaneous
    "OC(=O)CC(=O)O",        # Malonic acid
    "OC(=O)CCC(=O)O",       # Succinic acid
    "OC(=O)c1ccccc1O",      # Salicylic acid
    "Nc1ccc(cc1)S(=O)(=O)N",# Sulfanilamide
    "CC12CCC(CC1)CC2",       # Adamantane-like
    "C1CC1",                 # Cyclopropane
    "C=C(C)C(=O)O",         # Methacrylic acid
    "C=CC(=O)O",             # Acrylic acid
    "OC(=O)/C=C/C(=O)O",    # Fumaric acid
    r"OC(=O)/C=C\C(=O)O",    # Maleic acid
    "c1cc[nH]c1",            # Pyrrole
    "c1ccncc1",              # Pyridine (dup intentional for weight)
    "c1cnc[nH]1",            # Imidazole
    "c1cnnc[nH]1",           # Pyrazole
    "c1nccs1",               # Thiazole
    "C1=CC=NC=C1",           # 1,4-Dihydropyridine-like
]

EXPECTED_VALENCE = {1: 1, 6: 4, 7: 3, 8: 2, 9: 1, 16: 2, 17: 1, 35: 1, 53: 1, 15: 3, 5: 3, 14: 4}


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def canon(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(m) if m else None
    except Exception:
        return None


def smiles_to_xyz(smi, seed=42):
    """Generate 3-D coordinates from SMILES using ETKDGv3."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None, None, None
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        if AllChem.EmbedMolecule(mol, params) < 0:
            return None, None, None
        AllChem.MMFFOptimizeMolecule(mol)
        conf = mol.GetConformer()
        n = mol.GetNumAtoms()
        anums = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(n)]
        coords = np.array([[conf.GetAtomPosition(i).x,
                            conf.GetAtomPosition(i).y,
                            conf.GetAtomPosition(i).z] for i in range(n)])
        lines = [f"{n}", f"gen from {smi}"]
        for i in range(n):
            sym = mol.GetAtomWithIdx(i).GetSymbol()
            lines.append(f"{sym} {coords[i,0]:.5f} {coords[i,1]:.5f} {coords[i,2]:.5f}")
        xyz_str = "\n".join(lines) + "\n"
        return anums, coords, xyz_str
    except Exception:
        return None, None, None


def check_stability(mol):
    if mol is None:
        return 0.0, False
    n_stable, all_ok = 0, True
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
    return round(n_stable / max(mol.GetNumAtoms(), 1) * 100, 1), all_ok


def run_stage1(xyz_str):
    try:
        mol = xyz_to_rdkit_mol(xyz_str)
        if mol is None:
            return None, None
        smi = Chem.MolToSmiles(mol)
        if smi and "." not in smi:
            return mol, smi
    except Exception:
        pass
    return None, None


def run_stage2(coords, anums):
    try:
        pos_t = torch.tensor(coords, dtype=torch.float32)
        for thresh in [1.8, 2.0, 2.3, 2.6]:
            raw_bonds  = infer_bonds_from_distance(pos_t, anums, threshold=thresh)
            pruned     = prune_bonds_for_valence(anums, raw_bonds)
            mol        = graph_to_rdkit_mol(anums, pruned)
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


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

def main():
    out_dir = os.path.join(ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "pipeline_test_120.csv")
    sum_path = os.path.join(out_dir, "pipeline_test_120_summary.txt")

    fieldnames = [
        "id", "input_smiles", "expected_smiles",
        "stage1_pass", "stage1_smiles", "stage1_match", "stage1_atom_stab", "stage1_mol_stable",
        "stage2_pass", "stage2_smiles", "stage2_match", "stage2_atom_stab", "stage2_mol_stable",
        "stage3_pass", "stage3_smiles", "stage3_match", "stage3_atom_stab", "stage3_mol_stable",
        "any_stage_pass", "any_stage_match", "best_stage", "best_smiles",
        "mol_wt", "num_atoms", "elapsed_s", "error"
    ]

    print("=" * 70)
    print("  120-MOLECULE 3-STAGE RDKIT PIPELINE TEST")
    print("=" * 70)
    print(f"  Output CSV     : {csv_path}")
    print(f"  Output Summary : {sum_path}")
    print("=" * 70)

    rows = []
    t_total = time.perf_counter()

    for idx, smi_raw in enumerate(SMILES_120, 1):
        t0 = time.perf_counter()
        expected = canon(smi_raw)
        row = {
            "id": idx, "input_smiles": smi_raw,
            "expected_smiles": expected or "INVALID",
            "stage1_pass": False, "stage1_smiles": "", "stage1_match": False,
            "stage1_atom_stab": 0.0, "stage1_mol_stable": False,
            "stage2_pass": False, "stage2_smiles": "", "stage2_match": False,
            "stage2_atom_stab": 0.0, "stage2_mol_stable": False,
            "stage3_pass": False, "stage3_smiles": "", "stage3_match": False,
            "stage3_atom_stab": 0.0, "stage3_mol_stable": False,
            "any_stage_pass": False, "any_stage_match": False,
            "best_stage": None, "best_smiles": "",
            "mol_wt": 0.0, "num_atoms": 0, "elapsed_s": 0.0, "error": ""
        }

        if expected is None:
            row["error"] = "Invalid SMILES"
            rows.append(row)
            print(f"  [{idx:>3}/120] SKIP  {smi_raw[:40]:<40}  (invalid SMILES)")
            continue

        try:
            anums, coords, xyz_str = smiles_to_xyz(smi_raw)
            if anums is None:
                row["error"] = "3D embedding failed"
                rows.append(row)
                print(f"  [{idx:>3}/120] SKIP  {expected[:40]:<40}  (3D embed failed)")
                continue

            row["num_atoms"] = len(anums)

            # -- Stage 1
            mol1, smi1 = run_stage1(xyz_str)
            if mol1:
                as1, ms1 = check_stability(mol1)
                row.update({
                    "stage1_pass": True, "stage1_smiles": smi1,
                    "stage1_match": canon(smi1) == expected,
                    "stage1_atom_stab": as1, "stage1_mol_stable": ms1,
                })

            # -- Stage 2
            mol2, smi2 = run_stage2(coords, anums)
            if mol2:
                as2, ms2 = check_stability(mol2)
                row.update({
                    "stage2_pass": True, "stage2_smiles": smi2,
                    "stage2_match": canon(smi2) == expected,
                    "stage2_atom_stab": as2, "stage2_mol_stable": ms2,
                })

            # -- Stage 3
            mol3, smi3 = run_stage3(anums)
            if mol3:
                as3, ms3 = check_stability(mol3)
                row.update({
                    "stage3_pass": True, "stage3_smiles": smi3,
                    "stage3_match": canon(smi3) == expected,
                    "stage3_atom_stab": as3, "stage3_mol_stable": ms3,
                })

            # -- Best stage (first passing, priority: 1 > 2 > 3)
            for s_label, s_pass, s_smi, s_mol in [
                (1, row["stage1_pass"], smi1 if mol1 else None, mol1),
                (2, row["stage2_pass"], smi2 if mol2 else None, mol2),
                (3, row["stage3_pass"], smi3 if mol3 else None, mol3),
            ]:
                if s_pass and row["best_stage"] is None:
                    row["best_stage"] = s_label
                    row["best_smiles"] = s_smi or ""
                    row["any_stage_pass"] = True
                    row["any_stage_match"] = (canon(s_smi) == expected) if s_smi else False
                    if s_mol:
                        try:
                            row["mol_wt"] = round(Descriptors.MolWt(s_mol), 2)
                        except Exception:
                            pass

            elapsed = round(time.perf_counter() - t0, 3)
            row["elapsed_s"] = elapsed

            # -- Console line
            s1 = "S1:PASS" if row["stage1_pass"] else "S1:FAIL"
            s2 = "S2:PASS" if row["stage2_pass"] else "S2:FAIL"
            s3 = "S3:PASS" if row["stage3_pass"] else "S3:FAIL"
            match_tag = "MATCH" if row["any_stage_match"] else "DIFF "
            print(f"  [{idx:>3}/120] {expected[:35]:<35} {s1}  {s2}  {s3}  {match_tag}  {elapsed:.2f}s")

        except Exception as e:
            row["error"] = str(e)
            print(f"  [{idx:>3}/120] ERROR  {expected[:40]:<40}  {e}")

        rows.append(row)

    total_time = round(time.perf_counter() - t_total, 1)

    # ---------------------------------------------------------------------------
    #  Write CSV
    # ---------------------------------------------------------------------------
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ---------------------------------------------------------------------------
    #  Compute summary stats
    # ---------------------------------------------------------------------------
    tested  = [r for r in rows if not r["error"]]
    N       = len(tested)

    s1_pass  = sum(1 for r in tested if r["stage1_pass"])
    s2_pass  = sum(1 for r in tested if r["stage2_pass"])
    s3_pass  = sum(1 for r in tested if r["stage3_pass"])
    any_pass = sum(1 for r in tested if r["any_stage_pass"])

    s1_match  = sum(1 for r in tested if r["stage1_match"])
    s2_match  = sum(1 for r in tested if r["stage2_match"])
    s3_match  = sum(1 for r in tested if r["stage3_match"])
    any_match = sum(1 for r in tested if r["any_stage_match"])

    skipped   = len(rows) - N

    summary_lines = [
        "=" * 70,
        "  120-MOLECULE PIPELINE TEST — SUMMARY REPORT",
        "=" * 70,
        f"  Total input SMILES      : {len(SMILES_120)}",
        f"  Successfully tested     : {N}",
        f"  Skipped (bad embed/smi) : {skipped}",
        f"  Total time              : {total_time}s",
        "-" * 70,
        "  PASS RATE (molecule produced a valid connected SMILES):",
        f"    Stage 1 (rdDetermineBonds) : {s1_pass:>4} / {N}  ({s1_pass/N*100:.1f}%)",
        f"    Stage 2 (distance-based)   : {s2_pass:>4} / {N}  ({s2_pass/N*100:.1f}%)",
        f"    Stage 3 (ETKDGv3 fallback) : {s3_pass:>4} / {N}  ({s3_pass/N*100:.1f}%)",
        f"    Any stage passes           : {any_pass:>4} / {N}  ({any_pass/N*100:.1f}%)",
        "-" * 70,
        "  ACCURACY (output canonical SMILES == expected canonical SMILES):",
        f"    Stage 1 exact match        : {s1_match:>4} / {s1_pass}  ({s1_match/max(s1_pass,1)*100:.1f}%)",
        f"    Stage 2 exact match        : {s2_match:>4} / {s2_pass}  ({s2_match/max(s2_pass,1)*100:.1f}%)",
        f"    Stage 3 exact match        : {s3_match:>4} / {s3_pass}  ({s3_match/max(s3_pass,1)*100:.1f}%)",
        f"    Best-stage match (overall) : {any_match:>4} / {N}  ({any_match/N*100:.1f}%)",
        "-" * 70,
        "  NOTE: 'DIFF' = valid molecule but different bond orders (e.g. single",
        "        instead of double). This is EXPECTED for Stages 2 & 3 which are",
        "        geometry-only fallbacks without bond-order inference.",
        "=" * 70,
        f"  Full per-molecule data saved to:",
        f"    {csv_path}",
        "=" * 70,
    ]

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    with open(sum_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")

    print(f"\n  [Done] CSV and summary written to results/\n")


if __name__ == "__main__":
    main()
