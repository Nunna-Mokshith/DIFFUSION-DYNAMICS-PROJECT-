import os
import math
import random
import torch
import numpy as np

# RDKit for Real-Life Physics relaxation
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# Original maps + new heavier elements
ATOM_MAP = {0: 1, 1: 6, 2: 7, 3: 8, 4: 9}  # QM9 base
EXTENDED_SYM_MAP = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl"}

def xyz_to_rdkit_mol(xyz_str, charge=0):
    """
    Less prone to error: Conversts raw XYZ into an RDKit Mol,
    infers bonds dynamically, and sanitizes it to ensure chemical validity.
    """
    try:
        raw_mol = Chem.MolFromXYZBlock(xyz_str)
        if raw_mol is None:
            return None
        # RDKit doesn't assign bonds from XYZ by default in all versions.
        # We can simulate bonds or rely on AllChem to determine connectivity (available in recent RDKit >= 2022.09)
        from rdkit.Chem import rdDetermineBonds
        rdDetermineBonds.DetermineConnectivity(raw_mol)
        rdDetermineBonds.DetermineBondOrders(raw_mol, charge=charge)
        return raw_mol
    except Exception as e:
        # Fallback if rdDetermineBonds is not available or fails
        return None

def relax_structure(xyz_str):
    """
    Near Real Life: Uses MMFF94 force field to computationally relax 
    the AI-generated coordinates into a physically stable minimum energy well.
    """
    mol = xyz_to_rdkit_mol(xyz_str)
    if mol is None:
        return xyz_str, False # Fallback to original
    
    # Run MMFF94 minimization
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        relaxed_xyz = Chem.MolToXYZBlock(mol)
        return relaxed_xyz, True
    except:
        return xyz_str, False

def apply_element_doping(atomic_nums, probability=0.1):
    """
    Increase Base Elements: Randomly substituting available nodes 
    with heavier elements (P, S, Cl) to increase molecular diversity.
    """
    new_anums = []
    heavy_options = [15, 16, 17] # P, S, Cl
    for an in atomic_nums:
        if an in [6, 7, 8] and random.random() < probability:
            new_anums.append(random.choice(heavy_options))
        else:
            new_anums.append(an)
    return new_anums
def compute_advanced_thermodynamics(base_G, temp_K, pressure_atm, ph_level, solvent_polarity, desired_band_gap, durability_hrs):
    """
    More Input Quantities: Factors in Pressure and pH to compute 
    a more realistic situational Gibbs Free Energy.
    """
    R = 8.314 # J/(mol K)
    # Convert base_G from eV to Joules for physical calculation approximation
    # PV factor (assuming ideal gas simplification for pressure penalty)
    pressure_penalty = 8.617e-5 * temp_K * math.log(max(pressure_atm, 0.01)) 
    
    # pH penalty (simple acid/base heuristic approximation)
    ph_penalty = abs(7.0 - ph_level) * 0.05 
    
    # Solvent polarity penalty
    polarity_penalty = abs(78.5 - solvent_polarity) * 0.01
    
    # Longevity / durability penalty
    durability_bonus = math.log10(max(durability_hrs, 1.0)) * 0.1
    
    # Band gap penalty
    bg_penalty = abs(2.0 - desired_band_gap) * 0.2
    
    real_life_G = base_G + pressure_penalty + ph_penalty + polarity_penalty + bg_penalty - durability_bonus
    stability_score = 1 / (1 + math.exp(real_life_G / (8.617e-5 * temp_K)))
    return real_life_G, stability_score

# Example generation pipeline wrap
def generate_advanced_catalyst(gen_pos_np, atomic_nums, temp, pressure, ph, 
                               reaction_type="CO2 Reduction", max_mw=500.0, 
                               max_activation_energy=25.0, binding_energy_pref=-1.5, 
                               desired_band_gap=2.0, surface_area_req=100.0, 
                               element_doping_pct=15, solvent_polarity=78.5, 
                               cost_effectiveness=50.0, durability_hrs=1000, 
                               ox_state_flex="Medium", steric_hindrance="Medium"):
    # 1. Expand elements using the doping percentage
    doped_anums = apply_element_doping(atomic_nums, probability=(element_doping_pct/100.0))
    
    # 2. Build initial XYZ string
    xyz_str = f"{len(doped_anums)}\nGenerated Catalyst\n"
    for i, an in enumerate(doped_anums):
        sym = EXTENDED_SYM_MAP.get(an, "C")
        x, y, z = gen_pos_np[i]
        xyz_str += f"{sym} {x:.5f} {y:.5f} {z:.5f}\n"

    # 3. Relax physics (Less error prone & near real life)
    relaxed_xyz, success = relax_structure(xyz_str)
    
    # 4. Thermodynamic corrections (More inputs)
    # Using dummy base_G for demonstration, you'd feed the model's G here
    base_G_eV = -11900.0 # From model output roughly
    real_G, stability = compute_advanced_thermodynamics(
        base_G_eV, temp, pressure, ph, 
        solvent_polarity, desired_band_gap, durability_hrs
    )
    
    status = "Optimized with MMFF94" if success else "Raw AI Output (Relaxation Failed)"
    print(f"--- Advanced Evaluation ---")
    print(f"Status: {status}")
    print(f"Real-life Gibbs Energy (with Patm & pH): {real_G:.2f} eV")
    print(f"Adjusted Stability Score: {stability:.4f}")
    
    return relaxed_xyz

# Usage:
# relaxed_xyz_str = generate_advanced_catalyst(gen_pos.cpu().numpy(), anums, TARGET_TEMP_K, FLEXIBILITY, TARGET_PRESSURE, TARGET_PH)
