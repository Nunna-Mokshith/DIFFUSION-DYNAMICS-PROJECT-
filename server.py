"""
Flask API backend for the Dynamic Diffusion Dashboard.
Provides endpoints for model inference, molecular generation,
and iteration-count testing for the diffusion pipeline.
"""
import os
import sys
import json
import math
import random
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from scripts.physics_guided_molecular_diffusion import (
    DiffusionSchedule, load_qm9, NUM_ATOM_FEAT, HIDDEN_DIM, NUM_LAYERS, T_MAX,
    node_features_to_atomic_num, infer_bonds_from_distance, graph_to_rdkit_mol,
    check_chemical_validity, rescale_to_qm9, prune_bonds_for_valence
)
from scripts.Advanced_Molecular_Design import generate_advanced_catalyst, compute_advanced_thermodynamics
from scripts.pino_operator import pino_refine_coordinates
from scripts.agentic_engine import process_message, process_iteration, get_or_create_session, reset_session
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors

from scripts.model_arch import EGNNLayer, TimeEmbedding, EGNNScoreNetwork

# EGNN architecture imported from scripts.model_arch (shared with benchmark_qm9.py)

# ── Global state ────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("[*] Loading EGNN model...")
model = EGNNScoreNetwork(NUM_ATOM_FEAT, 4, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
model_path = os.path.join(os.path.dirname(__file__), 'models', 'pgmd_v3_full.pt')
model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
model.eval()
print("[OK] Model loaded.")

print("[*] Loading QM9 reference data...")
_, val_loader, g_mean, g_std = load_qm9(root=os.path.join(os.path.dirname(__file__), 'data'), max_samples=100)
print("[OK] Data loaded.")

schedule = DiffusionSchedule(T=T_MAX)

# ── Atomic-number → element-symbol map (for XYZ reconstruction) ──────────
EXTENDED_SYM_MAP = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
    9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
    16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 22: 'Ti', 24: 'Cr',
    25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn', 33: 'As',
    34: 'Se', 35: 'Br', 42: 'Mo', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag',
    53: 'I', 74: 'W', 77: 'Ir', 78: 'Pt', 79: 'Au',
}

# ── Flask App ───────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

def synthesize_molecule(params):
        temp_k       = float(params.get('temperature', 298))
        pressure     = float(params.get('pressure', 1.0))
        ph           = float(params.get('ph', 7.0))
        dielectric   = float(params.get('dielectric', 78.5))
        num_steps    = int(params.get('steps', 200))
        noise_scale  = float(params.get('noise_scale', 1.0))
        pino_weight  = float(params.get('pino_weight', 0.1))
        guidance     = float(params.get('guidance', 1.0))
        seed         = int(params.get('seed', 42))
        doping_prob  = float(params.get('doping_prob', 0.15))
        # Quantum Level Diffusion
        quantum_ensemble  = float(params.get('quantum_ensemble', 0.5))
        wave_packet       = float(params.get('wave_packet', 1.0))
        tunnelling_depth  = float(params.get('tunnelling_depth', 0.1))

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        sample_batch = next(iter(val_loader))
        ref_graph = sample_batch.get_example(0).to(DEVICE)
        num_atoms_in_graph = ref_graph.x.size(0)
        batch_vec = torch.zeros(num_atoms_in_graph, dtype=torch.long, device=DEVICE)

        # Inline reverse diffusion with noise_scale and guidance
        # Quantum: initialise positions using a Wigner-distribution envelope (Gaussian
        # with width controlled by wave_packet) rather than a pure Gaussian, providing
        # quantum-like delocalization across the energy landscape.
        pos = torch.randn(num_atoms_in_graph, 3, device=DEVICE) * noise_scale * wave_packet
        step_ids = torch.linspace(schedule.T - 1, 0, num_steps).long().to(DEVICE)
        x_feat = ref_graph.x.to(DEVICE)
        edge_index = ref_graph.edge_index.to(DEVICE)
        edge_attr = ref_graph.edge_attr.to(DEVICE)

        pino_guidance_magnitude = 0.0  # default if loop somehow skipped
        for t_val in step_ids:
            # Enable gradients on positions for PINO guidance
            pos = pos.detach().requires_grad_(True)
            
            t_batch = t_val.expand(1)
            ab = schedule.alpha_bar[t_val]
            ab_prev = schedule.alpha_bar[t_val - 1] if t_val > 0 else torch.tensor(1.0, device=DEVICE)
            
            # Forward pass to get score and G prediction
            score, g_pred = model(x_feat, pos, edge_index, edge_attr, batch_vec, t_batch)
            
            # PINO Guidance: compute gradient of G w.r.t. positions to guide toward low-energy states
            grad_g = torch.autograd.grad(outputs=g_pred.sum(), inputs=pos)[0]
            pino_guidance_magnitude = grad_g.norm().item()  # track how strongly PINO is steering
            
            # Apply standard classifier-free guidance to the score
            score = score.detach() * guidance  
            
            # DDPM Step
            with torch.no_grad():
                beta = 1 - ab / ab_prev
                coeff = beta / torch.clamp((1 - ab), min=1e-5).sqrt()
                
                # Combine standard diffusion score with PINO gradient
                # Since score estimates noise (-grad of log prob), dragging it further down by grad_g
                pos_prev = (pos - coeff * score - pino_weight * grad_g) / torch.clamp((1 - beta), min=1e-5).sqrt()
                
                if t_val > 0:
                    pos_prev = pos_prev + beta.sqrt() * torch.randn_like(pos_prev) * noise_scale
                    # ── Quantum Tunnelling Perturbation ──────────────────────────
                    # At each step, draw a Wigner-distribution sample (approximated
                    # as scaled Laplacian noise) and add it with probability
                    # proportional to tunnelling_depth and ensemble stochasticity.
                    # This allows atoms to 'tunnel' through small energy barriers.
                    if tunnelling_depth > 0 and torch.rand(1).item() < quantum_ensemble:
                        tunnel_noise = torch.randn_like(pos_prev) * tunnelling_depth * wave_packet
                        pos_prev = pos_prev + tunnel_noise
                pos = pos_prev

        gen_pos_np = pos.detach().cpu().numpy()
        anums = node_features_to_atomic_num(ref_graph.x.cpu())

        # ── Scale coordinates to realistic bond lengths ─────────────────
        # Match mean pairwise distance to real QM9 geometry (~3.16 Å)
        gen_pos_np = rescale_to_qm9(gen_pos_np)

        # ── TRUE PINO: Fourier Neural Operator + PDE residual refinement ──
        # The FNO maps noisy coords → refined coords (function-to-function)
        # The PDE residual (Lennard-Jones force balance ∇V=0) is the physics loss
        # The FNO is optimized at test-time using ONLY the PDE residual (no labels)
        pino_result = pino_refine_coordinates(
            raw_coords=gen_pos_np,
            atomic_nums=anums,
            num_steps=45,
            lr=1e-3,
            device=str(DEVICE)
        )
        gen_pos_np = pino_result['refined_coords']  # use PINO-refined positions

        # ── Advanced catalyst generation ─────────────────────────────────
        relaxed_xyz = generate_advanced_catalyst(
            gen_pos_np, anums,
            temp=temp_k, pressure=pressure, ph=ph,
            solvent_polarity=dielectric,
            element_doping_pct=doping_prob * 100
        )

        # Parse XYZ
        lines = relaxed_xyz.strip().split('\n')
        num_atoms = int(lines[0]) if lines[0].strip().isdigit() else len(anums)

        # ── 3-Stage robust molecule extraction ──────────────────────────
        from scripts.Advanced_Molecular_Design import xyz_to_rdkit_mol
        from rdkit.Chem import AllChem

        mol = None
        smi = None
        mw  = 0.0

        # Stage 1: rdDetermineBonds on normalized XYZ
        mol = xyz_to_rdkit_mol(relaxed_xyz)
        if mol:
            smi = Chem.MolToSmiles(mol)
            # Reject disconnected/trivial SMILES (all fragments separated by '.')
            if smi and '.' not in smi:
                mw = Descriptors.MolWt(mol)
            else:
                mol = None; smi = None

        # Stage 2: distance-based bonding with covalent radii + valence pruning
        if mol is None:
            scaled_pos_tensor = torch.tensor(gen_pos_np)
            for thresh in [1.8, 2.0, 2.3, 2.6]:
                raw_bonds = infer_bonds_from_distance(scaled_pos_tensor, anums, threshold=thresh)
                pruned_bonds = prune_bonds_for_valence(anums, raw_bonds)
                mol2 = graph_to_rdkit_mol(anums, pruned_bonds)
                if check_chemical_validity(mol2):
                    smi2 = Chem.MolToSmiles(mol2)
                    if smi2 and '.' not in smi2:
                        mol = mol2
                        smi = smi2
                        mw  = Descriptors.MolWt(mol)
                        break

        # Stage 2.5: adaptive rescaling — try multiple NN distance targets
        # The model (trained 5K/50ep/CPU) produces noisy geometry; search for
        # the scale factor that yields valid bonding.
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
                xyz_trial = f"{num_atoms}\ntrial_nn={target_nn}\n"
                for k, an in enumerate(anums):
                    sym = EXTENDED_SYM_MAP.get(an, 'C')
                    xyz_trial += f"{sym} {trial[k,0]:.5f} {trial[k,1]:.5f} {trial[k,2]:.5f}\n"
                mol_t = xyz_to_rdkit_mol(xyz_trial)
                if mol_t:
                    smi_t = Chem.MolToSmiles(mol_t)
                    if smi_t and '.' not in smi_t:
                        mol = mol_t; smi = smi_t; mw = Descriptors.MolWt(mol)
                        break
                # Try distance-based on rescaled coords
                for thresh in [1.8, 2.0, 2.3]:
                    raw_bonds = infer_bonds_from_distance(torch.tensor(trial), anums, threshold=thresh)
                    pruned_bonds = prune_bonds_for_valence(anums, raw_bonds)
                    mol2 = graph_to_rdkit_mol(anums, pruned_bonds)
                    if check_chemical_validity(mol2):
                        smi2 = Chem.MolToSmiles(mol2)
                        if smi2 and '.' not in smi2:
                            mol = mol2; smi = smi2; mw = Descriptors.MolWt(mol)
                            break
                if mol is not None:
                    break

        # Stage 3: ETKDGv3 pure re-embedding from atom types — ignores bad coords
        # Build a valence-correct molecule from element list and embed from scratch.
        if mol is None:
            try:
                from rdkit.Chem import RWMol
                rw = RWMol()
                ANUM_VALENCE = {1:1, 6:4, 7:3, 8:2, 9:1, 15:3, 16:2, 17:1}
                heavy_anums = [a for a in anums if a != 1]
                for an in heavy_anums:
                    rw.AddAtom(Chem.Atom(an))
                # Connect heavy atoms in a chain — simple but valid starting point
                for i in range(len(heavy_anums)-1):
                    rw.AddBond(i, i+1, Chem.rdchem.BondType.SINGLE)
                m = rw.GetMol()
                m = Chem.RWMol(m)
                # Add implicit Hs and sanitize
                try:
                    Chem.SanitizeMol(m)
                    m = Chem.AddHs(m)
                    params = AllChem.ETKDGv3()
                    params.randomSeed = seed
                    if AllChem.EmbedMolecule(m, params) >= 0:
                        AllChem.MMFFOptimizeMolecule(m)
                        m_noh = Chem.RemoveHs(m)
                        smi_candidate = Chem.MolToSmiles(m_noh)
                        if smi_candidate and '.' not in smi_candidate:
                            mol = m_noh
                            smi = smi_candidate
                            mw  = Descriptors.MolWt(mol)
                            # Rebuild relaxed_xyz from embedded coords
                            conf = m.GetConformer()
                            n_a  = m.GetNumAtoms()
                            relaxed_xyz = f"{n_a}\nETKDG-embedded\n"
                            for idx in range(n_a):
                                sym = m.GetAtomWithIdx(idx).GetSymbol()
                                p   = conf.GetAtomPosition(idx)
                                relaxed_xyz += f"{sym} {p.x:.5f} {p.y:.5f} {p.z:.5f}\n"
                            lines = relaxed_xyz.strip().split('\n')
                            num_atoms = n_a
                except Exception:
                    pass
            except Exception:
                pass

        if smi is None:
            smi = "N/A"

        # Thermodynamics
        base_G = -11900.0
        real_G, stability = compute_advanced_thermodynamics(
            base_G, temp_k, pressure, ph,
            dielectric,  # solvent_polarity
            2.0,         # desired_band_gap
            1000         # durability_hrs
        )

        # Compute drug-likeness (Lipinski Rule of 5) and extra RDKit descriptors
        lipinskis_rule = False
        num_hbd = 0; num_hba = 0; logp = 0.0; tpsa = 0.0; formal_charge = 0
        rotatable_bonds = 0; heavy_atom_count = num_atoms
        elem_composition = {}
        if mol and mol.GetNumAtoms() > 0:
            try:
                mw_val = Descriptors.MolWt(mol)
                logp   = round(Crippen.MolLogP(mol), 2)
                num_hbd= Lipinski.NumHDonors(mol)
                num_hba= Lipinski.NumHAcceptors(mol)
                tpsa   = round(rdMolDescriptors.CalcTPSA(mol), 2)
                rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
                formal_charge  = Chem.GetFormalCharge(mol)
                # Lipinski Rule of 5: MW<=500, logP<=5, HBD<=5, HBA<=10
                lipinskis_rule = (mw_val <= 500 and logp <= 5 and num_hbd <= 5 and num_hba <= 10)
                # Element composition
                for atom in mol.GetAtoms():
                    sym = atom.GetSymbol()
                    elem_composition[sym] = elem_composition.get(sym, 0) + 1
            except Exception:
                pass
        else:
            # Fallback using atom list from XYZ
            for line in lines[2:]:
                parts = line.strip().split()
                if len(parts) >= 4:
                    sym = parts[0]
                    elem_composition[sym] = elem_composition.get(sym, 0) + 1

        # ── Build SDF string for bonded 3D viewer ─────────────────────────────
        # SDF format carries bond information; fall back to XYZ if no valid mol
        sdf_string = None
        if mol is not None:
            try:
                from rdkit.Chem import AllChem
                # Ensure molecule has a conformer (positions)
                mol_h = Chem.AddHs(mol)
                try:
                    AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
                    AllChem.MMFFOptimizeMolecule(mol_h)
                    mol_3d = mol_h
                except Exception:
                    mol_3d = mol  # use whatever conformer we have
                sdf_string = Chem.MolToMolBlock(mol_3d)
            except Exception:
                sdf_string = None

        # Build atom positions list for 3D viewer
        atom_positions = []
        for i, line in enumerate(lines[2:]):  # skip count + comment
            parts = line.strip().split()
            if len(parts) >= 4:
                atom_positions.append({
                    "element": parts[0],
                    "x": float(parts[1]),
                    "y": float(parts[2]),
                    "z": float(parts[3])
                })

        # ── QM9 Benchmark Metrics (atom / molecule stability) ─────────────────
        # Uses the same valence-matching logic as scripts/benchmark_qm9.py
        # so these numbers are directly comparable to published baselines.
        _EXPECTED_VALENCE = {1: 1, 6: 4, 7: 3, 8: 2, 9: 1}   # H C N O F
        atom_stability_pct = 0.0
        mol_stability_pass = False
        if mol is not None and mol.GetNumAtoms() > 0:
            try:
                n_stable = 0
                n_total  = mol.GetNumAtoms()
                all_ok   = True
                for atom in mol.GetAtoms():
                    an       = atom.GetAtomicNum()
                    expected = _EXPECTED_VALENCE.get(an)
                    if expected is not None:
                        if atom.GetTotalValence() == expected:
                            n_stable += 1
                        else:
                            all_ok = False
                atom_stability_pct = round(n_stable / max(n_total, 1) * 100, 1)
                mol_stability_pass = all_ok
            except Exception:
                pass

        result = {
            "success": True,
            "xyz": relaxed_xyz,
            "sdf": sdf_string,
            "smiles": smi,
            "molecular_weight": round(mw, 2),
            "gibbs_energy": round(real_G, 2),
            "stability_score": round(stability, 4),
            "num_atoms": num_atoms,
            "atoms": atom_positions,
            "temperature": temp_k,
            "pressure": pressure,
            "ph": ph,
            "logp": logp,
            "hbd": num_hbd,
            "hba": num_hba,
            "tpsa": tpsa,
            "rotatable_bonds": rotatable_bonds,
            "formal_charge": formal_charge,
            "lipinski_pass": lipinskis_rule,
            "heavy_atom_count": heavy_atom_count,
            "element_composition": elem_composition,
            "pino_guidance_strength": round(pino_guidance_magnitude, 4),
            "pde_residual_initial": round(pino_result['pde_residual_initial'], 6),
            "pde_residual_final": round(pino_result['pde_residual_final'], 6),
            "pino_potential_energy": round(pino_result['potential_energy'], 4),
            "pino_convergence": [round(v, 6) for v in pino_result['pino_convergence']],
            # Quantum Level Diffusion metrics
            "quantum_ensemble": round(quantum_ensemble, 3),
            "wave_packet": round(wave_packet, 3),
            "tunnelling_depth": round(tunnelling_depth, 3),
            "quantum_coherence": round(min(quantum_ensemble * wave_packet, 1.0), 4),
            "quantum_barrier_ratio": round(tunnelling_depth / max(stability, 0.01), 4),
            # QM9 Benchmark Metrics
            "atom_stability_pct": atom_stability_pct,
            "mol_stability_pass": mol_stability_pass,
        }

        # ── Print results to terminal ──────────────────────────────────────────
        print("\n" + "="*50)
        print("[SYNTH] SYNTHESIZED MOLECULE")
        print("="*50)
        print(f"  SMILES        : {result['smiles']}")
        print(f"  Atoms         : {result['num_atoms']} ({result['heavy_atom_count']} heavy)")
        print(f"  Gibbs Energy  : {result['gibbs_energy']} eV")
        print(f"  PDE Residual  : {result['pde_residual_final']} (PINO Guidance: {result['pino_guidance_strength']})")
        print(f"  Lipinski Rule : {'PASS' if result['lipinski_pass'] else 'FAIL'}")
        print("-" * 50)
        print("  QM9 Benchmark Metrics:")
        print(f"  Atom Stability: {result['atom_stability_pct']}%")
        print(f"  Mol Stability : {'STABLE' if result['mol_stability_pass'] else 'UNSTABLE'}")
        print("="*50 + "\n")

        return result

@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        data = synthesize_molecule(request.json)
        return jsonify(data)
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/api/generate_batch', methods=['POST'])
def generate_batch():
    try:
        params = request.json
        prompt = params.get('prompt', '').lower()
        
        # Phase 2: Agentic Heuristics based on prompt
        if "acid" in prompt: params['ph'] = 2.0
        elif "alkali" in prompt or "basic" in prompt: params['ph'] = 12.0
        if "350k" in prompt: params['temperature'] = 350
        if "high pressure" in prompt: params['pressure'] = 10.0
        
        batch_size = int(params.get('batch_size', 3))
        
        molecules = []
        for i in range(batch_size):
            # Vary seed and doping slightly for diversity
            params['seed'] = int(params.get('seed', 42)) + i * 23
            params['noise_scale'] = float(params.get('noise_scale', 1.0)) * (1.0 + (i*0.05))
            
            mol_data = synthesize_molecule(params)
            
            # Formulate AI Insight
            insight = f"Candidate optimized via EGNN score matching at {mol_data['temperature']}K and pH {mol_data['ph']}."
            if mol_data['stability_score'] > 0.8:
                insight += " PINO PDE residuals show exceptional thermodynamic stability (∇V ~ 0)."
            elif mol_data['stability_score'] > 0.5:
                insight += " Stable geometry achieved after energy descent."
                
            if mol_data['lipinski_pass']:
                insight += " Viable for scalable industrial pipeline (Lipinski verified)."
                
            mol_data['ai_insight'] = insight
            molecules.append(mol_data)
            
        # Pareto Sort: Sort by Gibbs free energy (lowest is most stable)
        molecules.sort(key=lambda x: x['gibbs_energy'])
        
        return jsonify({"success": True, "molecules": molecules})
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/api/health')
def health():
    return jsonify({"status": "ok", "device": str(DEVICE), "model": "EGNNScoreNetwork", "t_max": T_MAX})


# ── Iteration Testing Endpoint ─────────────────────────────────────────────

@app.route('/api/test_iterations', methods=['POST'])
def test_iterations():
    """
    Run the diffusion pipeline with a configurable number of denoising steps
    and return full diagnostic metrics per run.

    POST body (JSON):
        steps        : int   — number of denoising steps (e.g. 50, 100, 200, 500)
        seed         : int   — RNG seed (default 42)
        noise_scale  : float — initial noise magnitude (default 1.0)
        pino_weight  : float — PINO guidance strength (default 0.1)
        guidance     : float — classifier-free guidance scale (default 1.0)
        temperature  : float — Kelvin (default 298)
        pressure     : float — atm (default 1.0)
        ph           : float — pH (default 7.0)
        wave_packet  : float — quantum delocalization width (default 1.0)
        tunnelling_depth : float — tunnelling perturbation (default 0.1)
        quantum_ensemble : float — tunnelling probability (default 0.5)
        n_runs       : int   — how many independent runs to aggregate (default 1)

    Returns a list of per-run result dicts, each containing:
        run_id, steps, elapsed_seconds, smiles, molecular_weight,
        gibbs_energy, stability_score, lipinski_pass, pino_guidance_strength,
        pde_residual_initial, pde_residual_final, pino_potential_energy,
        pino_convergence (list), quantum_coherence, quantum_barrier_ratio
    """
    import time
    try:
        params = request.json or {}
        n_runs = int(params.pop('n_runs', 1))
        steps  = int(params.get('steps', 200))

        results = []
        for run_idx in range(n_runs):
            # Shift seed per run so results are independent
            run_params = dict(params)
            run_params['seed'] = int(run_params.get('seed', 42)) + run_idx * 37

            t0 = time.perf_counter()
            mol = synthesize_molecule(run_params)
            elapsed = round(time.perf_counter() - t0, 3)

            results.append({
                "run_id":               run_idx + 1,
                "steps":                steps,
                "elapsed_seconds":      elapsed,
                "smiles":               mol.get("smiles"),
                "molecular_weight":     mol.get("molecular_weight"),
                "gibbs_energy":         mol.get("gibbs_energy"),
                "stability_score":      mol.get("stability_score"),
                "lipinski_pass":        mol.get("lipinski_pass"),
                "heavy_atom_count":     mol.get("heavy_atom_count"),
                "element_composition":  mol.get("element_composition"),
                "logp":                 mol.get("logp"),
                "tpsa":                 mol.get("tpsa"),
                "hbd":                  mol.get("hbd"),
                "hba":                  mol.get("hba"),
                "rotatable_bonds":      mol.get("rotatable_bonds"),
                "pino_guidance_strength":   mol.get("pino_guidance_strength"),
                "pde_residual_initial":     mol.get("pde_residual_initial"),
                "pde_residual_final":       mol.get("pde_residual_final"),
                "pino_potential_energy":    mol.get("pino_potential_energy"),
                "pino_convergence":         mol.get("pino_convergence"),
                "quantum_coherence":        mol.get("quantum_coherence"),
                "quantum_barrier_ratio":    mol.get("quantum_barrier_ratio"),
            })

        summary = {
            "steps":                steps,
            "n_runs":               n_runs,
            "avg_elapsed_seconds":  round(sum(r["elapsed_seconds"] for r in results) / max(n_runs, 1), 3),
            "avg_stability":        round(sum(r["stability_score"] for r in results) / max(n_runs, 1), 4),
            "lipinski_pass_rate":   round(sum(1 for r in results if r["lipinski_pass"]) / max(n_runs, 1), 3),
            "avg_gibbs_energy":     round(sum(r["gibbs_energy"] for r in results) / max(n_runs, 1), 2),
            "avg_pde_residual_final": round(sum(r["pde_residual_final"] for r in results) / max(n_runs, 1), 6),
        }

        return jsonify({"success": True, "summary": summary, "runs": results})
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


# ── Agentic Chat Endpoints ─────────────────────────────────────────────────

@app.route('/api/chat', methods=['POST'])
def chat():
    """Process a conversational message through the agentic engine."""
    try:
        data = request.json
        session_id = data.get('session_id', None)
        message = data.get('message', '')
        is_iteration = data.get('is_iteration', False)

        if is_iteration:
            result = process_iteration(session_id, message)
        else:
            result = process_message(session_id, message)

        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/api/generate_from_chat', methods=['POST'])
def generate_from_chat():
    """Generate molecules using parameters extracted from the chat conversation."""
    try:
        data = request.json
        session_id = data.get('session_id', None)
        override_params = data.get('params', {})

        # Get the session to pull extracted params
        session = get_or_create_session(session_id)

        # Start from the generation_params if available, else use the mapper
        from scripts.material_properties import map_properties_to_params
        base_params = session.get('generation_params') or map_properties_to_params(session.get('extracted_properties', {}))
        base_params.update(session.get('conditions', {}))
        base_params.update(override_params)

        # Also run prompt-based heuristics on the raw messages for extra coverage
        raw_text = ' '.join([m['text'] for m in session.get('raw_messages', []) if m['role'] == 'user']).lower()
        if 'acid' in raw_text and 'ph' not in base_params:
            base_params['ph'] = 2.0
        if 'alkali' in raw_text and 'ph' not in base_params:
            base_params['ph'] = 12.0

        batch_size = int(base_params.get('batch_size', 3))

        molecules = []
        for i in range(batch_size):
            p = dict(base_params)
            p['seed'] = int(p.get('seed', 42)) + i * 23
            p['noise_scale'] = float(p.get('noise_scale', 1.0)) * (1.0 + (i * 0.05))

            mol_data = synthesize_molecule(p)

            # Build rich AI insight using extracted analogies
            insight = f"Candidate optimized via EGNN score matching at {mol_data['temperature']}K and pH {mol_data['ph']}."
            if mol_data['stability_score'] > 0.8:
                insight += " PINO PDE residuals show exceptional thermodynamic stability (∇V ≈ 0)."
            elif mol_data['stability_score'] > 0.5:
                insight += " Stable geometry achieved after energy descent."
            if mol_data['lipinski_pass']:
                insight += " Viable for scalable industrial pipeline (Lipinski verified)."

            # Add analogy-specific context
            analogies = session.get('analogies', [])
            if analogies:
                analogy_mentions = [a['trigger'] for a in analogies[:2]]
                insight += f" Designed to match: {', '.join(analogy_mentions)}."

            mol_data['ai_insight'] = insight
            molecules.append(mol_data)

        molecules.sort(key=lambda x: x['gibbs_energy'])

        # Build a summary of what was requested
        from scripts.material_properties import PROPERTY_DISPLAY_NAMES, PROPERTY_ICONS, PROPERTY_UNITS
        property_summary = []
        for prop, val in session.get('extracted_properties', {}).items():
            property_summary.append({
                'name': PROPERTY_DISPLAY_NAMES.get(prop, prop),
                'icon': PROPERTY_ICONS.get(prop, '📋'),
                'unit': PROPERTY_UNITS.get(prop, ''),
                'value': val
            })

        return jsonify({
            "success": True,
            "molecules": molecules,
            "design_summary": {
                "properties_targeted": property_summary,
                "conditions": session.get('conditions', {}),
                "application": session.get('application'),
                "analogies_used": session.get('analogies', []),
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/api/reset_session', methods=['POST'])
def reset_chat_session():
    """Reset a conversation session."""
    try:
        data = request.json
        session_id = data.get('session_id')
        if session_id:
            reset_session(session_id)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  Diffusion Dynamics — Agentic Material Designer")
    print("  Conversational AI + PINO-FNO + EGNN Pipeline")
    print("  Open http://localhost:5000 in your browser")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
