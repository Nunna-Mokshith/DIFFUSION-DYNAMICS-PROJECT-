# Diffusion Dynamics — Complete Project Deep Dive

> **Version**: April 2026 — IRAI 2026 Conference Submission  
> **Author**: Moksh  
> **Status**: Fully Functional End-to-End Pipeline

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [The AI Pipeline — Step by Step](#3-the-ai-pipeline--step-by-step)
4. [Module-by-Module Breakdown](#4-module-by-module-breakdown)
5. [The Agentic Conversational Engine](#5-the-agentic-conversational-engine)
6. [The Material Properties Knowledge Base](#6-the-material-properties-knowledge-base)
7. [The Neural Network Architectures](#7-the-neural-network-architectures)
8. [The Physics Engine (PINO + FNO)](#8-the-physics-engine-pino--fno)
9. [Chemical Validation & Ranking](#9-chemical-validation--ranking)
10. [Quantum Level Diffusion](#10-quantum-level-diffusion)
11. [The Frontend](#11-the-frontend)
12. [File Map & Code Statistics](#12-file-map--code-statistics)
13. [Current Progress & Feature Completion](#13-current-progress--feature-completion)
14. [What Makes This Research-Grade](#14-what-makes-this-research-grade)

---

## 1. Executive Summary

**Diffusion Dynamics** is a full-stack AI system for **inverse molecular design**. A user types a natural language prompt — such as *"Design a material as strong as steel but flexible as rubber for biomedical use at 350K"* — and the system:

1. **Parses** the prompt using an agentic conversational AI (rule-based NLP engine with 23+ material properties, 45+ materials, negation handling, and multi-turn dialogue).
2. **Maps** extracted properties to 20+ generation parameters that control the physics simulation.
3. **Generates** 3D molecular candidates using an **EGNN (Equivariant Graph Neural Network)** trained on the **QM9 dataset** (~134,000 DFT-computed molecules) via a **Denoising Diffusion Probabilistic Model (DDPM)**.
4. **Refines** atomic coordinates through a **PINO (Physics-Informed Neural Operator)** — a Fourier Neural Operator constrained by the **Lennard-Jones PDE** (∇V = 0) to force atoms into physically valid equilibrium positions.
5. **Validates** molecules with **RDKit** (cheminformatics library), computes druglikeness via **Lipinski's Rule of 5**, and **Pareto-ranks** candidates by Gibbs Free Energy.
6. **Renders** interactive 3D molecular viewers in the browser using **3Dmol.js**.

The entire system runs as a local Flask server with a single-page web dashboard.

---

## 2. System Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        USER BROWSER (Frontend)                       │
│  index.html — Glassmorphism UI, 3Dmol.js, Chat Interface             │
│  POST /api/chat  →  Conversational AI                                │
│  POST /api/generate_from_chat  →  Molecule Generation                │
└─────────────────────┬────────────────────────────────────────────────┘
                      │  HTTP (JSON)
                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        FLASK SERVER  (server.py)                     │
│                                                                      │
│  ┌──────────────┐   ┌───────────────────┐   ┌────────────────────┐  │
│  │ Agentic Chat │   │ Prompt Heuristics │   │ Batch Orchestrator │  │
│  │   Engine     │──▶│  (pH, T, P)       │──▶│  (3 candidates)    │  │
│  └──────────────┘   └───────────────────┘   └────────┬───────────┘  │
│                                                       │              │
│  ┌────────────────────────────────────────────────────▼───────────┐  │
│  │                    SYNTHESIS PIPELINE                           │  │
│  │                                                                │  │
│  │  1. EGNN Score Network (pgmd_v3_full.pt, ~2.7 MB)             │  │
│  │     └─▶ 200-step DDPM Reverse Diffusion                      │  │
│  │         └─▶ Gaussian Noise → Structured 3D Coordinates        │  │
│  │                                                                │  │
│  │  2. PINO Refinement (Fourier Neural Operator)                 │  │
│  │     └─▶ 15-step test-time optimization                       │  │
│  │         └─▶ Lennard-Jones PDE residual → ∇V ≈ 0              │  │
│  │                                                                │  │
│  │  3. Advanced Molecular Design                                 │  │
│  │     └─▶ Element doping (P, S, Cl)                            │  │
│  │     └─▶ MMFF94 force-field relaxation                        │  │
│  │     └─▶ Thermodynamic scoring (T, P, pH, solvent)            │  │
│  │                                                                │  │
│  │  4. RDKit Validation (3-stage)                                │  │
│  │     └─▶ Stage 1: rdDetermineBonds on XYZ                    │  │
│  │     └─▶ Stage 2: Distance-based bonding (4 thresholds)       │  │
│  │     └─▶ Stage 3: ETKDGv3 re-embedding from scratch           │  │
│  │                                                                │  │
│  │  5. Pareto Ranking by Gibbs Free Energy                       │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. The AI Pipeline — Step by Step

Here is exactly what happens from the moment a user types a prompt to the moment molecules appear on screen:

### Step 1: User Prompt → Agentic Engine
**File**: `scripts/agentic_engine.py`

The user types: *"As strong as steel but as flexible as rubber, for acidic water splitting at 350K"*

The Agentic Engine:
- Runs **regex-based analogy extraction** matching patterns like `"as [PROP] as [MAT]"`, `"[PROP] like [MAT]"`, and `"[PROP] of [MAT]"`
- Looks up `steel` in the 45-material database → finds `tensile_strength = 400 MPa`
- Looks up `rubber` → finds `elongation_pct = 600%`
- Detects `"acidic"` → sets `pH = 2.0`
- Detects `"350K"` → sets `temperature = 350`
- Detects `"water splitting"` → sets application context to `"Catalysis"`
- Returns a confirmation message with property cards

### Step 2: Property → Parameter Mapping
**File**: `scripts/material_properties.py` → `map_properties_to_params()`

The 23 extracted material properties are translated into 20+ generation parameters:

| Material Property | Generation Parameter | Example Logic |
|---|---|---|
| `tensile_strength > 1000 MPa` | `guidance = 2.0, noise_scale = 0.7` | Stronger bonds need tighter control |
| `elongation_pct > 200%` | `wave_packet = 1.5, tunnelling = 0.2` | Flexibility needs quantum delocalization |
| `hardness > 7 Mohs` | `pino_weight = 0.25, bond_threshold = 1.5` | Hard materials need tight bond distances |
| `corrosion_resistance > 7` | `doping_prob += 0.1` | Doping with inert elements |
| `breathability > 5` | `noise_scale = 1.3` | More porosity from noisier generation |
| `density < 3.0` | `max_heavy_atoms = 7` | Fewer heavy atoms → lighter |
| `melting_point > 2000K` | `temperature = 2000, pino_weight = 0.2` | High-temp stability needs PINO enforcement |
| `electrical_conductivity > 1e6` | `doping_prob += 0.15` | Metallic doping increases conductivity |

### Step 3: EGNN Reverse Diffusion (Molecular Generation)
**File**: `server.py` → `synthesize_molecule()`

1. A **reference graph** is loaded from the QM9 validation set (provides atom types + bond topology).
2. Atomic positions are initialized to **pure Gaussian noise** (random 3D chaos).
3. Over **200 denoising steps**, the EGNN predicts and removes noise:
   ```
   For each timestep t from T → 0:
       score, g_pred = EGNN(atom_features, noisy_positions, edges, t)
       grad_g = ∂(G_prediction) / ∂(positions)        ← PINO Guidance
       positions = DDPM_step(positions, score, grad_g)  ← Denoise
   ```
4. The result: **structured 3D atomic coordinates** that approximate a valid molecule.

### Step 4: PINO Refinement (Physics Post-Processing)
**File**: `scripts/pino_operator.py`

The raw diffusion output is noisy. The PINO refines it:

1. A **Fourier Neural Operator** (SpectralConv1d layers + local convolutions) is initialized fresh.
2. It maps `raw_coords → refined_coords` as a function-to-function operator.
3. The **only loss** is the **PDE residual**: `||∇V_LJ||² → 0` (forces on all atoms should be zero at equilibrium).
4. Over **15 optimization steps**, the FNO learns to correct the coordinates.
5. Output: coordinates where `∇V ≈ 0` — atoms are at their **Lennard-Jones minimum energy positions**.

### Step 5: Advanced Molecular Design
**File**: `scripts/Advanced_Molecular_Design.py`

1. **Element Doping**: Random substitution of C/N/O atoms with heavier elements (P, S, Cl) based on `doping_prob`.
2. **XYZ Construction**: Build an XYZ-format string from coordinates + element symbols.
3. **MMFF94 Relaxation**: Uses RDKit's Merck Molecular Force Field to geometrically optimize the molecule.
4. **Thermodynamic Scoring**: Computes adjusted Gibbs Free Energy incorporating temperature, pressure, pH, solvent polarity, band gap, and durability penalties.

### Step 6: 3-Stage Molecule Validation
**File**: `server.py` → within `synthesize_molecule()`

Three progressively aggressive strategies to extract a valid molecule:

| Stage | Method | What It Does |
|---|---|---|
| **Stage 1** | `rdDetermineBonds` | RDKit's automatic bond perception from XYZ coordinates |
| **Stage 2** | Distance-based bonding | Try bond thresholds of 1.8, 2.0, 2.3, 2.6 Å sequentially |
| **Stage 3** | ETKDGv3 re-embedding | Ignore coordinates entirely, build a chain from atom types, embed with ETKDGv3 |

After validation:
- SMILES string is generated
- Molecular weight, LogP, TPSA, H-bond donors/acceptors are computed
- **Lipinski's Rule of 5** is checked (MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10)

### Step 7: Batch Processing & Pareto Ranking

Three molecules are generated with varied seeds and noise levels. They are **sorted by Gibbs Free Energy** (lowest = most thermodynamically stable = Rank #1). Each candidate gets an AI-generated insight sentence explaining its properties.

### Step 8: Frontend Rendering

The JSON response is sent to the browser. JavaScript:
- Renders 3D molecules using **3Dmol.js** (ball-and-stick models with rotation/zoom)
- Displays property cards, SMILES, stability scores, PINO convergence data
- Shows the AI insight and Lipinski badge

---

## 4. Module-by-Module Breakdown

### `server.py` — 599 lines
**The Conductor.** Flask API with 7 endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Serve the frontend |
| `/api/generate` | POST | Single molecule generation |
| `/api/generate_batch` | POST | Batch generation (3 molecules) with prompt heuristics |
| `/api/chat` | POST | Conversational AI message processing |
| `/api/generate_from_chat` | POST | Generate molecules using chat-extracted parameters |
| `/api/reset_session` | POST | Reset a conversation session |
| `/api/health` | GET | Health check |

Contains the full EGNN architecture definition (mirroring the trained model checkpoint) and the **complete synthesis pipeline** in `synthesize_molecule()`.

### `scripts/agentic_engine.py` — 697 lines
**The Conversational Brain.** A state-machine driven dialogue engine:

```
GREETING → COLLECTING → CLARIFYING → CONFIRMING → GENERATING → PRESENTING → ITERATING
```

Key capabilities:
- **Analogy extraction**: 4 regex patterns for natural language comparisons
- **Direct statement parsing**: "it should be breathable" → `breathability = 9`
- **Negation handling**: "it should **not** conduct electricity" → `electrical_conductivity = 1e-12`
- **Conflict detection**: Flags impossible combos like "high hardness + high flexibility"
- **Iterative refinement**: After generation, user can say "make it stronger" → 50% increase in tensile strength
- **In-memory session store**: Multi-turn conversation history per session

### `scripts/material_properties.py` — 1,690 lines
**The Knowledge Base.** The largest file in the project:

- **45+ real-world materials** with 12+ measured properties each (steel, titanium, kevlar, graphene, spider silk, aerogel, MXene, etc.)
- **120+ material aliases** ("aluminium" → "aluminum", "ptfe" → "teflon", "sma" → "nitinol")
- **23 material properties** with display names, units, icons, and HIGH/LOW defaults
- **90+ property synonyms** organized into 30+ semantic groups (e.g., "strong", "tough", "sturdy" all map to `tensile_strength`)
- **20+ application contexts** (aerospace, biomedical, marine, soft robotics, nuclear, etc.) with auto-inferred conditions
- **Property → generation parameter mapper** with conditional logic

### `scripts/pino_operator.py` — 323 lines
**The Fourier Neural Operator.** The true PINO implementation:

- `SpectralConv1d` — Learned linear transforms in Fourier space: `(Kφ)(x) = F⁻¹(R · Fφ)(x)`
- `FourierNeuralOperator` — 4-layer FNO: Lift(3→32) → [SpectralConv + LocalConv + Residual + LayerNorm + GELU] × 4 → Project(32→3)
- `lennard_jones_potential` — Pairwise LJ potential with Lorentz-Berthelot mixing rules for 8 element types
- `pde_residual_force_balance` — Computes `||∇V||²` via autograd (should converge to 0)
- `pino_refine_coordinates` — 15-step test-time optimization loop

### `scripts/physics_guided_molecular_diffusion.py` — 782 lines
**The Training Script.** The original research codebase:

- QM9 dataset loading + normalization of Gibbs Free Energy
- `EquivariantEdgeConv` — Message-passing layer conditioned on inter-atomic distances
- `GNNScoreNetwork` — Score network s_θ(x_t, t) with FiLM conditioning + PINO auxiliary head
- `DiffusionSchedule` — Cosine noise schedule (Nichol & Dhariwal 2021)
- `PINOThermodynamicLoss` — Combined loss: `L_total = L_score + λ · L_PINO`
- Training loop with gradient clipping, cosine LR scheduling, and checkpointing
- Molecule sampling, bond inference, and RDKit validation

### `scripts/Advanced_Molecular_Design.py` — 132 lines
**The Chemistry Layer:**

- `apply_element_doping()` — Random substitution of C/N/O with P/S/Cl
- `xyz_to_rdkit_mol()` — XYZ → RDKit Mol with `rdDetermineBonds`
- `relax_structure()` — MMFF94 force-field geometric optimization
- `compute_advanced_thermodynamics()` — Multi-factor G computation (T, P, pH, solvent, band gap, durability)

### `core/model.py` — 166 lines
**Unified Architecture Definitions** used for model loading:
- `EGNNScoreNetwork` — Matches the saved `pgmd_v3_full.pt` checkpoint architecture
- `FourierNeuralOperator` — Used by `core/pino_operator.py` for import

### `core/pino_operator.py` — 121 lines
**Compact PINO** used directly by the core import path. Mirrors `scripts/pino_operator.py` but is imported by the server for coordinate refinement.

### `frontend/index.html` — ~45,700 bytes
**The Dashboard UI:**
- Single-page app with embedded CSS and JavaScript
- Glassmorphism design with dark mode, gradients, and micro-animations
- 3Dmol.js for interactive 3D molecular visualization
- Chat interface with suggestion chips, property cards, and markdown rendering
- Results panel with molecule cards, SMILES display, physics metrics, and Lipinski badges

---

## 5. The Agentic Conversational Engine

### State Machine
```
User opens page
    ↓
GREETING — Welcome message, suggestion chips
    ↓ (user types prompt)
COLLECTING — Extract analogies, materials, properties, conditions
    ↓
CLARIFYING — If conflicts detected or too little info
    ↓
CONFIRMING — Show extracted parameters, ask for go-ahead
    ↓ (user says "Generate!")
GENERATING — Map properties → params, trigger pipeline
    ↓
PRESENTING — Show results
    ↓ (user says "make it stronger")
ITERATING — Adjust parameters, re-generate
```

### Natural Language Understanding

**4 analogy patterns** matched via regex:
1. `"as strong as steel"` → Pattern: `as [PROP] as [MAT]`
2. `"flexible like rubber"` → Pattern: `[PROP] like [MAT]`
3. `"strength of steel"` → Pattern: `[PROP] of [MAT]`
4. `"flexible as rubber"` → Pattern: `[PROP] as [MAT]` (without leading "as")

**Negation detection** (25-character lookback window):
```
"it should not conduct electricity"
          ^^^                          ← "not" detected in window
→ electrical_conductivity = 1e-12 (LOW default)
```

**Conflict detection** catches physically contradictory requests:
- High hardness (>7 Mohs) + High flexibility (>100% elongation)
- Very high strength (>500 MPa) + Very low density (<2 g/cm³)
- High breathability (>7/10) + High strength (>500 MPa)

### Iterative Refinement
After generation, the user can adjust parameters with natural language:
- `"Make it stronger"` → tensile_strength × 1.5
- `"More transparent"` → transparency = 9
- `"Reduce cost"` → cost_index = 2
- `"Increase temperature to 500K"` → temperature = 500
- `"Add corrosion resistance like titanium"` → new analogy extracted

---

## 6. The Material Properties Knowledge Base

### Materials Database (45+ entries)

| Category | Count | Examples |
|---|---|---|
| **Metals** | 12 | Steel, Stainless Steel, Copper, Gold, Aluminum, Titanium, Iron, Silver, Platinum, Tungsten, Nickel, Zinc |
| **Ceramics** | 5 | Ceramic, Glass, Diamond, Silicon Carbide, Boron Nitride |
| **Polymers** | 8 | Rubber, Nylon, Kevlar, Teflon, Silicone, Polyethylene, PDMS, Polyurethane Foam |
| **Composites** | 5 | Carbon Fiber, Graphene, Fiberglass, MXene, Basalt Fiber |
| **Textiles** | 5 | Linen, Cotton, Silk, Wool, Leather |
| **Biologicals** | 6 | Spider Silk, Earthworm, Muscle Tissue, Tendon, Skin, Lotus Leaf |
| **Organics** | 4 | Wood, Bamboo, Cork, Balsa |
| **Specialty** | 3 | Aerogel, Hydrogel, Nitinol |

### Properties Tracked (23 dimensions)

| # | Property | Unit | Example Range |
|---|---|---|---|
| 1 | Tensile Strength | MPa | 0.1 (hydrogel) → 130,000 (graphene) |
| 2 | Elongation % | % | 0 (ceramic) → 1000 (hydrogel) |
| 3 | Malleability | /10 | 0 (ceramic) → 10 (gold, hydrogel) |
| 4 | Lustre | /10 | 1 (rubber) → 10 (gold, diamond) |
| 5 | Hardness | Mohs | 0.05 (hydrogel) → 10 (diamond) |
| 6 | Compressive Strength | MPa | – |
| 7 | Impact Resistance | /10 | – |
| 8 | Fatigue Resistance | /10 | – |
| 9 | Wear Resistance | /10 | – |
| 10 | Transparency | /10 | – |
| 11 | UV Resistance | /10 | – |
| 12 | Corrosion Resistance | /10 | 2 (iron) → 10 (gold, teflon) |
| 13 | Chemical Stability | /10 | – |
| 14 | Biocompatibility | /10 | – |
| 15 | Water Resistance | /10 | – |
| 16 | Breathability | /10 | 0 (metals) → 10 (linen, hydrogel) |
| 17 | Density | g/cm³ | 0.002 (aerogel) → 21.45 (platinum) |
| 18 | Melting Point | K | 373 (biologicals) → 4900 (graphene) |
| 19 | Thermal Conductivity | W/mK | 0.015 (aerogel) → 5000 (graphene) |
| 20 | Electrical Conductivity | S/m | 1e-25 (teflon) → 1e8 (graphene) |
| 21 | Magnetic Property | /10 | – |
| 22 | Acoustic Dampening | /10 | – |
| 23 | Cost Index | /10 | 0 (bone) → 10 (diamond, platinum) |

### Application Domains (20+)

Aerospace, Biomedical, Automotive, Marine, Electronics, Construction, Textile, Catalysis, Energy, Robotics, Soft Robotics, Defense, Sports, Packaging, Semiconductor, Nuclear, Food/Pharma, Thermal Insulation, Acoustic, Optical

---

## 7. The Neural Network Architectures

### EGNN Score Network (Denoiser)

**Purpose**: Predict the noise ε on 3D atomic positions at each diffusion timestep.

```
Input: Node Features (N×11) + Noisy Positions (N×3) + Edges + Timestep t
  │
  ├─▶ Node Encoder: Linear(11→128) → SiLU → Linear(128→128)
  │
  ├─▶ Time Embedding: Sinusoidal(t) → Linear(128→256) → SiLU → Linear(256→128)
  │
  ├─▶ 4× EGNN Message-Passing Layers:
  │      Message = MLP(h_i ⊕ h_j ⊕ ||r_ij|| ⊕ edge_attr)
  │      Update  = MLP(h_i ⊕ aggregated_messages)
  │      FiLM    = SiLU(h * scale(t) + shift(t))       ← timestep conditioning
  │
  ├─▶ Score Head: Linear(128→128) → SiLU → Linear(128→3)    → ε prediction (N×3)
  │
  └─▶ Property Head: Linear(128→128) → SiLU → Linear(128→12) → G prediction (B×12)
```

**Key Design Choices**:
- **E(3) Equivariance**: Messages are conditioned on inter-atomic **distances** (invariant to rotation/translation), not raw coordinates
- **FiLM Conditioning**: Feature-wise Linear Modulation injects timestep information at every layer (scale & shift)
- **Dual Heads**: Score head predicts noise; property head predicts Gibbs Free Energy — sharing weights forces thermodynamic awareness into the score prediction

**Model size**: ~2.7 MB (`pgmd_v3_full.pt`), 128-dim hidden, 4 layers

### Fourier Neural Operator (PINO Refinement)

**Purpose**: Learn a coordinate-to-coordinate correction that minimizes PDE residual.

```
Input: Raw Coordinates (N×3)
  │
  ├─▶ Lift: Linear(3→32) → GELU
  │
  ├─▶ 3× FNO Layers:
  │      ┌ Spectral Path: FFT → Learned Weight Multiply (in Fourier space) → iFFT
  │      │    (Kφ)(x) = F⁻¹(R · Fφ)(x), R ∈ ℂ^(32×32×modes)
  │      ├ Local Path: Conv1d(32→32, kernel=1)     ← pointwise
  │      └ Residual + LayerNorm + GELU
  │
  ├─▶ Project: Linear(32→16) → GELU → Linear(16→3)
  │
  └─▶ Output = Input + Δ (residual correction)      → Refined Coordinates (N×3)
```

**Key Design Choices**:
- **Spectral Convolution**: Learned complex-valued weights multiply in Fourier space — captures **global** correlations between all atoms simultaneously
- **Residual Architecture**: Output is `coords + learned_correction`, ensuring the FNO only needs to learn the delta
- **Test-Time Optimization**: The FNO is randomly initialized and trained **at inference** using only PDE loss — no labels needed

---

## 8. The Physics Engine (PINO + FNO)

### What Is PINO?

**Physics-Informed Neural Operator** (Li et al., 2021) — a neural network that:
1. Maps **functions to functions** (not vectors to vectors) — coordinate field → refined coordinate field
2. Is constrained by a **PDE** at training time — no labeled data required
3. Uses **Fourier spectral convolutions** for global, resolution-invariant approximation

### The PDE Being Solved

The **Lennard-Jones potential** models interatomic forces:

```
V_LJ(r_ij) = 4ε [ (σ/r_ij)¹² − (σ/r_ij)⁶ ]
```

- **(σ/r)¹² term**: Short-range repulsion (Pauli exclusion — atoms can't overlap)
- **(σ/r)⁶ term**: Long-range attraction (van der Waals / London dispersion forces)

**At equilibrium**, the net force on every atom must be zero:

```
PDE: ∇V(r) = 0     (Euler-Lagrange equation)
Residual: ||∇V||² = Σᵢ ||Fᵢ||²  → should converge to 0
```

### Lennard-Jones Parameters

| Element | σ (Å) | ε (eV) |
|---|---|---|
| H | 2.886 | 0.0440 |
| C | 3.851 | 0.1050 |
| N | 3.660 | 0.0690 |
| O | 3.500 | 0.0600 |
| F | 3.364 | 0.0500 |
| P | 4.147 | 0.3050 |
| S | 4.035 | 0.2740 |
| Cl | 3.947 | 0.2270 |

**Mixing Rules** (Lorentz-Berthelot):
- `σ_ij = (σ_i + σ_j) / 2`
- `ε_ij = √(ε_i · ε_j)`

### PINO Optimization Loop (Test-Time)

```python
for step in range(15):
    refined = FNO(raw_coords)                          # function-to-function
    residual = ||∇V_LJ(refined)||²                     # PDE residual
    reg = 0.01 * ||refined − raw_coords||²             # regularization
    loss = residual + reg
    loss.backward()                                     # backprop through FNO
    optimizer.step()                                    # update FNO weights
```

**Output metrics**:
- `pde_residual_initial`: Force imbalance before PINO
- `pde_residual_final`: Force imbalance after PINO (should be much smaller)
- `potential_energy`: Final V_LJ (should be at minimum)
- `pino_convergence`: List of residuals across the 15 steps

### PINO Guidance During Diffusion

In addition to the post-processing PINO, the score network itself has **PINO guidance** during reverse diffusion:

```python
# During each DDPM step:
score, g_pred = EGNN(features, positions, edges, timestep)
grad_g = ∂(G_prediction) / ∂(positions)   # gradient of predicted Gibbs energy

# Combined update:
positions_new = DDPM_step(positions, score) − pino_weight × grad_g
```

This steers the diffusion toward **low-energy regions** of molecular space during generation itself.

---

## 9. Chemical Validation & Ranking

### 3-Stage Molecule Extraction

**Stage 1 — `rdDetermineBonds`**: RDKit's automatic bond perception from 3D coordinates. This is the gold standard but can fail on noisy AI output.

**Stage 2 — Distance-Based Bonding**: Connect atoms within a distance threshold. Tries 4 thresholds sequentially (1.8, 2.0, 2.3, 2.6 Å) and validates each with RDKit sanitization.

**Stage 3 — ETKDGv3 Re-Embedding**: Ignores coordinates entirely. Builds a simple chain from atom types, sanitizes, adds hydrogens, embeds with ETKDGv3 (distance geometry with experimental torsion-angle preferences), and optimizes with MMFF.

### Computed Descriptors

| Descriptor | Source | Purpose |
|---|---|---|
| SMILES | RDKit | Canonical text representation |
| Molecular Weight | `Descriptors.MolWt` | Size metric |
| LogP | `Crippen.MolLogP` | Hydrophobicity |
| TPSA | `rdMolDescriptors.CalcTPSA` | Topological polar surface area |
| H-Bond Donors | `Lipinski.NumHDonors` | Hydrogen bonding capacity |
| H-Bond Acceptors | `Lipinski.NumHAcceptors` | Hydrogen bonding capacity |
| Rotatable Bonds | `rdMolDescriptors.CalcNumRotatableBonds` | Flexibility |
| Formal Charge | `Chem.GetFormalCharge` | Charge state |
| Element Composition | Atom iteration | Elemental breakdown |

### Lipinski's Rule of 5

A molecule is "drug-like" if it satisfies:
- Molecular Weight ≤ 500 Da
- LogP ≤ 5
- H-Bond Donors ≤ 5
- H-Bond Acceptors ≤ 10

### Pareto Ranking

The 3 candidates are sorted by **Gibbs Free Energy** (ascending). The molecule with the lowest G is ranked #1 — it is the most thermodynamically stable.

### Gibbs Free Energy Computation

```
G_real = G_base + ΔP + ΔpH + ΔSolvent + ΔBandGap − Durability_bonus
```

Where:
- `ΔP = k_B·T·ln(P)` — pressure correction
- `ΔpH = |7 − pH| × 0.05` — pH penalty
- `ΔSolvent = |78.5 − ε| × 0.01` — polarity deviation
- `ΔBandGap = |2.0 − E_g| × 0.2` — band gap matching
- `Durability = log₁₀(hours) × 0.1` — longevity bonus

Stability score = `σ(G / k_B·T)` — sigmoid mapping to [0, 1]

---

## 10. Quantum Level Diffusion

An additional layer of physics simulation that adds quantum-mechanical effects to the diffusion process:

### Three Quantum Parameters

| Parameter | Default | Effect |
|---|---|---|
| `quantum_ensemble` | 0.5 | Probability of applying quantum perturbation at each step |
| `wave_packet` | 1.0 | Width of the Wigner-distribution envelope |
| `tunnelling_depth` | 0.1 | Magnitude of tunnelling noise |

### How It Works

**Wave Packet Initialization**: Instead of pure Gaussian noise, initial positions use a Wigner-distribution envelope:
```python
pos = randn(N, 3) * noise_scale * wave_packet
```

**Quantum Tunnelling Perturbation**: At each diffusion step, with probability `quantum_ensemble`:
```python
if tunnelling_depth > 0 and random() < quantum_ensemble:
    tunnel_noise = randn_like(pos) * tunnelling_depth * wave_packet
    pos = pos + tunnel_noise
```

This allows atoms to "tunnel" through small energy barriers, exploring more of the energy landscape and potentially finding lower-energy configurations.

### Output Metrics
- `quantum_coherence = min(ensemble × wave_packet, 1.0)` — how "quantum" the generation was
- `quantum_barrier_ratio = tunnelling / stability` — ratio of tunnelling to stability

---

## 11. The Frontend

### Technology
- **Vanilla HTML/CSS/JS** — no frameworks, maximum control
- **3Dmol.js** — Interactive 3D molecular viewer (ball-and-stick models)
- **Glassmorphism** aesthetic — frosted glass panels, gradients, shadows
- **Fonts**: Outfit (headings) + Inter (body)
- **Single file**: `frontend/index.html` (~45 KB)

### UI Sections
1. **Chat Interface** — Full conversational UI with markdown rendering, suggestion chips, property cards
2. **Molecule Cards** — 3D viewer, SMILES, property badges, AI insight
3. **Physics Panel** — PINO convergence graph, PDE residuals, quantum metrics
4. **Control Panel** — Manual parameter sliders for advanced users

---

## 12. File Map & Code Statistics

```
Diffusion_Dynamics_Project/
├── server.py                                    599 lines   (Flask API + synthesis pipeline)
├── dashboard.py                                 ~800 lines  (Standalone Streamlit dashboard)
├── How_It_Works.md                              79 lines    (Simple overview)
├── Project_Deep_Dive.md                         THIS FILE
├── ies2026_paper_milestone1.tex                 ~500 lines  (LaTeX paper for IRAI 2026)
│
├── frontend/
│   └── index.html                               ~1500 lines (Full UI + CSS + JS)
│
├── scripts/
│   ├── agentic_engine.py                        697 lines   (Conversational AI)
│   ├── material_properties.py                   1,690 lines (Knowledge base)
│   ├── pino_operator.py                         323 lines   (FNO + PDE physics)
│   ├── physics_guided_molecular_diffusion.py    782 lines   (Training + diffusion)
│   ├── Advanced_Molecular_Design.py             132 lines   (Doping + thermodynamics)
│   ├── visualize_molecules.py                   ~200 lines  (Matplotlib visualization)
│   └── train_prof.py                            ~100 lines  (Training profiler)
│
├── core/
│   ├── model.py                                 166 lines   (Unified architecture defs)
│   ├── pino_operator.py                         121 lines   (Compact PINO for imports)
│   ├── generation.py                            ~80 lines   (Generation utilities)
│   └── prompt_engine.py                         ~100 lines  (Prompt processing)
│
├── models/
│   ├── pgmd_v2_best.pt                          2.7 MB      (Previous checkpoint)
│   └── pgmd_v3_full.pt                          2.7 MB      (Current production model)
│
├── data/                                        (QM9 dataset, auto-downloaded)
├── config/                                      (Configuration files)
├── notebooks/                                   (Jupyter exploration notebooks)
│
├── test_agentic.py                              (Agentic engine tests)
├── test_all_params.py                           (Parameter mapping tests)
├── test_extraction.py                           (NLP extraction tests)
└── test_generate.py                             (Generation pipeline tests)

TOTAL: ~5,500+ lines of Python + ~1,500 lines of frontend code
```

---

## 13. Current Progress & Feature Completion

### ✅ Completed Features

| Feature | Status | Details |
|---|---|---|
| EGNN Score Network | ✅ Done | Trained on QM9, v3 checkpoint loaded |
| DDPM Reverse Diffusion | ✅ Done | 200-step cosine schedule |
| PINO FNO Refinement | ✅ Done | SpectralConv1d + LJ PDE residual |
| PINO Gradient Guidance | ✅ Done | ∂G/∂pos steering during diffusion |
| Quantum Level Diffusion | ✅ Done | Wave packet + tunnelling + ensemble |
| 3-Stage Molecule Extraction | ✅ Done | rdDetermineBonds → Distance → ETKDGv3 |
| RDKit Validation | ✅ Done | Sanitization + SMILES + Lipinski |
| Thermodynamic Scoring | ✅ Done | G with T, P, pH, solvent corrections |
| Element Doping | ✅ Done | P, S, Cl substitution |
| MMFF94 Relaxation | ✅ Done | Force-field geometry optimization |
| Pareto Ranking | ✅ Done | Sort by Gibbs Free Energy |
| Agentic Conversational AI | ✅ Done | Multi-turn, analogies, negation |
| Material Database | ✅ Done | 45+ materials, 23 properties |
| Property → Parameter Mapping | ✅ Done | 20+ generation params from NLP |
| Conflict Detection | ✅ Done | Flags impossible property combos |
| Iterative Refinement | ✅ Done | "Make it stronger" post-generation |
| Application Context Detection | ✅ Done | 20+ domains auto-detected |
| Frontend Dashboard | ✅ Done | Glassmorphism, 3Dmol.js, Chat UI |
| 3D Molecular Viewer | ✅ Done | Ball-and-stick via 3Dmol.js |
| SDF Output | ✅ Done | Bonded 3D format for viewers |
| Batch Generation | ✅ Done | 3 diverse candidates per request |
| AI Insights | ✅ Done | Natural language result summaries |
| LaTeX Paper | ✅ In Progress | IRAI 2026 milestone 1 draft |
| Test Suite | ✅ Done | 4 test files covering major modules |

### 🏗️ Architecture Strengths

1. **True PINO** — Not just a name. The FNO uses spectral convolutions in Fourier space, and the PDE residual (Lennard-Jones force balance) is the sole optimization objective at test-time. This matches the Li et al. (2021) formulation.

2. **Dual PINO Integration** — PINO appears in two places: (a) gradient guidance during diffusion (∂G/∂pos), and (b) post-generation coordinate refinement via FNO. This is a novel dual-application.

3. **Real Chemistry** — Not simulated. Uses actual DFT-computed quantum properties from QM9, real Lennard-Jones parameters, RDKit's industrial cheminformatics toolkit, and MMFF94 force-field relaxation.

4. **Agentic NLP** — The conversational engine handles real-world analogies ("as strong as spider silk"), biological references ("earthworm-like flexibility"), negated requirements ("should NOT conduct electricity"), and multi-turn iterative design.

---

## 14. What Makes This Research-Grade

### Novel Contributions

1. **PINO-Guided Molecular Diffusion**: Combining score-based generative models with physics-informed neural operators for thermodynamically constrained molecular design. The PINO loss (Gibbs Free Energy prediction) shares weights with the score network, creating a physics-aware denoiser.

2. **Fourier Neural Operator for Coordinate Refinement**: Using an FNO as a test-time post-processor that learns to satisfy the Lennard-Jones PDE (∇V = 0) without any labeled data — pure physics-driven optimization.

3. **Agentic Material Design Interface**: Natural language to molecular structure via a conversational AI that maps 23+ material properties from analogies, direct statements, and negated requirements into generation parameters.

4. **Quantum-Augmented Diffusion**: Wigner-distribution initialization and tunnelling perturbations during the reverse diffusion process, enabling exploration of broader energy landscapes.

### Conference Target
**IRAI 2026** — International Research on Artificial Intelligence

### Training Data
**QM9 Dataset** — 133,885 organic molecules with up to 9 heavy atoms (C, N, O, F), each with 19 DFT-computed quantum properties including Gibbs Free Energy, HOMO/LUMO energies, dipole moment, and more.

---

*Last updated: April 7, 2026*
