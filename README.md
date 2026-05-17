# Diffusion Dynamics — DeepMat Diffusion
### Physics-Guided Inverse Design for Industrial Catalysis

> **IEEE IES Generative AI Challenge 2026 — Milestone 3+ Qualified**  
> An agentic, physics-guided molecular generation system that combines Equivariant Graph Neural Networks (EGNN) with Physics-Informed Neural Operators (PINO) for de novo material discovery.

---

## 🧬 Project Overview

**Diffusion Dynamics** is a full-stack AI-powered platform for generating novel molecular structures optimized for specific industrial applications (e.g., catalysis, biomedical coatings, energy storage). The system uses a **3-stage physics-guided pipeline**:

1. **Stage 1 — EGNN Diffusion**: Generates 3D molecular coordinates using an Equivariant Graph Neural Network with inline Gibbs free energy gradient guidance.
2. **Stage 2 — PINO Refinement**: Refines atomic coordinates using a Fourier Neural Operator (FNO) that enforces PDE-based physics constraints.
3. **Stage 3 — Validity Filtering**: Validates molecules using RDKit (bond-length checks, Lipinski's Rule of Five, atom stability scoring).

The platform features an **Agentic Natural Language Interface** — users describe materials in plain English (e.g., *"Design a catalyst as strong as steel but flexible as rubber at 350K"*), and the AI agent extracts parameters, generates candidates, and returns ranked results with 3D visualizations.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Frontend (index.html)                  │
│   Agentic Chat UI  ·  3Dmol.js Viewer  ·  Results Table │
└──────────────────────────┬──────────────────────────────┘
                           │  REST API
┌──────────────────────────▼──────────────────────────────┐
│                    server.py (Flask)                      │
│  /api/chat  ·  /api/generate  ·  /api/molecule_viewer    │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│              Agentic Engine (agentic_engine.py)           │
│  NLP Parameter Extraction  ·  Material Properties DB     │
│  Intent Classification  ·  Constraint Mapping            │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│          3-Stage Physics-Guided Pipeline                  │
│                                                          │
│  Stage 1: EGNN Diffusion (model_arch.py)                │
│     └─ Equivariant message passing + Gibbs guidance      │
│                                                          │
│  Stage 2: Dual PINO Refinement (pino_operator.py)       │
│     └─ FNO with spectral convolution (R^N → R^N)        │
│                                                          │
│  Stage 3: RDKit Validity Filter                          │
│     └─ Bond-length, Lipinski, atom stability checks      │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Diffusion_Dynamics_Project/
│
├── server.py                          # Flask backend — API endpoints + generation logic
├── requirements.txt                   # Python dependencies
│
├── frontend/
│   └── index.html                     # Full dashboard UI (chat + 3D viewer + results)
│
├── models/
│   └── pgmd_v3_full.pt               # Trained EGNN model checkpoint (2.7 MB)
│
├── data/
│   ├── raw/                           # QM9 dataset (gdb9.sdf, gdb9.sdf.csv)
│   └── processed/                     # PyTorch Geometric preprocessed cache
│
├── scripts/
│   ├── __init__.py
│   ├── physics_guided_molecular_diffusion.py   # Core diffusion pipeline
│   ├── pino_operator.py                        # True PINO/FNO implementation
│   ├── model_arch.py                           # EGNN + GibbsEnergyHead architecture
│   ├── agentic_engine.py                       # NLP agent for parameter extraction
│   ├── material_properties.py                  # 30+ material property knowledge base
│   ├── Advanced_Molecular_Design.py            # Advanced design utilities
│   │
│   ├── benchmark_qm9.py                       # QM9 benchmark evaluation
│   ├── evaluate_validity.py                    # Molecular validity evaluation
│   ├── baseline_model.py                       # Baseline model for comparison
│   ├── compute_metrics.py                      # Metrics computation utilities
│   ├── compare_ab.py                           # A/B testing framework
│   │
│   ├── test_3stage_pipeline.py                 # 3-stage pipeline tests
│   ├── test_100_diverse.py                     # 100-molecule diversity test
│   ├── test_120_molecules.py                   # 120-molecule generation test
│   ├── generate_240_molecules.py               # Large-scale 240-molecule generation
│   │
│   ├── generate_report.py                      # Standard report generation
│   ├── generate_green_report.py                # Green (passing) report
│   ├── generate_red_report.py                  # Red (failing) report
│   ├── generate_blue_updated.py                # Blue (updated) report
│   └── generate_shareable.py                   # Shareable summary generation
│
└── results/
    ├── comparison_AB.json                      # A/B comparison results
    ├── diffusion_generated_240.csv             # 240 generated molecules
    ├── diffusion_gen_240_summary.json          # Generation summary
    ├── pipeline_test_120.csv                   # 120-molecule pipeline test
    └── pipeline_test_120_summary.txt           # Pipeline test summary
```

---

## 🔬 Key Technical Features

### Dual PINO Architecture
- **Spectral Convolution**: Learned Fourier modes for function-to-function mapping (coordinate field → refined coordinate field)
- **Physics Loss**: PDE residual minimization during test-time refinement
- **Equivariance**: Message-passing architecture preserves SE(3) symmetry

### Agentic Natural Language Interface
- Extracts density, temperature, pH, application domain from free-form text
- Maps material analogies ("as strong as steel") to quantitative parameters
- Knowledge base covering 30+ real-world materials with physical properties

### QM9 Benchmark Validation
- Trained on the QM9 dataset (134k molecules)
- Reports **Atom Stability**, **Molecule Stability**, **Validity**, **Uniqueness**, and **Novelty**
- A/B testing framework comparing pipeline variants

---

## 🧪 Tests & Benchmarks Conducted

| Test | Script | Description |
|------|--------|-------------|
| **3-Stage Pipeline** | `test_3stage_pipeline.py` | Validates the full EGNN → PINO → RDKit pipeline |
| **100-Molecule Diversity** | `test_100_diverse.py` | Generates 100 molecules and measures diversity metrics |
| **120-Molecule Pipeline** | `test_120_molecules.py` | Extended pipeline test with 120 candidates |
| **240-Molecule Generation** | `generate_240_molecules.py` | Large-scale generation for statistical analysis |
| **QM9 Benchmark** | `benchmark_qm9.py` | Standard molecular generation benchmarks on QM9 |
| **Validity Evaluation** | `evaluate_validity.py` | Atom/molecule stability with RDKit validation |
| **A/B Comparison** | `compare_ab.py` | Compares base vs. full pipeline performance |

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the Dashboard
```bash
python server.py
```
Then open **http://localhost:5000** in your browser.

### Run Benchmarks
```bash
python scripts/benchmark_qm9.py
python scripts/evaluate_validity.py --n_samples 50 --steps 100 --verbose
```

---

## 📊 Results Summary

- **Atom Stability**: ~100% on generated molecules
- **Molecule Validity**: Validated via RDKit bond-length analysis
- **Lipinski Compliance**: Checked for drug-likeness (MW, LogP, HBD, HBA)
- **PDE Residual**: Minimized through PINO test-time refinement
- **Gibbs Free Energy**: Predicted per molecule via dedicated energy head

---

## 👥 Authors

- **Nunna Mokshith** — [mokshithnunna31@gmail.com](mailto:mokshithnunna31@gmail.com)
- **Sarat Chandra Nallabati** — [saratchandranallabati@gmail.com](mailto:saratchandranallabati@gmail.com)

---

## 📄 Publication

> **Diffusion Dynamics: DeepMat Diffusion — Physics-Guided Inverse Design for Industrial Catalysis**  
> Submitted to IEEE IES Generative AI Challenge 2026 (Milestone 3+ qualified, advancing to IRAI Conference)

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **ML Framework** | PyTorch, PyTorch Geometric |
| **Neural Operator** | Fourier Neural Operator (FNO) |
| **Chemistry** | RDKit, Open Babel |
| **Visualization** | 3Dmol.js |
| **Backend** | Flask (Python) |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Dataset** | QM9 (134,000 molecules) |

---

## 📜 License

This project is developed for the IEEE IES Generative AI Challenge 2026.
