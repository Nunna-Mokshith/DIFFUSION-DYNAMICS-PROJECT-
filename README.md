# Diffusion Dynamics — Agentic Material Designer

**Diffusion Dynamics** is a full-stack AI system for **inverse molecular design**. This project was developed as a submission for the IRAI 2026 Conference. It allows a user to type a natural language prompt — such as *"Design a material as strong as steel but flexible as rubber for biomedical use at 350K"* — and generates thermodynamically stable, valid 3D molecular structures.

## System Pipeline Overview

1. **Agentic Conversational Engine**: Parses the user prompt using an advanced NLP engine that understands analogies, negations, and direct statements about 23+ different material properties across a built-in database of 45+ reference materials.
2. **Parameter Mapping**: Translates extracted properties into 20+ physics parameters for generation.
3. **Molecular Generation (EGNN)**: An Equivariant Graph Neural Network (trained on the QM9 dataset) generates 3D molecular candidates using a 200-step Denoising Diffusion Probabilistic Model (DDPM).
4. **Physics Optimization (PINO)**: A Physics-Informed Neural Operator refines the atomic coordinates in a 15-step process constrained by the Lennard-Jones PDE (∇V = 0), forcing atoms into true physical equilibrium.
5. **Validation & Assessment**: Three-stage RDKit validation, Pareto-level thermodynamic ranking based on adjusted Gibbs Free Energy, and assessment via Lipinski's Rule of 5.
6. **Web Dashboard**: A single-page, glassmorphism-themed frontend to visualize 3D molecules using 3Dmol.js alongside a robust chat interface.

## Further Documentation

For detailed information about the inner workings, algorithms, and data flow of this application, please explore the existing documentation pieces:
- `Project_Deep_Dive.md` - Complete deep dive into every module, architecture choice, and physics calculation.
- `How_It_Works.md` - A simpler breakdown.
- `Progress_Update.md` - Information about completion limits.
- `ies2026_paper_milestone1.tex` - The actual LaTeX paper for IRAI 2026.

## Running the Application Locally

1. Install Python dependencies (such as Flask, RDKit, PyTorch, etc.).
2. Execute the server:
   ```bash
   python server.py
   ```
3. Open a browser and visit `http://127.0.0.1:5000` to access the Material Designer dashboard.

---
*Created for the IRAI 2026 Conference by Moksh.*
