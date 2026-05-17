# -*- coding: utf-8 -*-
"""
Generate Updated Blue Report — Diffusion Model Generated Molecules
===================================================================
Uses results from generate_240_molecules.py (actual model inference),
NOT known SMILES through RDKit.

Saves:
  - the test results_Blue (updated).html  on Desktop
  - the test results_Blue (updated).pdf   via Edge headless
"""

import csv, os, json

CSV_PATH  = r'c:\Users\moksh\OneDrive\Desktop\DiffusionDashboard\Diffusion_Dynamics_Project\results\diffusion_generated_240.csv'
JSON_PATH = r'c:\Users\moksh\OneDrive\Desktop\DiffusionDashboard\Diffusion_Dynamics_Project\results\diffusion_gen_240_summary.json'
OUT_HTML  = r'c:\Users\moksh\OneDrive\Desktop\the test results_Blue (updated).html'

# Load data
rows = []
with open(CSV_PATH, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

with open(JSON_PATH) as f:
    summary = json.load(f)

N         = summary["n_total"]
n_valid   = summary["n_valid"]
val_pct   = summary["validity_pct"]
n_unique  = summary["n_unique"]
uniq_pct  = summary["unique_pct"]
avg_stab  = summary["avg_atom_stab"]
lip_pct   = summary["lipinski_pct"]
avg_gibbs = summary["avg_gibbs"]
stages    = summary["stages"]
n_N       = summary["n_with_N"]
n_O       = summary["n_with_O"]
n_F       = summary["n_with_F"]
elem_cov  = summary["element_coverage"]
total_t   = summary["total_time_s"]

def pct(a, b): return f"{a/max(b,1)*100:.1f}%"

def badge(val):
    ok = str(val) == "True"
    color = "#22c55e" if ok else "#ef4444"
    label = "VALID" if ok else "FAIL"
    return f'<span style="background:{color};color:#fff;padding:2px 9px;border-radius:20px;font-size:11px;font-weight:700">{label}</span>'

def stab_bar(pct_val):
    try: v = float(pct_val)
    except: v = 0
    color = "#22c55e" if v >= 90 else "#f59e0b" if v >= 70 else "#ef4444"
    return f'<div style="display:flex;align-items:center;gap:6px"><div style="width:60px;height:8px;background:#1e3a5f;border-radius:4px"><div style="width:{v*0.6:.0f}px;height:8px;background:{color};border-radius:4px"></div></div><span style="font-size:11px">{v:.0f}%</span></div>'

def elem_badges(r):
    badges = []
    if r.get("has_N") == "True":
        badges.append('<span style="background:#7c3aed;color:#fff;padding:1px 6px;border-radius:10px;font-size:9px;font-weight:700">N</span>')
    if r.get("has_O") == "True":
        badges.append('<span style="background:#dc2626;color:#fff;padding:1px 6px;border-radius:10px;font-size:9px;font-weight:700">O</span>')
    if r.get("has_F") == "True":
        badges.append('<span style="background:#0891b2;color:#fff;padding:1px 6px;border-radius:10px;font-size:9px;font-weight:700">F</span>')
    return " ".join(badges) if badges else '<span style="color:#475569;font-size:9px">C,H</span>'

# Build table rows
table_rows_html = ""
for r in rows:
    row_bg = "#071428" if int(r['id'])%2==0 else "#091830"
    smi = (r["gen_smiles"] or "-")[:40]
    gibbs = f"{float(r['gibbs_pred']):.3f}" if r["gibbs_pred"] else "-"
    lip_ok = "Yes" if r["lipinski"] == "True" else "No"
    lip_color = "#22c55e" if r["lipinski"] == "True" else "#ef4444"

    table_rows_html += f'''<tr style="background:{row_bg}">
  <td style="text-align:center;color:#93c5fd;font-weight:700">{r["id"]}</td>
  <td><code style="color:#86efac;font-size:11px">{smi}</code></td>
  <td style="text-align:center">{badge(r["valid"])}</td>
  <td style="text-align:center;color:#fbbf24;font-weight:700">{r["stage"] or "-"}</td>
  <td>{stab_bar(r["atom_stab_pct"])}</td>
  <td style="text-align:center;color:#34d399;font-weight:600">{gibbs}</td>
  <td style="text-align:center;color:#c4b5fd">{r["mol_wt"]}</td>
  <td style="text-align:center">{r["num_heavy"]}</td>
  <td style="text-align:center">{elem_badges(r)}</td>
  <td style="text-align:center;color:{lip_color};font-weight:600">{lip_ok}</td>
  <td style="text-align:center;color:#94a3b8;font-size:11px">{r["elapsed_s"]}s</td>
</tr>'''

# Element distribution bar data
elem_total = sum(elem_cov.values())
elem_items = sorted(elem_cov.items(), key=lambda x: -x[1])

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>The Test Results — Blue (Updated) | Diffusion Model Generation</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #030e1f; color: #e2e8f0; font-family: 'Inter', sans-serif; min-height: 100vh; }}
  code {{ font-family: 'JetBrains Mono', monospace; }}
  .hero {{ background: linear-gradient(135deg, #0a1628 0%, #0d2147 40%, #0a1a40 100%);
           border-bottom: 1px solid #1e3a6e; padding: 48px 40px 36px; position: relative; overflow: hidden; }}
  .hero::before {{ content: ""; position: absolute; inset: 0;
    background: radial-gradient(ellipse 70% 70% at 60% -20%, rgba(59,130,246,0.18) 0%, transparent 60%);
    pointer-events: none; }}
  .hero-badge {{ display: inline-block; background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.4); color: #93c5fd; padding: 4px 16px;
    border-radius: 20px; font-size: 12px; font-weight: 600; letter-spacing: 1px;
    text-transform: uppercase; margin-bottom: 16px; }}
  .hero h1 {{ font-size: 2.6rem; font-weight: 900;
    background: linear-gradient(135deg, #ffffff 0%, #93c5fd 50%, #3b82f6 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.15; margin-bottom: 8px; }}
  .hero-sub {{ color: #64748b; font-size: 14px; margin-top: 8px; }}
  .updated-tag {{ display: inline-block; background: linear-gradient(135deg, #f59e0b, #fbbf24);
    color: #000; padding: 3px 12px; border-radius: 8px; font-size: 11px; font-weight: 800;
    margin-left: 12px; vertical-align: middle; text-transform: uppercase; letter-spacing: 1px; }}
  .sec {{ padding: 32px 40px; }}
  .metrics-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 24px; }}
  .metric-card {{ background: linear-gradient(135deg, #0d1f3c, #0f2548);
    border: 2px solid; border-radius: 16px; padding: 24px 28px; position: relative; overflow: hidden; }}
  .metric-card::before {{ content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: var(--acc); border-radius: 16px 16px 0 0; }}
  .mc-label {{ font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px; }}
  .mc-val {{ font-size: 3rem; font-weight: 900; color: #fff; line-height: 1; }}
  .mc-sub {{ font-size: 12px; color: #475569; margin-top: 6px; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin-bottom: 24px; }}
  .stat-card {{ background: linear-gradient(135deg, #0d1f3c, #0f2548);
    border: 1px solid #1e3a6e; border-radius: 12px; padding: 18px; position: relative; overflow: hidden; }}
  .stat-card::before {{ content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: var(--acc, linear-gradient(90deg,#3b82f6,#60a5fa)); border-radius: 12px 12px 0 0; }}
  .sl {{ font-size: 10px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }}
  .sv {{ font-size: 1.6rem; font-weight: 800; color: #fff; line-height: 1; }}
  .ss {{ font-size: 11px; color: #475569; margin-top: 4px; }}
  .note-box {{ background: rgba(59,130,246,0.08); border: 1px solid rgba(59,130,246,0.25);
    border-radius: 10px; padding: 14px 20px; color: #93c5fd; font-size: 12px;
    margin-bottom: 24px; line-height: 1.6; }}
  .table-section {{ padding: 0 40px 48px; }}
  .table-wrap {{ overflow-x: auto; border-radius: 14px; border: 1px solid #1e3a6e; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  thead tr {{ background: linear-gradient(90deg, #0d2147 0%, #0f2a5a 100%);
    border-bottom: 2px solid #1e3a6e; }}
  th {{ padding: 14px 12px; text-align: left; font-weight: 700; font-size: 11px;
    color: #93c5fd; letter-spacing: 0.5px; text-transform: uppercase; white-space: nowrap; }}
  td {{ padding: 10px 12px; border-bottom: 1px solid #0d1f3c; vertical-align: middle; white-space: nowrap; }}
  tr:hover td {{ background: rgba(59,130,246,0.06) !important; }}
  .footer {{ background: #030e1f; border-top: 1px solid #1e3a6e;
    padding: 24px 40px; text-align: center; color: #334155; font-size: 12px; }}
  h2 {{ font-size: 1.15rem; font-weight: 700; color: #fff; margin-bottom: 14px; }}
</style>
</head>
<body>

<!-- HERO -->
<div class="hero">
  <div class="hero-badge">&#x2697;&#xFE0F; Diffusion Dynamics — Model Generation</div>
  <h1>The Test Results <span class="updated-tag">Updated</span><br>
      <span style="font-size:1.6rem">Diffusion Model Generated Molecules on QM9</span></h1>
  <div class="hero-sub">
    {N} molecules generated via reverse diffusion &nbsp;|&nbsp; EGNN + PINO + Gibbs Head &nbsp;|&nbsp;
    80 denoising steps &nbsp;|&nbsp; QM9 reference graphs &nbsp;|&nbsp; Elements: C, H, N, O
  </div>
</div>

<!-- KEY METRICS -->
<div class="sec">
  <div class="metrics-grid">
    <div class="metric-card" style="border-color:#22c55e44;--acc:linear-gradient(90deg,#22c55e,#4ade80)">
      <div class="mc-label" style="color:#22c55e">Structural Validity</div>
      <div class="mc-val">{val_pct}<span style="font-size:1.5rem;color:#22c55e">%</span></div>
      <div class="mc-sub">{n_valid}/{N} generated molecules are valid</div>
    </div>
    <div class="metric-card" style="border-color:#f59e0b44;--acc:linear-gradient(90deg,#f59e0b,#fbbf24)">
      <div class="mc-label" style="color:#f59e0b">Avg Gibbs Energy</div>
      <div class="mc-val">{avg_gibbs:.4f}<span style="font-size:1.2rem;color:#f59e0b"> eV</span></div>
      <div class="mc-sub">Model's predicted thermodynamic energy</div>
    </div>
    <div class="metric-card" style="border-color:#a78bfa44;--acc:linear-gradient(90deg,#a78bfa,#c4b5fd)">
      <div class="mc-label" style="color:#a78bfa">Lipinski Compliance</div>
      <div class="mc-val">{lip_pct}<span style="font-size:1.5rem;color:#a78bfa">%</span></div>
      <div class="mc-sub">{int(n_valid * lip_pct / 100)}/{n_valid} valid molecules are drug-like</div>
    </div>
  </div>

  <!-- STATS CARDS -->
  <div class="stats-grid">
    <div class="stat-card" style="--acc:linear-gradient(90deg,#3b82f6,#60a5fa)">
      <div class="sl">Total Generated</div>
      <div class="sv">{N}</div>
      <div class="ss">Molecules from noise</div>
    </div>
    <div class="stat-card" style="--acc:linear-gradient(90deg,#22c55e,#4ade80)">
      <div class="sl">Atom Stability</div>
      <div class="sv">{avg_stab}%</div>
      <div class="ss">Avg across all valid mols</div>
    </div>
    <div class="stat-card" style="--acc:linear-gradient(90deg,#a78bfa,#c4b5fd)">
      <div class="sl">Unique SMILES</div>
      <div class="sv">{n_unique}/{len([r for r in rows if r['gen_smiles']])}</div>
      <div class="ss">{uniq_pct}% diversity</div>
    </div>
    <div class="stat-card" style="--acc:linear-gradient(90deg,#7c3aed,#a78bfa)">
      <div class="sl">Contains N</div>
      <div class="sv">{n_N}/{N}</div>
      <div class="ss">{round(n_N/N*100,1)}% of molecules</div>
    </div>
    <div class="stat-card" style="--acc:linear-gradient(90deg,#dc2626,#f87171)">
      <div class="sl">Contains O</div>
      <div class="sv">{n_O}/{N}</div>
      <div class="ss">{round(n_O/N*100,1)}% of molecules</div>
    </div>
    <div class="stat-card" style="--acc:linear-gradient(90deg,#0891b2,#22d3ee)">
      <div class="sl">Contains F</div>
      <div class="sv">{n_F}/{N}</div>
      <div class="ss">{round(n_F/N*100,1)}% of molecules</div>
    </div>
    <div class="stat-card" style="--acc:linear-gradient(90deg,#ec4899,#f472b6)">
      <div class="sl">Stage 1 / 2 / 3</div>
      <div class="sv" style="font-size:1.2rem">{stages.get('1',stages.get(1,0))} / {stages.get('2',stages.get(2,0))} / {stages.get('3',stages.get(3,0))}</div>
      <div class="ss">Bond inference pipeline</div>
    </div>
    <div class="stat-card" style="--acc:linear-gradient(90deg,#06b6d4,#22d3ee)">
      <div class="sl">Generation Time</div>
      <div class="sv">{total_t}s</div>
      <div class="ss">{round(total_t/N, 2)}s per molecule</div>
    </div>
  </div>

  <!-- NOTE -->
  <div class="note-box">
    <strong>&#x1F4A1; This report uses MODEL-GENERATED molecules</strong> &mdash;
    each molecule was created by running reverse diffusion (80 denoising steps) from random noise
    using the trained EGNN + PINO checkpoint (<code>pgmd_v3_full.pt</code>), guided by QM9 reference graphs.
    The 3-stage bond inference pipeline was then applied to convert generated 3D coordinates into SMILES.
    This is the standard evaluation methodology for molecular diffusion models (EDM, GeoDiff, etc.).
  </div>
</div>

<!-- TABLE -->
<div class="table-section">
  <h2 style="padding-left:0">&#x1F9EA; Per-Molecule Detail &mdash; All {N} Generated Molecules</h2>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Generated SMILES</th>
          <th>Valid</th>
          <th>Stage</th>
          <th>Atom Stab</th>
          <th>Gibbs (eV)</th>
          <th>Mol Wt</th>
          <th>Heavy</th>
          <th>Elements</th>
          <th>Lipinski</th>
          <th>Time</th>
        </tr>
      </thead>
      <tbody>
        {table_rows_html}
      </tbody>
    </table>
  </div>
</div>

<div class="footer">
  <div style="margin-bottom:8px;color:#1e3a6e;font-size:24px">&#x2022; &#x2022; &#x2022;</div>
  Generated by <strong style="color:#3b82f6">Diffusion Dynamics &mdash; EGNN + PINO Model</strong>
  &nbsp;|&nbsp; {N} molecules via reverse diffusion &nbsp;|&nbsp;
  "the test results_Blue (updated)" &nbsp;|&nbsp; 2026
</div>

</body>
</html>"""

with open(OUT_HTML, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"HTML saved: {OUT_HTML}")
print(f"Size: {os.path.getsize(OUT_HTML)//1024} KB")
