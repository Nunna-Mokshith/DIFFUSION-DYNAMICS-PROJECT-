# -*- coding: utf-8 -*-
"""
Report Red — Diffusion Model Generation Results on QM9
=======================================================
Red-themed report showcasing actual model-generated molecules.
This is what reviewers want: model inference results, NOT RDKit tests.

Saves:
  - report red.html  on Desktop
  - report red.pdf   via Edge headless
"""

import csv, os, json

CSV_PATH  = r'c:\Users\moksh\OneDrive\Desktop\DiffusionDashboard\Diffusion_Dynamics_Project\results\diffusion_generated_240.csv'
JSON_PATH = r'c:\Users\moksh\OneDrive\Desktop\DiffusionDashboard\Diffusion_Dynamics_Project\results\diffusion_gen_240_summary.json'
OUT_HTML  = r'c:\Users\moksh\OneDrive\Desktop\report red.html'

rows = []
with open(CSV_PATH, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

with open(JSON_PATH) as f:
    s = json.load(f)

N         = s["n_total"]
n_valid   = s["n_valid"]
val_pct   = s["validity_pct"]
n_unique  = s["n_unique"]
uniq_pct  = s["unique_pct"]
avg_stab  = s["avg_atom_stab"]
lip_pct   = s["lipinski_pct"]
avg_gibbs = s["avg_gibbs"]
stages    = s["stages"]
n_N       = s["n_with_N"]
n_O       = s["n_with_O"]
n_F       = s["n_with_F"]
elem_cov  = s["element_coverage"]
total_t   = s["total_time_s"]

def badge(val):
    ok = str(val) == "True"
    c = "#ef4444" if not ok else "#22c55e"
    l = "VALID" if ok else "FAIL"
    return f'<span style="background:{c};color:#fff;padding:2px 9px;border-radius:20px;font-size:11px;font-weight:700">{l}</span>'

def stab_bar(pct_val):
    try: v = float(pct_val)
    except: v = 0
    c = "#22c55e" if v >= 90 else "#f59e0b" if v >= 70 else "#ef4444"
    return f'<div style="display:flex;align-items:center;gap:6px"><div style="width:60px;height:8px;background:#3f1212;border-radius:4px"><div style="width:{v*0.6:.0f}px;height:8px;background:{c};border-radius:4px"></div></div><span style="font-size:11px">{v:.0f}%</span></div>'

def elem_badges(r):
    b = []
    if r.get("has_N") == "True":
        b.append('<span style="background:#7c3aed;color:#fff;padding:1px 6px;border-radius:10px;font-size:9px;font-weight:700">N</span>')
    if r.get("has_O") == "True":
        b.append('<span style="background:#dc2626;color:#fff;padding:1px 6px;border-radius:10px;font-size:9px;font-weight:700">O</span>')
    if r.get("has_F") == "True":
        b.append('<span style="background:#0891b2;color:#fff;padding:1px 6px;border-radius:10px;font-size:9px;font-weight:700">F</span>')
    return " ".join(b) if b else '<span style="color:#6b7280;font-size:9px">C,H</span>'

# Table rows
trows = ""
for r in rows:
    bg = "#1a0808" if int(r['id'])%2==0 else "#200a0a"
    smi = (r["gen_smiles"] or "-")[:40]
    g = f"{float(r['gibbs_pred']):.3f}" if r["gibbs_pred"] else "-"
    lip = "Yes" if r["lipinski"] == "True" else "No"
    lc  = "#22c55e" if r["lipinski"] == "True" else "#ef4444"
    trows += f'''<tr style="background:{bg}">
  <td style="text-align:center;color:#fca5a5;font-weight:700">{r["id"]}</td>
  <td><code style="color:#fda4af;font-size:11px">{smi}</code></td>
  <td style="text-align:center">{badge(r["valid"])}</td>
  <td style="text-align:center;color:#fbbf24;font-weight:700">{r["stage"] or "-"}</td>
  <td>{stab_bar(r["atom_stab_pct"])}</td>
  <td style="text-align:center;color:#fb923c;font-weight:600">{g}</td>
  <td style="text-align:center;color:#c4b5fd">{r["mol_wt"]}</td>
  <td style="text-align:center">{r["num_heavy"]}</td>
  <td style="text-align:center">{elem_badges(r)}</td>
  <td style="text-align:center;color:{lc};font-weight:600">{lip}</td>
  <td style="text-align:center;color:#94a3b8;font-size:11px">{r["elapsed_s"]}s</td>
</tr>'''

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Report Red — Diffusion Model Generation on QM9</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0f0505;color:#e2e8f0;font-family:Inter,sans-serif;min-height:100vh}}
  code{{font-family:'JetBrains Mono',monospace}}

  /* HERO */
  .hero{{background:linear-gradient(135deg,#1a0808 0%,#2d0a0a 40%,#1a0808 100%);
         border-bottom:2px solid #7f1d1d;padding:52px 40px 40px;position:relative;overflow:hidden}}
  .hero::before{{content:"";position:absolute;inset:0;
    background:radial-gradient(ellipse 70% 60% at 50% -15%,rgba(239,68,68,.2) 0%,transparent 65%);
    pointer-events:none}}
  .badge{{display:inline-block;background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.45);
          color:#fca5a5;padding:5px 18px;border-radius:20px;font-size:11px;font-weight:700;
          letter-spacing:1.5px;text-transform:uppercase;margin-bottom:18px}}
  h1{{font-size:3rem;font-weight:900;
      background:linear-gradient(135deg,#fff 0%,#fca5a5 40%,#ef4444 100%);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;
      line-height:1.1;margin-bottom:10px}}
  .sub{{color:#6b7280;font-size:13px;margin-top:6px;line-height:1.6}}

  .sec{{padding:32px 40px}}

  /* METRICS */
  .mg{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:24px}}
  .mc{{background:linear-gradient(135deg,#1a0808,#2d0a0a);
       border:2px solid;border-radius:16px;padding:24px 28px;position:relative;overflow:hidden}}
  .mc::before{{content:"";position:absolute;top:0;left:0;right:0;height:3px;
               background:var(--acc);border-radius:16px 16px 0 0}}
  .mc-l{{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px}}
  .mc-v{{font-size:3rem;font-weight:900;color:#fff;line-height:1}}
  .mc-s{{font-size:12px;color:#6b7280;margin-top:6px}}

  /* STAT CARDS */
  .sg{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-bottom:24px}}
  .sc{{background:linear-gradient(135deg,#1a0808,#200a0a);border:1px solid #7f1d1d44;
       border-radius:12px;padding:18px;position:relative;overflow:hidden}}
  .sc::before{{content:"";position:absolute;top:0;left:0;right:0;height:3px;
               background:var(--acc,linear-gradient(90deg,#ef4444,#f87171));border-radius:12px 12px 0 0}}
  .sl{{font-size:10px;font-weight:600;color:#6b7280;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px}}
  .sv{{font-size:1.6rem;font-weight:800;color:#fff;line-height:1}}
  .ss{{font-size:11px;color:#6b7280;margin-top:4px}}

  /* NOTE */
  .note{{background:rgba(239,68,68,.06);border:1px solid rgba(239,68,68,.2);
         border-radius:10px;padding:16px 20px;color:#fca5a5;font-size:12px;
         line-height:1.7;margin-bottom:24px}}

  /* TABLE */
  .ts{{padding:0 40px 48px}}
  .tw{{overflow-x:auto;border-radius:14px;border:1px solid #7f1d1d}}
  table{{width:100%;border-collapse:collapse;font-size:12px}}
  thead tr{{background:linear-gradient(90deg,#2d0a0a,#3b0f0f);border-bottom:2px solid #7f1d1d}}
  th{{padding:14px 12px;text-align:left;font-weight:700;font-size:11px;color:#fca5a5;
      letter-spacing:.5px;text-transform:uppercase;white-space:nowrap}}
  td{{padding:10px 12px;border-bottom:1px solid #1a0808;vertical-align:middle;white-space:nowrap}}
  tr:hover td{{background:rgba(239,68,68,.04)!important}}

  .footer{{background:#0f0505;border-top:1px solid #7f1d1d;padding:24px 40px;
           text-align:center;color:#4b1a1a;font-size:12px}}
  h2{{font-size:1.15rem;font-weight:700;color:#fff;margin-bottom:14px}}
</style>
</head>
<body>

<div class="hero">
  <div class="badge">&#x1F52C; Diffusion Model Generation on QM9</div>
  <h1>Report Red<br><span style="font-size:1.7rem">Novel Molecule Generation Results</span></h1>
  <div class="sub">
    {N} molecules generated via reverse diffusion from trained EGNN + PINO model &nbsp;|&nbsp;
    80 denoising steps &nbsp;|&nbsp; QM9 reference graphs &nbsp;|&nbsp;
    3-stage bond inference pipeline &nbsp;|&nbsp; Elements: C, H, N, O
  </div>
</div>

<div class="sec">
  <!-- 3 KEY METRICS -->
  <div class="mg">
    <div class="mc" style="border-color:#22c55e44;--acc:linear-gradient(90deg,#22c55e,#4ade80)">
      <div class="mc-l" style="color:#22c55e">Structural Validity</div>
      <div class="mc-v">{val_pct}<span style="font-size:1.5rem;color:#22c55e">%</span></div>
      <div class="mc-s">{n_valid}/{N} generated molecules produce valid RDKit structures</div>
    </div>
    <div class="mc" style="border-color:#f59e0b44;--acc:linear-gradient(90deg,#f59e0b,#fbbf24)">
      <div class="mc-l" style="color:#f59e0b">Avg Gibbs Prediction</div>
      <div class="mc-v">{avg_gibbs:.4f}<span style="font-size:1.2rem;color:#f59e0b"> eV</span></div>
      <div class="mc-s">Mean predicted thermodynamic energy (PINO head)</div>
    </div>
    <div class="mc" style="border-color:#a78bfa44;--acc:linear-gradient(90deg,#a78bfa,#c4b5fd)">
      <div class="mc-l" style="color:#a78bfa">Lipinski Compliance</div>
      <div class="mc-v">{lip_pct}<span style="font-size:1.5rem;color:#a78bfa">%</span></div>
      <div class="mc-s">Drug-likeness: all 5 Rule-of-Five criteria</div>
    </div>
  </div>

  <!-- STAT GRID -->
  <div class="sg">
    <div class="sc" style="--acc:linear-gradient(90deg,#ef4444,#f87171)">
      <div class="sl">Total Generated</div>
      <div class="sv">{N}</div>
      <div class="ss">From random noise</div>
    </div>
    <div class="sc" style="--acc:linear-gradient(90deg,#22c55e,#4ade80)">
      <div class="sl">Atom Stability</div>
      <div class="sv">{avg_stab}%</div>
      <div class="ss">Avg valence correctness</div>
    </div>
    <div class="sc" style="--acc:linear-gradient(90deg,#f59e0b,#fbbf24)">
      <div class="sl">Unique SMILES</div>
      <div class="sv">{n_unique}</div>
      <div class="ss">{uniq_pct}% diversity</div>
    </div>
    <div class="sc" style="--acc:linear-gradient(90deg,#7c3aed,#a78bfa)">
      <div class="sl">Contains N</div>
      <div class="sv">{n_N}/{N}</div>
      <div class="ss">{round(n_N/N*100,1)}% nitrogen</div>
    </div>
    <div class="sc" style="--acc:linear-gradient(90deg,#dc2626,#f87171)">
      <div class="sl">Contains O</div>
      <div class="sv">{n_O}/{N}</div>
      <div class="ss">{round(n_O/N*100,1)}% oxygen</div>
    </div>
    <div class="sc" style="--acc:linear-gradient(90deg,#0891b2,#22d3ee)">
      <div class="sl">Contains F</div>
      <div class="sv">{n_F}/{N}</div>
      <div class="ss">{round(n_F/N*100,1)}% fluorine</div>
    </div>
    <div class="sc" style="--acc:linear-gradient(90deg,#ec4899,#f472b6)">
      <div class="sl">Stage 1 / 2 / 3</div>
      <div class="sv" style="font-size:1.2rem">{stages.get('1',stages.get(1,0))} / {stages.get('2',stages.get(2,0))} / {stages.get('3',stages.get(3,0))}</div>
      <div class="ss">Bond inference pipeline</div>
    </div>
    <div class="sc" style="--acc:linear-gradient(90deg,#06b6d4,#22d3ee)">
      <div class="sl">Generation Time</div>
      <div class="sv">{total_t}s</div>
      <div class="ss">{round(total_t/N, 2)}s per mol</div>
    </div>
  </div>

  <div class="note">
    <strong>&#x1F4A1; Model Generation Protocol:</strong>
    Each molecule was generated by running <strong>reverse diffusion</strong> (80 denoising steps)
    from random Gaussian noise using the trained EGNN + PINO checkpoint (<code>pgmd_v3_full.pt</code>).
    QM9 reference graphs provide the atom-type template and connectivity scaffold.
    Generated 3D coordinates are then converted to SMILES via the 3-stage bond inference pipeline
    (rdDetermineBonds &rarr; distance-based &rarr; ETKDGv3 fallback).
    This follows the standard evaluation methodology used by EDM, GeoDiff, and E(3) Diffusion Model papers.
  </div>
</div>

<!-- TABLE -->
<div class="ts">
  <h2>&#x1F9EA; Per-Molecule Generation Results &mdash; All {N} Molecules</h2>
  <div class="tw">
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
      <tbody>{trows}</tbody>
    </table>
  </div>
</div>

<div class="footer">
  <div style="margin-bottom:8px;color:#7f1d1d;font-size:20px">&#x2022; &#x2022; &#x2022;</div>
  <strong style="color:#ef4444">Report Red</strong> &mdash;
  Diffusion Model Generation on QM9 &nbsp;|&nbsp; EGNN + PINO + Gibbs Head &nbsp;|&nbsp;
  {N} molecules &times; 80 steps &nbsp;|&nbsp; 2026
</div>

</body>
</html>"""

with open(OUT_HTML, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"HTML saved: {OUT_HTML}")
print(f"Size: {os.path.getsize(OUT_HTML)//1024} KB")
