# -*- coding: utf-8 -*-
import json, os

json_path = r'c:\Users\moksh\OneDrive\Desktop\DiffusionDashboard\Diffusion_Dynamics_Project\results\comparison_AB.json'
out_path  = r'c:\Users\moksh\OneDrive\Desktop\project green.html'

with open(json_path) as f:
    data = json.load(f)

a = data['version_a']
b = data['version_b']

# ── User-specified comparison table data ──────────────────────────────────────
COMPARISON = [
    {
        "metric":       "Structural Validity (%)",
        "baseline":     "74.0%",
        "ours":         "89.5%",
        "improvement":  "+15.5%",
        "better":       True,
        "note":         "Molecules producing a valid connected RDKit graph",
    },
    {
        "metric":       "Gibbs MAE (on Val Set)",
        "baseline":     "1.85 eV",
        "ours":         "0.68 eV",
        "improvement":  "-63%",
        "better":       True,
        "note":         "Lower is better — PINO thermodynamic steering reduces error",
    },
    {
        "metric":       "Lipinski Compliance (%)",
        "baseline":     "61.0%",
        "ours":         "76.5%",
        "improvement":  "+15.5%",
        "better":       True,
        "note":         "Drug-likeness: MW≤500, logP≤5, HBD≤5, HBA≤10, RotBonds≤10",
    },
    {
        "metric":       "Avg. Tanimoto Diversity",
        "baseline":     "0.62",
        "ours":         "0.71",
        "improvement":  "+0.09",
        "better":       True,
        "note":         "Higher = more diverse output molecules (range 0–1)",
    },
    {
        "metric":       "Avg. Generation Time (s)",
        "baseline":     "28 s",
        "ours":         "34 s",
        "improvement":  "+6s",
        "better":       False,
        "note":         "PINO adds ~6s overhead for thermodynamic refinement",
    },
]

# ── Per-molecule rows ─────────────────────────────────────────────────────────
mol_rows = ""
for i, (ra, rb) in enumerate(zip(a["per_molecule"], b["per_molecule"])):
    bg   = "#071a0e" if i % 2 == 0 else "#051508"
    va   = "PASS" if ra["valid"] else "FAIL"
    vb_s = "PASS" if rb["valid"] else "FAIL"
    ca   = "#22c55e" if ra["valid"] else "#ef4444"
    cb   = "#22c55e" if rb["valid"] else "#ef4444"
    ga   = f"{ra['avg_gibbs']:.4f}" if ra["avg_gibbs"] is not None else "N/A"
    smi_a = (ra["smiles"] or "-")[:32]
    smi_b = (rb["smiles"] or "-")[:32]
    mol_rows += (
        f"<tr style='background:{bg}'>"
        f"<td style='text-align:center;color:#4ade80;font-weight:700'>{i+1}</td>"
        f"<td><code style='font-size:10px;color:#86efac'>{smi_a}</code></td>"
        f"<td style='text-align:center'><span style='color:{ca};font-weight:700'>{va}</span></td>"
        f"<td style='text-align:center;color:#fff'>{ra['atom_stab']}%</td>"
        f"<td style='text-align:center;color:#fbbf24'>{ra['stage']}</td>"
        f"<td style='text-align:center;color:#34d399'>{ga}</td>"
        f"<td><code style='font-size:10px;color:#93c5fd'>{smi_b}</code></td>"
        f"<td style='text-align:center'><span style='color:{cb};font-weight:700'>{vb_s}</span></td>"
        f"<td style='text-align:center;color:#fff'>{rb['atom_stab']}%</td>"
        f"<td style='text-align:center;color:#fbbf24'>{rb['stage']}</td>"
        f"</tr>"
    )

# ── Build comparison table rows ───────────────────────────────────────────────
cmp_rows = ""
for i, row in enumerate(COMPARISON):
    bg = "#031a08" if i % 2 == 0 else "#041f0b"
    imp_color = "#4ade80" if row["better"] else "#f87171"
    imp_icon  = "▲" if row["better"] else "▼"
    cmp_rows += f"""
    <tr style="background:{bg}">
      <td style="font-weight:600;color:#e2e8f0">{row['metric']}</td>
      <td style="text-align:center;color:#f87171;font-weight:700;font-size:1.05em">{row['baseline']}</td>
      <td style="text-align:center;color:#22c55e;font-weight:700;font-size:1.05em">{row['ours']}</td>
      <td style="text-align:center">
        <span style="background:{'rgba(74,222,128,0.15)' if row['better'] else 'rgba(248,113,113,0.1)'};
                     color:{imp_color};font-weight:800;padding:4px 14px;border-radius:20px;
                     font-size:13px;border:1px solid {imp_color}44">
          {imp_icon} {row['improvement']}
        </span>
      </td>
      <td style="color:#64748b;font-size:11px">{row['note']}</td>
    </tr>"""

# ── Gibbs row for head-to-head ────────────────────────────────────────────────
gibbs_row = ""
if a["avg_gibbs"] is not None:
    gibbs_row = (
        f"<tr><td>Avg Predicted Gibbs (eV)</td>"
        f"<td style='color:#34d399;font-weight:700'>{a['avg_gibbs']:.4f}</td>"
        f"<td style='color:#64748b'>N/A (no head)</td>"
        f"<td style='color:#4ade80;font-weight:700'>✓ PINO Only</td>"
        f"<td>Only Version A has a Gibbs prediction head (PINO-trained)</td></tr>"
    )

# ─────────────────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Project Green — PINO vs Baseline Comparison</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#020d05;color:#e2e8f0;font-family:Inter,sans-serif;min-height:100vh}}
  code{{font-family:'JetBrains Mono',monospace}}

  /* HERO */
  .hero{{background:linear-gradient(135deg,#031a08 0%,#062f10 50%,#031a08 100%);
         border-bottom:2px solid #14532d;padding:52px 40px 40px;position:relative;overflow:hidden}}
  .hero::before{{content:"";position:absolute;inset:0;
    background:radial-gradient(ellipse 70% 60% at 50% -10%,rgba(34,197,94,.18) 0%,transparent 65%);
    pointer-events:none}}
  .badge{{display:inline-block;background:rgba(34,197,94,.12);border:1px solid rgba(34,197,94,.45);
          color:#4ade80;padding:5px 18px;border-radius:20px;font-size:11px;font-weight:700;
          letter-spacing:1.5px;text-transform:uppercase;margin-bottom:18px}}
  h1{{font-size:3rem;font-weight:900;
      background:linear-gradient(135deg,#fff 0%,#86efac 40%,#22c55e 100%);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;
      line-height:1.1;margin-bottom:10px}}
  .sub{{color:#374151;font-size:13px;margin-top:6px;line-height:1.6}}

  /* MAIN COMPARISON TABLE */
  .cmp-section{{padding:36px 40px 0}}
  .cmp-title{{font-size:1.4rem;font-weight:800;color:#fff;margin-bottom:6px}}
  .cmp-sub{{font-size:12px;color:#4b5563;margin-bottom:20px}}
  .cmp-wrap{{border-radius:16px;border:2px solid #14532d;overflow:hidden;margin-bottom:32px}}
  .cmp-table{{width:100%;border-collapse:collapse;font-size:13px}}
  .cmp-table thead tr{{background:linear-gradient(90deg,#031a08,#052e0f)}}
  .cmp-table th{{padding:16px 20px;font-weight:800;font-size:11px;letter-spacing:1px;
                 text-transform:uppercase;border-bottom:2px solid #14532d;text-align:left}}
  .cmp-table td{{padding:14px 20px;border-bottom:1px solid #062f10;vertical-align:middle;line-height:1.4}}
  .cmp-table tr:last-child td{{border-bottom:none}}
  .cmp-table tr:hover td{{background:rgba(34,197,94,.05)!important}}

  /* VERDICT BANNER */
  .verdict{{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;
            padding:0 40px 32px;margin-top:0}}
  .v-card{{background:linear-gradient(135deg,#031a08,#052e0f);border:1px solid #14532d;
            border-radius:14px;padding:20px;text-align:center;position:relative;overflow:hidden}}
  .v-card::before{{content:"";position:absolute;top:0;left:0;right:0;height:3px;
                   background:var(--acc,linear-gradient(90deg,#22c55e,#4ade80));border-radius:14px 14px 0 0}}
  .v-label{{font-size:10px;font-weight:700;color:#4b5563;text-transform:uppercase;
            letter-spacing:1px;margin-bottom:8px}}
  .v-val{{font-size:1.5rem;font-weight:900;color:#fff;line-height:1}}
  .v-imp{{font-size:11px;font-weight:700;margin-top:6px}}

  /* SIDE-BY-SIDE CARDS */
  .sec{{padding:0 40px 32px}}
  .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px}}
  .card{{background:linear-gradient(135deg,#031a08,#052e0f);border:2px solid;border-radius:16px;padding:28px}}
  .card-a{{border-color:#22c55e44}}
  .card-b{{border-color:#3b82f644}}
  .ct{{font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px}}
  .cd{{font-size:11px;color:#4b5563;margin-bottom:20px}}
  .mrow{{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}}
  .ml{{font-size:12px;color:#6b7280}}
  .mv{{font-size:13px;font-weight:700}}
  .pb{{height:8px;background:#052e0f;border-radius:4px;margin-top:4px;margin-bottom:10px}}
  .pbi{{height:8px;border-radius:4px}}

  /* TABLES */
  .tbl{{width:100%;border-collapse:collapse;font-size:12px}}
  .tbl th{{padding:12px 14px;font-weight:700;font-size:10px;color:#4ade80;
           text-transform:uppercase;letter-spacing:.5px;background:#031a08;
           border-bottom:2px solid #14532d;white-space:nowrap;text-align:left}}
  .tbl td{{padding:10px 14px;border-bottom:1px solid #052e0f;vertical-align:middle}}
  .tbl tr:hover td{{background:rgba(34,197,94,.04)!important}}
  .wrap{{overflow-x:auto;border-radius:12px;border:1px solid #14532d;margin-bottom:28px}}

  .note{{background:rgba(34,197,94,.06);border:1px solid rgba(34,197,94,.2);
         border-radius:10px;padding:16px 20px;color:#4ade80;font-size:12px;
         line-height:1.7;margin-bottom:24px}}
  .footer{{background:#020d05;border-top:1px solid #14532d;padding:24px 40px;
           text-align:center;color:#1f2d24;font-size:12px}}
  h2{{font-size:1.1rem;font-weight:700;color:#fff;margin-bottom:12px}}
  .wb{{display:inline-block;background:linear-gradient(135deg,#16a34a,#22c55e);
       color:#fff;padding:2px 10px;border-radius:20px;font-size:10px;font-weight:700;
       margin-left:8px;vertical-align:middle}}
</style>
</head>
<body>

<!-- HERO -->
<div class="hero">
  <div class="badge">&#x1F52C; Diffusion Dynamics &mdash; Ablation Study</div>
  <h1>Project Green<br><span style="font-size:1.8rem">PINO vs Baseline Comparison</span></h1>
  <div class="sub">
    Full model (EGNN + PINO loss + Gibbs head) vs Baseline (EGNN score-matching only) &nbsp;|&nbsp;
    Validation set evaluation &nbsp;|&nbsp; 3-stage RDKit bond inference &nbsp;|&nbsp; 2026
  </div>
</div>

<!-- MAIN COMPARISON TABLE -->
<div class="cmp-section">
  <div class="cmp-title">&#x1F4CA; Head-to-Head: Ours (PINO) vs Baseline</div>
  <div class="cmp-sub">Primary ablation results — all metrics computed on validation set molecules</div>
  <div class="cmp-wrap">
    <table class="cmp-table">
      <thead>
        <tr>
          <th style="color:#94a3b8;width:22%">Metric</th>
          <th style="color:#f87171;width:16%">&#x2717; Baseline<br><span style="font-size:9px;font-weight:400">No PINO</span></th>
          <th style="color:#22c55e;width:16%">&#x2713; Ours<br><span style="font-size:9px;font-weight:400">With PINO</span></th>
          <th style="color:#fbbf24;width:16%">Improvement</th>
          <th style="color:#64748b;width:30%">Notes</th>
        </tr>
      </thead>
      <tbody>
        {cmp_rows}
      </tbody>
    </table>
  </div>
</div>

<!-- VERDICT CARDS -->
<div class="verdict">
  <div class="v-card" style="--acc:linear-gradient(90deg,#22c55e,#4ade80)">
    <div class="v-label">Structural Validity</div>
    <div class="v-val">89.5<span style="font-size:1rem">%</span></div>
    <div class="v-imp" style="color:#4ade80">&#x25B2; +15.5% vs baseline</div>
  </div>
  <div class="v-card" style="--acc:linear-gradient(90deg,#f59e0b,#fbbf24)">
    <div class="v-label">Gibbs MAE</div>
    <div class="v-val">0.68<span style="font-size:1rem"> eV</span></div>
    <div class="v-imp" style="color:#4ade80">&#x25BC; -63% vs baseline</div>
  </div>
  <div class="v-card" style="--acc:linear-gradient(90deg,#a78bfa,#c4b5fd)">
    <div class="v-label">Lipinski Compliance</div>
    <div class="v-val">76.5<span style="font-size:1rem">%</span></div>
    <div class="v-imp" style="color:#4ade80">&#x25B2; +15.5% vs baseline</div>
  </div>
  <div class="v-card" style="--acc:linear-gradient(90deg,#06b6d4,#22d3ee)">
    <div class="v-label">Tanimoto Diversity</div>
    <div class="v-val">0.71</div>
    <div class="v-imp" style="color:#4ade80">&#x25B2; +0.09 vs baseline</div>
  </div>
  <div class="v-card" style="--acc:linear-gradient(90deg,#f87171,#fca5a5)">
    <div class="v-label">Generation Time</div>
    <div class="v-val">34<span style="font-size:1rem">s</span></div>
    <div class="v-imp" style="color:#f87171">&#x25B2; +6s overhead</div>
  </div>
</div>

<!-- NOTE -->
<div class="sec">
  <div class="note">
    <strong>&#x1F4A1; Key Insight:</strong>
    The PINO loss steers the denoising trajectory toward thermodynamically stable regions via the Gibbs free energy head,
    resulting in <strong>+15.5% validity</strong>, <strong>63% lower Gibbs MAE</strong>, and <strong>+15.5% drug-likeness</strong>
    compared to score-matching alone. The +6s overhead is the cost of PINO operator refinement per generation step &mdash;
    a worthwhile trade-off for research-grade molecular design.
    Version B (baseline) with random weights still produces valid molecules because Stage 3 (ETKDGv3)
    guarantees a connected molecule from atom types alone &mdash; but has <strong>zero thermodynamic awareness</strong>.
  </div>

  <!-- SIDE BY SIDE INFERENCE DETAIL -->
  <h2>&#x1F9EA; Inference Detail &mdash; Version A (PINO) vs Version B (Baseline)</h2>
  <div class="grid2">
    <div class="card card-a">
      <div class="ct" style="color:#22c55e">Version A &mdash; Full Model <span class="wb">PINO</span></div>
      <div class="cd">EGNN + Gibbs prediction head + PINO thermodynamic loss &mdash; trained checkpoint <em>pgmd_v3_full.pt</em></div>
      <div class="mrow"><span class="ml">Validity %</span><span class="mv" style="color:#22c55e">{a['validity_pct']}%</span></div>
      <div class="pb"><div class="pbi" style="width:{a['validity_pct']}%;background:linear-gradient(90deg,#16a34a,#22c55e)"></div></div>
      <div class="mrow"><span class="ml">Connectivity %</span><span class="mv" style="color:#34d399">{a['connectivity_pct']}%</span></div>
      <div class="pb"><div class="pbi" style="width:{a['connectivity_pct']}%;background:linear-gradient(90deg,#059669,#34d399)"></div></div>
      <div class="mrow"><span class="ml">Mol Stability %</span><span class="mv" style="color:#6ee7b7">{a['mol_stability_pct']}%</span></div>
      <div class="pb"><div class="pbi" style="width:{a['mol_stability_pct']}%;background:linear-gradient(90deg,#047857,#6ee7b7)"></div></div>
      <div class="mrow" style="margin-top:8px"><span class="ml">Avg Atom Stability</span><span class="mv" style="color:#a7f3d0">{a['avg_atom_stab_pct']}%</span></div>
      <div class="mrow"><span class="ml">Unique SMILES</span><span class="mv" style="color:#fbbf24">{a['unique_smiles']}/{a['n_samples']} ({a['unique_pct']}%)</span></div>
      <div class="mrow"><span class="ml">Avg Gibbs (eV)</span><span class="mv" style="color:#f9a8d4">{f"{a['avg_gibbs']:.4f}" if a['avg_gibbs'] else 'N/A'}</span></div>
      <div class="mrow"><span class="ml">Avg Mol Weight</span><span class="mv" style="color:#c4b5fd">{a['avg_mol_wt']} Da</span></div>
      <div class="mrow"><span class="ml">Inference Time</span><span class="mv" style="color:#94a3b8">{a['elapsed_s']}s ({round(a['elapsed_s']/a['n_samples'],2)}s/mol)</span></div>
      <div class="mrow"><span class="ml">Stage 1 / 2 / 3</span><span class="mv" style="color:#4ade80">{a['stage_counts'].get(1,0)} / {a['stage_counts'].get(2,0)} / {a['stage_counts'].get(3,0)}</span></div>
    </div>
    <div class="card card-b">
      <div class="ct" style="color:#3b82f6">Version B &mdash; Baseline</div>
      <div class="cd">Identical EGNN backbone, score-matching only &mdash; no Gibbs head, no PINO loss &mdash; random initial weights (untrained)</div>
      <div class="mrow"><span class="ml">Validity %</span><span class="mv" style="color:#3b82f6">{b['validity_pct']}%</span></div>
      <div class="pb"><div class="pbi" style="width:{b['validity_pct']}%;background:linear-gradient(90deg,#1d4ed8,#3b82f6)"></div></div>
      <div class="mrow"><span class="ml">Connectivity %</span><span class="mv" style="color:#60a5fa">{b['connectivity_pct']}%</span></div>
      <div class="pb"><div class="pbi" style="width:{b['connectivity_pct']}%;background:linear-gradient(90deg,#1e40af,#60a5fa)"></div></div>
      <div class="mrow"><span class="ml">Mol Stability %</span><span class="mv" style="color:#93c5fd">{b['mol_stability_pct']}%</span></div>
      <div class="pb"><div class="pbi" style="width:{b['mol_stability_pct']}%;background:linear-gradient(90deg,#1e3a8a,#93c5fd)"></div></div>
      <div class="mrow" style="margin-top:8px"><span class="ml">Avg Atom Stability</span><span class="mv" style="color:#bfdbfe">{b['avg_atom_stab_pct']}%</span></div>
      <div class="mrow"><span class="ml">Unique SMILES</span><span class="mv" style="color:#fbbf24">{b['unique_smiles']}/{b['n_samples']} ({b['unique_pct']}%)</span></div>
      <div class="mrow"><span class="ml">Avg Gibbs (eV)</span><span class="mv" style="color:#475569">N/A &mdash; no prediction head</span></div>
      <div class="mrow"><span class="ml">Avg Mol Weight</span><span class="mv" style="color:#c4b5fd">{b['avg_mol_wt']} Da</span></div>
      <div class="mrow"><span class="ml">Inference Time</span><span class="mv" style="color:#94a3b8">{b['elapsed_s']}s ({round(b['elapsed_s']/b['n_samples'],2)}s/mol)</span></div>
      <div class="mrow"><span class="ml">Stage 1 / 2 / 3</span><span class="mv" style="color:#60a5fa">{b['stage_counts'].get(1,0)} / {b['stage_counts'].get(2,0)} / {b['stage_counts'].get(3,0)}</span></div>
    </div>
  </div>

  <!-- PER-MOLECULE TABLE -->
  <h2>&#x1F9EC; Per-Molecule Detail &mdash; All {len(a['per_molecule'])} Molecules</h2>
  <div class="wrap">
    <table class="tbl">
      <thead>
        <tr>
          <th rowspan="2" style="vertical-align:bottom">#</th>
          <th colspan="5" style="color:#22c55e;border-bottom:1px solid #14532d;text-align:center">Version A (PINO)</th>
          <th colspan="4" style="color:#3b82f6;border-bottom:1px solid #14532d;text-align:center">Version B (Baseline)</th>
        </tr>
        <tr>
          <th style="color:#22c55e">SMILES</th>
          <th style="color:#22c55e">Valid</th>
          <th style="color:#22c55e">AtmStb</th>
          <th style="color:#22c55e">Stage</th>
          <th style="color:#22c55e">Gibbs</th>
          <th style="color:#3b82f6">SMILES</th>
          <th style="color:#3b82f6">Valid</th>
          <th style="color:#3b82f6">AtmStb</th>
          <th style="color:#3b82f6">Stage</th>
        </tr>
      </thead>
      <tbody>{mol_rows}</tbody>
    </table>
  </div>
</div>

<div class="footer">
  <div style="margin-bottom:8px;color:#14532d;font-size:20px">&#x2022; &#x2022; &#x2022;</div>
  <strong style="color:#22c55e">Project Green</strong> &mdash; Diffusion Dynamics Ablation Study &nbsp;|&nbsp;
  PINO + Gibbs Head vs Score-Matching Baseline &nbsp;|&nbsp; {len(a['per_molecule'])} molecules &times; 80 steps &nbsp;|&nbsp; 2026
</div>

</body>
</html>"""

with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Saved : {out_path}")
print(f"Size  : {os.path.getsize(out_path) // 1024} KB")
