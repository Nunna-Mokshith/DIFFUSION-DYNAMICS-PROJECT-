"""
Material Properties Knowledge Base
====================================
Contains real-world material property data for 30+ common materials.
The agentic engine uses this database to interpret natural-language
analogies like "as strong as steel" and map them to generation parameters.
"""

import re
import math

# ═══════════════════════════════════════════════════════════════════════════════
# 1. MATERIAL DATABASE — Real measured properties
# ═══════════════════════════════════════════════════════════════════════════════

MATERIALS_DB = {
    # ── Metals ────────────────────────────────────────────────────────────────
    "steel": {
        "category": "metal",
        "tensile_strength": 400,      # MPa (mild steel)
        "elongation_pct": 25,         # %
        "malleability": 4,            # 0-10
        "lustre": 7,                  # 0-10
        "hardness": 6.5,              # Mohs approx
        "corrosion_resistance": 3,    # 0-10
        "breathability": 0,           # 0-10
        "density": 7.85,              # g/cm³
        "melting_point": 1643,        # K
        "thermal_conductivity": 50,   # W/mK
        "electrical_conductivity": 6.99e6, # S/m
        "cost_index": 3,              # 0-10 (10=most expensive)
    },
    "stainless_steel": {
        "category": "metal",
        "tensile_strength": 520,
        "elongation_pct": 40,
        "malleability": 5,
        "lustre": 8,
        "hardness": 5.5,
        "corrosion_resistance": 8,
        "breathability": 0,
        "density": 8.0,
        "melting_point": 1673,
        "thermal_conductivity": 16,
        "electrical_conductivity": 1.39e6,
        "cost_index": 5,
    },
    "copper": {
        "category": "metal",
        "tensile_strength": 210,
        "elongation_pct": 50,
        "malleability": 9,
        "lustre": 8,
        "hardness": 3.0,
        "corrosion_resistance": 5,
        "breathability": 0,
        "density": 8.96,
        "melting_point": 1358,
        "thermal_conductivity": 401,
        "electrical_conductivity": 5.96e7,
        "cost_index": 6,
    },
    "gold": {
        "category": "metal",
        "tensile_strength": 120,
        "elongation_pct": 45,
        "malleability": 10,
        "lustre": 10,
        "hardness": 2.5,
        "corrosion_resistance": 10,
        "breathability": 0,
        "density": 19.32,
        "melting_point": 1337,
        "thermal_conductivity": 318,
        "electrical_conductivity": 4.52e7,
        "cost_index": 10,
    },
    "aluminum": {
        "category": "metal",
        "tensile_strength": 90,
        "elongation_pct": 40,
        "malleability": 8,
        "lustre": 7,
        "hardness": 2.75,
        "corrosion_resistance": 7,
        "breathability": 0,
        "density": 2.70,
        "melting_point": 933,
        "thermal_conductivity": 237,
        "electrical_conductivity": 3.77e7,
        "cost_index": 4,
    },
    "titanium": {
        "category": "metal",
        "tensile_strength": 434,
        "elongation_pct": 25,
        "malleability": 3,
        "lustre": 7,
        "hardness": 6.0,
        "corrosion_resistance": 9,
        "breathability": 0,
        "density": 4.51,
        "melting_point": 1941,
        "thermal_conductivity": 21.9,
        "electrical_conductivity": 2.38e6,
        "cost_index": 8,
    },
    "iron": {
        "category": "metal",
        "tensile_strength": 350,
        "elongation_pct": 30,
        "malleability": 5,
        "lustre": 6,
        "hardness": 4.0,
        "corrosion_resistance": 2,
        "breathability": 0,
        "density": 7.87,
        "melting_point": 1811,
        "thermal_conductivity": 80,
        "electrical_conductivity": 1.0e7,
        "cost_index": 2,
    },
    "silver": {
        "category": "metal",
        "tensile_strength": 170,
        "elongation_pct": 48,
        "malleability": 9,
        "lustre": 10,
        "hardness": 2.5,
        "corrosion_resistance": 6,
        "breathability": 0,
        "density": 10.49,
        "melting_point": 1235,
        "thermal_conductivity": 429,
        "electrical_conductivity": 6.30e7,
        "cost_index": 9,
    },
    "platinum": {
        "category": "metal",
        "tensile_strength": 165,
        "elongation_pct": 35,
        "malleability": 8,
        "lustre": 9,
        "hardness": 3.5,
        "corrosion_resistance": 10,
        "breathability": 0,
        "density": 21.45,
        "melting_point": 2041,
        "thermal_conductivity": 71.6,
        "electrical_conductivity": 9.43e6,
        "cost_index": 10,
    },
    "tungsten": {
        "category": "metal",
        "tensile_strength": 1510,
        "elongation_pct": 2,
        "malleability": 1,
        "lustre": 6,
        "hardness": 7.5,
        "corrosion_resistance": 7,
        "breathability": 0,
        "density": 19.25,
        "melting_point": 3695,
        "thermal_conductivity": 173,
        "electrical_conductivity": 1.89e7,
        "cost_index": 7,
    },
    "nickel": {
        "category": "metal",
        "tensile_strength": 380,
        "elongation_pct": 30,
        "malleability": 5,
        "lustre": 7,
        "hardness": 4.0,
        "corrosion_resistance": 7,
        "breathability": 0,
        "density": 8.91,
        "melting_point": 1728,
        "thermal_conductivity": 90.7,
        "electrical_conductivity": 1.43e7,
        "cost_index": 5,
    },
    "zinc": {
        "category": "metal",
        "tensile_strength": 37,
        "elongation_pct": 1,
        "malleability": 3,
        "lustre": 5,
        "hardness": 2.5,
        "corrosion_resistance": 6,
        "breathability": 0,
        "density": 7.13,
        "melting_point": 693,
        "thermal_conductivity": 116,
        "electrical_conductivity": 1.69e7,
        "cost_index": 3,
    },

    # ── Ceramics ──────────────────────────────────────────────────────────────
    "ceramic": {
        "category": "ceramic",
        "tensile_strength": 250,
        "elongation_pct": 0,
        "malleability": 0,
        "lustre": 4,
        "hardness": 9.0,
        "corrosion_resistance": 9,
        "breathability": 1,
        "density": 3.9,
        "melting_point": 2345,
        "thermal_conductivity": 30,
        "electrical_conductivity": 1e-10,
        "cost_index": 4,
    },
    "glass": {
        "category": "ceramic",
        "tensile_strength": 33,
        "elongation_pct": 0,
        "malleability": 0,
        "lustre": 9,
        "hardness": 5.5,
        "corrosion_resistance": 8,
        "breathability": 0,
        "density": 2.5,
        "melting_point": 1773,
        "thermal_conductivity": 1.0,
        "electrical_conductivity": 1e-12,
        "cost_index": 2,
    },
    "diamond": {
        "category": "ceramic",
        "tensile_strength": 2800,
        "elongation_pct": 0,
        "malleability": 0,
        "lustre": 10,
        "hardness": 10.0,
        "corrosion_resistance": 10,
        "breathability": 0,
        "density": 3.51,
        "melting_point": 3823,
        "thermal_conductivity": 2200,
        "electrical_conductivity": 1e-13,
        "cost_index": 10,
    },
    "silicon_carbide": {
        "category": "ceramic",
        "tensile_strength": 450,
        "elongation_pct": 0,
        "malleability": 0,
        "lustre": 5,
        "hardness": 9.5,
        "corrosion_resistance": 9,
        "breathability": 0,
        "density": 3.21,
        "melting_point": 3003,
        "thermal_conductivity": 120,
        "electrical_conductivity": 1e2,
        "cost_index": 6,
    },

    # ── Polymers / Elastomers ─────────────────────────────────────────────────
    "rubber": {
        "category": "polymer",
        "tensile_strength": 15,
        "elongation_pct": 600,
        "malleability": 8,
        "lustre": 1,
        "hardness": 1.0,
        "corrosion_resistance": 5,
        "breathability": 2,
        "density": 1.1,
        "melting_point": 453,
        "thermal_conductivity": 0.16,
        "electrical_conductivity": 1e-14,
        "cost_index": 2,
    },
    "nylon": {
        "category": "polymer",
        "tensile_strength": 75,
        "elongation_pct": 60,
        "malleability": 5,
        "lustre": 3,
        "hardness": 2.0,
        "corrosion_resistance": 7,
        "breathability": 3,
        "density": 1.14,
        "melting_point": 533,
        "thermal_conductivity": 0.25,
        "electrical_conductivity": 1e-12,
        "cost_index": 3,
    },
    "kevlar": {
        "category": "polymer",
        "tensile_strength": 3620,
        "elongation_pct": 3.6,
        "malleability": 1,
        "lustre": 2,
        "hardness": 3.0,
        "corrosion_resistance": 8,
        "breathability": 4,
        "density": 1.44,
        "melting_point": 773,
        "thermal_conductivity": 0.04,
        "electrical_conductivity": 1e-12,
        "cost_index": 7,
    },
    "teflon": {
        "category": "polymer",
        "tensile_strength": 27,
        "elongation_pct": 300,
        "malleability": 6,
        "lustre": 3,
        "hardness": 1.5,
        "corrosion_resistance": 10,
        "breathability": 1,
        "density": 2.2,
        "melting_point": 600,
        "thermal_conductivity": 0.25,
        "electrical_conductivity": 1e-25,
        "cost_index": 5,
    },
    "silicone": {
        "category": "polymer",
        "tensile_strength": 11,
        "elongation_pct": 700,
        "malleability": 9,
        "lustre": 2,
        "hardness": 1.0,
        "corrosion_resistance": 7,
        "breathability": 3,
        "density": 1.1,
        "melting_point": 573,
        "thermal_conductivity": 0.2,
        "electrical_conductivity": 1e-14,
        "cost_index": 4,
    },
    "polyethylene": {
        "category": "polymer",
        "tensile_strength": 33,
        "elongation_pct": 600,
        "malleability": 7,
        "lustre": 2,
        "hardness": 1.5,
        "corrosion_resistance": 8,
        "breathability": 1,
        "density": 0.94,
        "melting_point": 408,
        "thermal_conductivity": 0.42,
        "electrical_conductivity": 1e-16,
        "cost_index": 1,
    },

    # ── Composites / Advanced ─────────────────────────────────────────────────
    "carbon_fiber": {
        "category": "composite",
        "tensile_strength": 3500,
        "elongation_pct": 1.5,
        "malleability": 0,
        "lustre": 5,
        "hardness": 7.0,
        "corrosion_resistance": 9,
        "breathability": 1,
        "density": 1.75,
        "melting_point": 3773,
        "thermal_conductivity": 120,
        "electrical_conductivity": 6e4,
        "cost_index": 8,
    },
    "graphene": {
        "category": "composite",
        "tensile_strength": 130000,
        "elongation_pct": 25,
        "malleability": 5,
        "lustre": 6,
        "hardness": 10.0,
        "corrosion_resistance": 9,
        "breathability": 0,
        "density": 1.0,
        "melting_point": 4900,
        "thermal_conductivity": 5000,
        "electrical_conductivity": 1e8,
        "cost_index": 9,
    },
    "fiberglass": {
        "category": "composite",
        "tensile_strength": 1500,
        "elongation_pct": 4,
        "malleability": 2,
        "lustre": 4,
        "hardness": 6.5,
        "corrosion_resistance": 8,
        "breathability": 1,
        "density": 2.55,
        "melting_point": 1394,
        "thermal_conductivity": 1.0,
        "electrical_conductivity": 1e-10,
        "cost_index": 3,
    },

    # ── Natural / Textiles ────────────────────────────────────────────────────
    "linen": {
        "category": "textile",
        "tensile_strength": 60,
        "elongation_pct": 3,
        "malleability": 4,
        "lustre": 5,
        "hardness": 1.0,
        "corrosion_resistance": 4,
        "breathability": 10,
        "density": 1.5,
        "melting_point": 533,
        "thermal_conductivity": 0.07,
        "electrical_conductivity": 1e-10,
        "cost_index": 4,
    },
    "cotton": {
        "category": "textile",
        "tensile_strength": 40,
        "elongation_pct": 8,
        "malleability": 6,
        "lustre": 3,
        "hardness": 0.5,
        "corrosion_resistance": 3,
        "breathability": 9,
        "density": 1.54,
        "melting_point": 523,
        "thermal_conductivity": 0.04,
        "electrical_conductivity": 1e-10,
        "cost_index": 2,
    },
    "silk": {
        "category": "textile",
        "tensile_strength": 500,
        "elongation_pct": 20,
        "malleability": 7,
        "lustre": 9,
        "hardness": 1.0,
        "corrosion_resistance": 3,
        "breathability": 8,
        "density": 1.34,
        "melting_point": 443,
        "thermal_conductivity": 0.05,
        "electrical_conductivity": 1e-11,
        "cost_index": 8,
    },
    "wool": {
        "category": "textile",
        "tensile_strength": 200,
        "elongation_pct": 30,
        "malleability": 6,
        "lustre": 4,
        "hardness": 0.5,
        "corrosion_resistance": 3,
        "breathability": 8,
        "density": 1.31,
        "melting_point": 403,
        "thermal_conductivity": 0.04,
        "electrical_conductivity": 1e-10,
        "cost_index": 5,
    },
    "leather": {
        "category": "textile",
        "tensile_strength": 20,
        "elongation_pct": 50,
        "malleability": 7,
        "lustre": 5,
        "hardness": 2.0,
        "corrosion_resistance": 4,
        "breathability": 7,
        "density": 0.86,
        "melting_point": 473,
        "thermal_conductivity": 0.14,
        "electrical_conductivity": 1e-10,
        "cost_index": 6,
    },

    # ── Wood / Organic ────────────────────────────────────────────────────────
    "wood": {
        "category": "organic",
        "tensile_strength": 50,
        "elongation_pct": 1,
        "malleability": 2,
        "lustre": 3,
        "hardness": 3.0,
        "corrosion_resistance": 2,
        "breathability": 6,
        "density": 0.6,
        "melting_point": 573,
        "thermal_conductivity": 0.12,
        "electrical_conductivity": 1e-10,
        "cost_index": 1,
    },
    "bamboo": {
        "category": "organic",
        "tensile_strength": 350,
        "elongation_pct": 2,
        "malleability": 3,
        "lustre": 3,
        "hardness": 3.5,
        "corrosion_resistance": 3,
        "breathability": 7,
        "density": 0.8,
        "melting_point": 573,
        "thermal_conductivity": 0.17,
        "electrical_conductivity": 1e-10,
        "cost_index": 1,
    },

    # ── Specialty ─────────────────────────────────────────────────────────────
    "bone": {
        "category": "biological",
        "tensile_strength": 130,
        "elongation_pct": 3,
        "malleability": 1,
        "lustre": 2,
        "hardness": 5.0,
        "corrosion_resistance": 4,
        "breathability": 2,
        "density": 1.85,
        "melting_point": 1943,
        "thermal_conductivity": 0.3,
        "electrical_conductivity": 1e-6,
        "cost_index": 0,
    },
    "concrete": {
        "category": "composite",
        "tensile_strength": 3,
        "elongation_pct": 0,
        "malleability": 0,
        "lustre": 1,
        "hardness": 7.0,
        "corrosion_resistance": 5,
        "breathability": 3,
        "density": 2.4,
        "melting_point": 1773,
        "thermal_conductivity": 1.7,
        "electrical_conductivity": 1e-9,
        "cost_index": 1,
    },

    # ── Biological / Nature-Inspired ─────────────────────────────────────────
    "spider_silk": {
        "category": "biological",
        "tensile_strength": 1300,
        "elongation_pct": 40,
        "malleability": 3,
        "lustre": 4,
        "hardness": 1.5,
        "corrosion_resistance": 5,
        "breathability": 4,
        "density": 1.3,
        "melting_point": 533,
        "thermal_conductivity": 0.18,
        "electrical_conductivity": 1e-11,
        "cost_index": 9,
        "wear_resistance": 7,
        "biocompatibility": 9,
    },
    "earthworm": {
        "category": "biological",
        "tensile_strength": 0.3,
        "elongation_pct": 800,
        "malleability": 10,
        "lustre": 1,
        "hardness": 0.1,
        "corrosion_resistance": 5,
        "breathability": 9,
        "density": 1.05,
        "melting_point": 373,
        "thermal_conductivity": 0.55,
        "electrical_conductivity": 1e-4,
        "cost_index": 0,
        "biocompatibility": 10,
        "water_resistance": 2,
    },
    "muscle_tissue": {
        "category": "biological",
        "tensile_strength": 0.3,
        "elongation_pct": 60,
        "malleability": 9,
        "lustre": 1,
        "hardness": 0.2,
        "corrosion_resistance": 4,
        "breathability": 8,
        "density": 1.06,
        "melting_point": 373,
        "thermal_conductivity": 0.5,
        "electrical_conductivity": 0.3,
        "cost_index": 0,
        "biocompatibility": 10,
        "fatigue_resistance": 9,
    },
    "tendon": {
        "category": "biological",
        "tensile_strength": 100,
        "elongation_pct": 10,
        "malleability": 2,
        "lustre": 1,
        "hardness": 1.0,
        "corrosion_resistance": 5,
        "breathability": 5,
        "density": 1.2,
        "melting_point": 373,
        "thermal_conductivity": 0.4,
        "electrical_conductivity": 1e-5,
        "cost_index": 0,
        "biocompatibility": 10,
        "fatigue_resistance": 8,
    },
    "skin": {
        "category": "biological",
        "tensile_strength": 20,
        "elongation_pct": 70,
        "malleability": 9,
        "lustre": 3,
        "hardness": 0.3,
        "corrosion_resistance": 6,
        "breathability": 9,
        "density": 1.1,
        "melting_point": 373,
        "thermal_conductivity": 0.37,
        "electrical_conductivity": 1e-5,
        "cost_index": 0,
        "biocompatibility": 10,
        "water_resistance": 7,
    },
    "lotus_leaf": {
        "category": "biological",
        "tensile_strength": 1,
        "elongation_pct": 10,
        "malleability": 5,
        "lustre": 3,
        "hardness": 0.2,
        "corrosion_resistance": 7,
        "breathability": 6,
        "density": 0.9,
        "melting_point": 373,
        "thermal_conductivity": 0.2,
        "electrical_conductivity": 1e-12,
        "cost_index": 0,
        "water_resistance": 10,
    },

    # ── Advanced Specialty ───────────────────────────────────────────────────
    "aerogel": {
        "category": "specialty",
        "tensile_strength": 0.5,
        "elongation_pct": 2,
        "malleability": 1,
        "lustre": 2,
        "hardness": 0.5,
        "corrosion_resistance": 7,
        "breathability": 9,
        "density": 0.002,
        "melting_point": 1723,
        "thermal_conductivity": 0.015,
        "electrical_conductivity": 1e-12,
        "cost_index": 8,
    },
    "hydrogel": {
        "category": "polymer",
        "tensile_strength": 0.1,
        "elongation_pct": 1000,
        "malleability": 10,
        "lustre": 1,
        "hardness": 0.05,
        "corrosion_resistance": 5,
        "breathability": 10,
        "density": 1.01,
        "melting_point": 373,
        "thermal_conductivity": 0.6,
        "electrical_conductivity": 1e-3,
        "cost_index": 4,
        "biocompatibility": 10,
        "water_resistance": 1,
    },
    "pdms": {
        "category": "polymer",
        "tensile_strength": 7,
        "elongation_pct": 900,
        "malleability": 10,
        "lustre": 2,
        "hardness": 0.5,
        "corrosion_resistance": 9,
        "breathability": 6,
        "density": 0.97,
        "melting_point": 550,
        "thermal_conductivity": 0.15,
        "electrical_conductivity": 1e-14,
        "cost_index": 4,
        "biocompatibility": 9,
        "water_resistance": 9,
        "transparency": 9,
    },
    "nitinol": {
        "category": "specialty",
        "tensile_strength": 1000,
        "elongation_pct": 8,
        "malleability": 6,
        "lustre": 7,
        "hardness": 6.0,
        "corrosion_resistance": 9,
        "breathability": 0,
        "density": 6.45,
        "melting_point": 1583,
        "thermal_conductivity": 18,
        "electrical_conductivity": 9.1e5,
        "cost_index": 9,
        "biocompatibility": 8,
        "fatigue_resistance": 8,
    },
    "polyurethane_foam": {
        "category": "polymer",
        "tensile_strength": 0.5,
        "elongation_pct": 400,
        "malleability": 9,
        "lustre": 1,
        "hardness": 0.3,
        "corrosion_resistance": 5,
        "breathability": 8,
        "density": 0.03,
        "melting_point": 453,
        "thermal_conductivity": 0.03,
        "electrical_conductivity": 1e-14,
        "cost_index": 2,
        "acoustic_dampening": 9,
    },
    "cork": {
        "category": "organic",
        "tensile_strength": 3,
        "elongation_pct": 40,
        "malleability": 7,
        "lustre": 2,
        "hardness": 1.0,
        "corrosion_resistance": 6,
        "breathability": 7,
        "density": 0.12,
        "melting_point": 573,
        "thermal_conductivity": 0.04,
        "electrical_conductivity": 1e-10,
        "cost_index": 2,
        "acoustic_dampening": 8,
        "water_resistance": 7,
    },
    "balsa": {
        "category": "organic",
        "tensile_strength": 7,
        "elongation_pct": 1,
        "malleability": 3,
        "lustre": 2,
        "hardness": 1.5,
        "corrosion_resistance": 3,
        "breathability": 6,
        "density": 0.12,
        "melting_point": 573,
        "thermal_conductivity": 0.055,
        "electrical_conductivity": 1e-10,
        "cost_index": 2,
    },
    "boron_nitride": {
        "category": "ceramic",
        "tensile_strength": 400,
        "elongation_pct": 0,
        "malleability": 0,
        "lustre": 5,
        "hardness": 9.5,
        "corrosion_resistance": 9,
        "breathability": 0,
        "density": 3.5,
        "melting_point": 3273,
        "thermal_conductivity": 600,
        "electrical_conductivity": 1e-13,
        "cost_index": 8,
    },
    "mxene": {
        "category": "composite",
        "tensile_strength": 2000,
        "elongation_pct": 5,
        "malleability": 4,
        "lustre": 8,
        "hardness": 7.0,
        "corrosion_resistance": 7,
        "breathability": 2,
        "density": 5.0,
        "melting_point": 3273,
        "thermal_conductivity": 55,
        "electrical_conductivity": 5e6,
        "cost_index": 10,
    },
    "basalt_fiber": {
        "category": "composite",
        "tensile_strength": 4840,
        "elongation_pct": 3.2,
        "malleability": 1,
        "lustre": 5,
        "hardness": 8.5,
        "corrosion_resistance": 9,
        "breathability": 0,
        "density": 2.7,
        "melting_point": 1773,
        "thermal_conductivity": 0.03,
        "electrical_conductivity": 1e-8,
        "cost_index": 4,
        "uv_resistance": 9,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PROPERTY SYNONYM MAP — Natural language → property key
# ═══════════════════════════════════════════════════════════════════════════════

# Each entry: (list_of_trigger_words, property_key, is_inverse)
PROPERTY_SYNONYMS = [
    # ── Mechanical Properties ─────────────────────────────────────────────
    (["strong", "tough", "strength", "sturdy", "durable", "robust", "resilient",
      "high-strength", "tensile", "load-bearing", "structural", "withstand",
      "reinforced", "high tensile", "mechanically strong"],
     "tensile_strength", False),
    (["flexible", "bendy", "elastic", "stretchy", "pliable", "springy",
      "supple", "flexibility", "elasticity", "stretchable", "compliant",
      "rubbery", "bio-elastic", "worm-like", "earthworm", "peristaltic",
      "morphing", "adaptive", "parabolic", "stretch", "viscoelastic",
      "rebound", "bounce", "bouncy", "springback", "spring back",
      "elastic recovery", "reversible deformation", "soft", "squishy",
      "conformable", "deformable", "shape-adaptive"],
     "elongation_pct", False),
    (["brittle", "shatters", "breaks easily", "non-flexible", "inelastic",
      "rigid material", "no stretch"],
     "elongation_pct", True),
    (["malleable", "shapeable", "mouldable", "moldable", "ductile", "workable",
      "bend", "shaped", "form", "deform", "formable", "malleability",
      "plastically deform", "cold-formable", "hammerable"],
     "malleability", False),
    (["hard", "rigid", "stiff", "unyielding", "firm", "solid", "hardness",
      "rigidity", "stiffness", "scratch-hard", "inflexible",
      "not dent", "dent-resistant", "indentation-resistant"],
     "hardness", False),
    (["compressive", "compression", "crush-resistant", "load-bearing",
      "compressive-strength", "withstand pressure", "squeeze",
      "structural load", "bearing capacity"],
     "compressive_strength", False),
    (["impact", "impact-resistant", "shock-resistant", "shatterproof",
      "shock-absorbing", "impact-proof", "toughness", "collision",
      "drop-resistant", "shock", "blast-resistant", "ballistic"],
     "impact_resistance", False),
    (["fatigue", "fatigue-resistant", "cyclic", "endurance", "fatigue-life",
      "repeated loading", "cyclic stress", "dynamic loading",
      "long-term stress", "sustained load", "repetitive"],
     "fatigue_resistance", False),
    (["wear-resistant", "abrasion", "scratch-resistant", "wear",
      "abrasion-resistant", "anti-scratch", "scratch", "friction",
      "grind", "grinding", "sliding", "slide", "tribolog",
      "anti-friction", "low friction", "self-lubricating",
      "fretting", "erosion", "surface wear", "galling",
      "no grind", "should not grind", "not grind"],
     "wear_resistance", False),
    (["vibration", "vibration-absorbing", "anti-vibration", "dampening",
      "damping", "shock damping", "oscillation", "resonance reduction",
      "vibration isolation"],
     "acoustic_dampening", False),
    (["shape memory", "shape-memory", "self-healing", "shape recovery",
      "smart material", "actuating", "morphing material", "recovers shape"],
     "fatigue_resistance", False),

    # ── Surface / Optical Properties ──────────────────────────────────────
    (["shiny", "lustrous", "glossy", "reflective", "bright", "gleaming",
      "polished", "mirror", "lustre", "luster", "metallic sheen",
      "iridescent", "pearlescent", "chrome", "gloss"],
     "lustre", False),
    (["transparent", "clear", "see-through", "translucent", "optical",
      "optically-clear", "transparency", "transmittance", "glass-like",
      "window-like", "crystal clear", "see through"],
     "transparency", False),
    (["opaque", "non-transparent", "blocks-light", "not see through",
      "light-blocking", "matte"],
     "transparency", True),
    (["uv-resistant", "uv-stable", "sun-resistant", "uv-protection",
      "sunlight-resistant", "photo-stable", "uv", "outdoor",
      "sun exposure", "light stable", "photodegradation-resistant",
      "weathering resistant", "radiation resistant"],
     "uv_resistance", False),
    (["superhydrophobic", "lotus effect", "self-cleaning", "water-beading",
      "non-wetting", "anti-icing", "anti-fouling", "anti-stick"],
     "water_resistance", False),

    # ── Chemical / Environmental ──────────────────────────────────────────
    (["non-corrosive", "corrosion", "rust-proof", "rustproof",
      "anti-rust", "weather", "weatherproof", "inert", "oxidation",
      "corrosion-resistant", "anti-corrosion", "tarnish-resistant",
      "marine grade", "salt water resistant", "saline"],
     "corrosion_resistance", False),
    (["rust", "rusting", "corrode", "corroding", "oxidize", "tarnish",
      "get rusted", "gets rusty", "rusts"],
     "corrosion_resistance", True),
    (["chemical-resistant", "chemically-stable", "acid-resistant",
      "alkali-resistant", "solvent-resistant", "chemical-stability",
      "chemically-inert", "chemical", "solvent", "base resistant",
      "ph-stable", "reactive environment", "harsh chemical"],
     "chemical_stability", False),
    (["biocompatible", "biocompatibility", "bio-compatible", "body-safe",
      "non-toxic", "hypoallergenic", "medical-grade", "bio-inert",
      "tissue-compatible", "implantable", "body-friendly",
      "in-vivo", "bio-inspired", "living tissue", "organic"],
     "biocompatibility", False),
    (["waterproof", "water-resistant", "hydrophobic", "moisture-resistant",
      "water-repellent", "waterproofing", "impermeable", "submersible",
      "water-tight", "moisture-proof", "rain-proof", "splash-proof"],
     "water_resistance", False),

    # ── Thermal Properties ────────────────────────────────────────────────
    (["heat-resistant", "refractory", "fireproof", "fire-resistant",
      "high-temperature", "heat-proof", "flame-retardant", "flame-resistant",
      "high-melting", "thermally-stable", "heat resistant",
      "withstand heat", "thermal stability", "high heat",
      "does not melt", "won't melt", "extreme temperature"],
     "melting_point", False),
    (["heat-dissipating", "heat-dissipation", "cooling", "heat-sink",
      "thermal-dissipation", "dissipate-heat", "heat dissipation",
      "thermally-conductive", "heat-spreading", "let heat out",
      "let the heat go", "internal heat", "heat go out",
      "heat escape", "release heat", "conducts heat",
      "thermal management", "thermal pathway", "thermal bridge",
      "heat transfer", "stay cool", "cooling efficiency",
      "thermal regulation", "active cooling", "passive cooling"],
     "thermal_conductivity", False),
    (["heat-insulating", "thermal-insulator", "thermal-insulation",
      "thermally-insulating", "low-thermal", "heat insulating",
      "keeps heat", "retains heat", "heat retention",
      "no heat loss", "thermal barrier", "keeps warm", "warm"],
     "thermal_conductivity", True),

    # ── Electrical Properties ─────────────────────────────────────────────
    (["conductive", "conductor", "electrical", "current", "conduct",
      "electrically-conductive", "electricity", "conducts-electricity",
      "conduct-electricity", "electric", "ohmic",
      "piezoelectric", "semiconductor", "semi-conductive"],
     "electrical_conductivity", False),
    (["insulating", "insulator", "non-conductive", "dielectric",
      "electrically-insulating", "doesnt-conduct", "insulation",
      "no electricity", "not conduct electricity",
      "should not conduct", "electrically neutral",
      "electromagnetic shielding", "emi shielding"],
     "electrical_conductivity", True),

    # ── Permeability / Porosity ───────────────────────────────────────────
    (["breathable", "porous", "airy", "ventilated", "permeable",
      "breathability", "porosity", "air-permeable", "gas-permeable",
      "permeability", "moisture-wicking", "gas exchange",
      "transpiration", "allows airflow", "skin-breathing"],
     "breathability", False),
    (["sealed", "airtight", "hermetic", "gas-tight", "non-porous",
      "impervious", "barrier", "no airflow", "vapor barrier"],
     "breathability", True),

    # ── Weight / Density ──────────────────────────────────────────────────
    (["lightweight", "light", "featherweight", "ultralight", "thin",
      "low-density", "weight", "low-weight", "minimal weight",
      "portable", "low mass", "gossamer", "paper-thin"],
     "density", True),
    (["heavy", "dense", "weighty", "massive", "high-density", "heavyweight",
      "solid mass", "high mass"],
     "density", False),

    # ── Magnetic Properties ───────────────────────────────────────────────
    (["magnetic", "ferromagnetic", "magnetizable", "magnetically-active",
      "magnet", "magnetism", "ferrite", "magnetic field"],
     "magnetic_property", False),
    (["non-magnetic", "diamagnetic", "paramagnetic", "amagnetic",
      "magnetically-inert", "mri-safe", "mri safe", "not magnetic",
      "non-ferrous"],
     "magnetic_property", True),

    # ── Acoustic Properties ───────────────────────────────────────────────
    (["sound-absorbing", "acoustic", "sound-dampening", "noise-reducing",
      "sound-insulating", "soundproof", "noise-dampening", "quiet",
      "vibration absorbing", "anti-vibration", "damping",
      "noise cancellation", "acoustic isolation"],
     "acoustic_dampening", False),
    (["resonant", "sound-conducting", "acoustic-conductor", "reverberant"],
     "acoustic_dampening", True),

    # ── Cost ──────────────────────────────────────────────────────────────
    (["cheap", "affordable", "inexpensive", "economical", "low-cost",
      "budget", "cost-effective", "scalable", "mass-producible",
      "commercially available", "off-the-shelf"],
     "cost_index", True),
    (["expensive", "premium", "high-end", "luxury", "costly", "rare",
      "exotic", "specialized"],
     "cost_index", False),
]

# ═══════════════════════════════════════════════════════════════════════════════
# 2b. NEGATION PATTERNS — Detecting "should NOT", "doesn't", "no", etc.
# ═══════════════════════════════════════════════════════════════════════════════

NEGATION_WORDS = [
    "not", "no", "don't", "dont", "doesn't", "doesnt", "shouldn't", "shouldnt",
    "should not", "must not", "mustn't", "mustnt", "without", "non", "never",
    "isn't", "isnt", "won't", "wont", "cannot", "can't", "cant", "nor",
    "avoid", "exclude", "zero", "none", "lack", "free of"
]

# Default values for HIGH and LOW targets when user makes direct statements
# e.g., "it should conduct electricity" → electrical_conductivity = HIGH_DEFAULT
PROPERTY_DEFAULTS = {
    "tensile_strength":       {"high": 500, "low": 20},
    "elongation_pct":         {"high": 400, "low": 2},
    "malleability":           {"high": 9, "low": 1},
    "hardness":               {"high": 9, "low": 1},
    "compressive_strength":   {"high": 500, "low": 10},
    "impact_resistance":      {"high": 9, "low": 1},
    "fatigue_resistance":     {"high": 9, "low": 1},
    "wear_resistance":        {"high": 9, "low": 1},
    "lustre":                 {"high": 9, "low": 1},
    "transparency":           {"high": 9, "low": 0},
    "uv_resistance":          {"high": 9, "low": 1},
    "corrosion_resistance":   {"high": 9, "low": 1},
    "chemical_stability":     {"high": 9, "low": 1},
    "biocompatibility":       {"high": 9, "low": 1},
    "water_resistance":       {"high": 9, "low": 1},
    "melting_point":          {"high": 2500, "low": 400},
    "thermal_conductivity":   {"high": 400, "low": 0.1},
    "electrical_conductivity": {"high": 1e7, "low": 1e-12},
    "breathability":          {"high": 9, "low": 0},
    "density":                {"high": 15, "low": 1.5},
    "magnetic_property":      {"high": 9, "low": 0},
    "acoustic_dampening":     {"high": 9, "low": 1},
    "cost_index":             {"high": 9, "low": 1},
}

# Material name aliases (user might say "steel" or "stainless steel" etc.)
MATERIAL_ALIASES = {
    "steel": "steel",
    "stainless steel": "stainless_steel",
    "stainless": "stainless_steel",
    "copper": "copper",
    "gold": "gold",
    "aluminum": "aluminum",
    "aluminium": "aluminum",
    "titanium": "titanium",
    "iron": "iron",
    "silver": "silver",
    "platinum": "platinum",
    "tungsten": "tungsten",
    "nickel": "nickel",
    "zinc": "zinc",
    "ceramic": "ceramic",
    "ceramics": "ceramic",
    "glass": "glass",
    "diamond": "diamond",
    "silicon carbide": "silicon_carbide",
    "sic": "silicon_carbide",
    "rubber": "rubber",
    "nylon": "nylon",
    "kevlar": "kevlar",
    "teflon": "teflon",
    "ptfe": "teflon",
    "silicone": "silicone",
    "polyethylene": "polyethylene",
    "pe": "polyethylene",
    "carbon fiber": "carbon_fiber",
    "carbon fibre": "carbon_fiber",
    "cf": "carbon_fiber",
    "graphene": "graphene",
    "fiberglass": "fiberglass",
    "fibreglass": "fiberglass",
    "linen": "linen",
    "cotton": "cotton",
    "silk": "silk",
    "wool": "wool",
    "leather": "leather",
    "wood": "wood",
    "bamboo": "bamboo",
    "bone": "bone",
    "concrete": "concrete",
    "cement": "concrete",
    # Biological / Nature-inspired
    "spider silk": "spider_silk",
    "spiderweb": "spider_silk",
    "spider web": "spider_silk",
    "dragline silk": "spider_silk",
    "earthworm": "earthworm",
    "worm": "earthworm",
    "muscle": "muscle_tissue",
    "muscles": "muscle_tissue",
    "muscle tissue": "muscle_tissue",
    "tendon": "tendon",
    "ligament": "tendon",
    "skin": "skin",
    "human skin": "skin",
    "lotus leaf": "lotus_leaf",
    "lotus": "lotus_leaf",
    # Specialty / Advanced
    "aerogel": "aerogel",
    "silica aerogel": "aerogel",
    "hydrogel": "hydrogel",
    "gel": "hydrogel",
    "pdms": "pdms",
    "polydimethylsiloxane": "pdms",
    "silicone elastomer": "pdms",
    "nitinol": "nitinol",
    "shape memory alloy": "nitinol",
    "sma": "nitinol",
    "ni-ti": "nitinol",
    "polyurethane foam": "polyurethane_foam",
    "foam": "polyurethane_foam",
    "memory foam": "polyurethane_foam",
    "cork": "cork",
    "balsa": "balsa",
    "balsa wood": "balsa",
    "boron nitride": "boron_nitride",
    "bn": "boron_nitride",
    "hexagonal boron nitride": "boron_nitride",
    "mxene": "mxene",
    "basalt fiber": "basalt_fiber",
    "basalt fibre": "basalt_fiber",
    "basalt": "basalt_fiber",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EXTRACTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def find_material_in_text(text):
    """Find all mentioned materials in user text, checking longer names first."""
    text_lower = text.lower()
    found = {}
    # Sort aliases by length (longest first) to match "stainless steel" before "steel"
    sorted_aliases = sorted(MATERIAL_ALIASES.keys(), key=len, reverse=True)
    for alias in sorted_aliases:
        if alias in text_lower:
            db_key = MATERIAL_ALIASES[alias]
            if db_key not in found:
                found[db_key] = MATERIALS_DB[db_key]
    return found


def find_properties_in_text(text):
    """Find all property references in user text."""
    text_lower = text.lower()
    found = []
    for synonyms, prop_key, is_inverse in PROPERTY_SYNONYMS:
        for word in synonyms:
            if word in text_lower:
                found.append({
                    "trigger_word": word,
                    "property": prop_key,
                    "is_inverse": is_inverse
                })
                break  # Only match first synonym per group
    return found


import re

def _check_negation(text, prop_position):
    """
    Check if a property mention is negated in the surrounding context.
    Looks at the 60 chars before the property word for negation words.
    Returns True if negated.
    """
    # Get the text window before the property mention
    # Reduced from 60 to 25 so "not" doesn't carry over "and" to other properties
    start = max(0, prop_position - 25)
    window = text[start:prop_position].lower()
    
    # Pre-process window to fix common typos like "should nt" -> "shouldnt"
    window = window.replace(" nt ", "nt ")
    
    for neg in NEGATION_WORDS:
        if re.search(r'\b' + re.escape(neg) + r'\b', window):
            return True
    return False


def extract_direct_statements(text):
    """
    Extract property requirements from plain English statements.
    Handles negation: 'should not conduct electricity' → LOW conductivity.
    Handles affirmation: 'should be strong' → HIGH strength.
    
    Returns list of dicts:
        {property, value, is_high, trigger_phrase, negated}
    """
    text_lower = text.lower()
    results = []
    seen_props = set()
    
    for synonyms, prop_key, base_inverse in PROPERTY_SYNONYMS:
        for word in synonyms:
            pos = text_lower.find(word)
            if pos == -1:
                continue
            if prop_key in seen_props:
                break
            seen_props.add(prop_key)
            
            # Check for negation in surrounding context
            negated = _check_negation(text_lower, pos)
            
            # Determine if user wants HIGH or LOW
            # base_inverse=True means the synonym implies LOW (e.g., "lightweight" → low density)
            wants_high = not base_inverse
            if negated:
                wants_high = not wants_high  # flip for negation
            
            # Get default value
            defaults = PROPERTY_DEFAULTS.get(prop_key, {"high": 9, "low": 1})
            value = defaults["high"] if wants_high else defaults["low"]
            
            # Build a human-readable trigger phrase
            context_start = max(0, pos - 30)
            context_end = min(len(text_lower), pos + len(word) + 10)
            trigger = text_lower[context_start:context_end].strip()
            if len(trigger) > 50:
                trigger = trigger[:50] + "..."
            
            results.append({
                "property": prop_key,
                "value": value,
                "is_high": wants_high,
                "negated": negated,
                "trigger_phrase": trigger,
                "display_intent": ("HIGH" if wants_high else "LOW") + f" {prop_key.replace('_', ' ')}",
            })
            break  # Only first synonym match per group
    
    return results


def _resolve_material(text_fragment):
    """Resolve a text fragment to a material DB key using alias matching."""
    text_fragment = text_fragment.strip().lower()
    # Sort aliases by length (longest first) to match "stainless steel" before "steel"
    sorted_aliases = sorted(MATERIAL_ALIASES.keys(), key=len, reverse=True)
    for alias in sorted_aliases:
        # Use word boundary matching to avoid "iron" matching inside "environment"
        if re.search(r'\b' + re.escape(alias) + r'\b', text_fragment):
            return MATERIAL_ALIASES[alias], alias
    return None, None


def _resolve_property(word):
    """Resolve a word to a property key."""
    word = word.strip().lower()
    for synonyms, pk, inv in PROPERTY_SYNONYMS:
        if word in synonyms:
            return pk, inv
    return None, False


def extract_analogies(text):
    """
    Extract material analogies from multiple natural language patterns:
      1. "as [PROP] as [MAT]"        — "as strong as steel"
      2. "[PROP] like [MAT]"         — "non-corrosive like titanium"
      3. "[PROP] of [MAT]"           — "strength of steel"
      4. "[PROP] as [MAT]"           — "flexible as rubber"
      5. "[MAT]-like [PROP]"         — implied from material mention
    Returns list of {property, material, value, unit} dicts.
    """
    text_lower = text.lower()
    analogies = []
    seen = set()  # avoid duplicates by (property, material) pair

    # ── Pattern 1: "as <prop> as <material>" ─────────────────────────────
    pat1 = r'as\s+([\w-]+)\s+as\s+([\w\s]+?)(?:\s*[,;.]|\s+(?:and|but|with|for|in|at|while|that)\b|$)'
    for m in re.finditer(pat1, text_lower):
        _add_analogy(analogies, seen, m.group(1), m.group(2), "as {} as {}")

    # ── Pattern 2: "<prop> like <material>" ──────────────────────────────
    pat2 = r'([\w-]+)\s+like\s+([\w\s]+?)(?:\s*[,;.]|\s+(?:and|but|with|for|in|at|while|that)\b|$)'
    for m in re.finditer(pat2, text_lower):
        _add_analogy(analogies, seen, m.group(1), m.group(2), "{} like {}")

    # ── Pattern 3: "<prop> of <material>" ────────────────────────────────
    pat3 = r'([\w-]+)\s+of\s+([\w\s]+?)(?:\s*[,;.]|\s+(?:and|but|with|for|in|at|while|that)\b|$)'
    for m in re.finditer(pat3, text_lower):
        _add_analogy(analogies, seen, m.group(1), m.group(2), "{} of {}")

    # ── Pattern 4: "<prop> as <material>" (without leading "as") ─────────
    pat4 = r'(?<!as\s)([\w-]+)\s+as\s+([\w\s]+?)(?:\s*[,;.]|\s+(?:and|but|with|for|in|at|while|that)\b|$)'
    for m in re.finditer(pat4, text_lower):
        # Skip if this was already captured by pattern 1
        _add_analogy(analogies, seen, m.group(1), m.group(2), "{} as {}")

    return analogies


def _add_analogy(analogies, seen, prop_word, mat_word, trigger_fmt):
    """Helper to resolve and add a single analogy if valid."""
    prop_word = prop_word.strip()
    mat_word = mat_word.strip()

    mat_key, mat_alias = _resolve_material(mat_word)
    if mat_key is None:
        return

    prop_key, is_inverse = _resolve_property(prop_word)
    if prop_key is None:
        return

    pair = (prop_key, mat_key)
    if pair in seen:
        return
    seen.add(pair)

    mat_data = MATERIALS_DB[mat_key]
    value = mat_data.get(prop_key, None)
    if value is not None:
        analogies.append({
            "property": prop_key,
            "material": mat_key,
            "material_display": mat_alias.title() if mat_alias else mat_key.replace('_', ' ').title(),
            "value": value,
            "is_inverse": is_inverse,
            "trigger": trigger_fmt.format(prop_word, mat_alias or mat_word),
        })


def extract_numeric_conditions(text):
    """
    Extract explicit numeric conditions:
    - Temperature: "350K", "at 500 K", "500 kelvin"
    - pH: "pH 2", "pH of 3.5", "acidic" (→2), "alkaline"/"basic" (→12)
    - Pressure: "10 atm", "high pressure"
    """
    text_lower = text.lower()
    conditions = {}

    # Temperature: "350K", "at 350 K", "500 kelvin", "350 degrees"
    temp_match = re.search(r'(\d+)\s*(?:k(?:elvin)?|°)\b', text_lower)
    if temp_match:
        conditions['temperature'] = float(temp_match.group(1))

    # pH
    ph_match = re.search(r'ph\s*(?:of\s*)?(\d+\.?\d*)', text_lower)
    if ph_match:
        conditions['ph'] = float(ph_match.group(1))
    elif 'acidic' in text_lower or 'acid' in text_lower:
        conditions['ph'] = 2.0
    elif 'alkaline' in text_lower or 'basic' in text_lower or 'alkali' in text_lower:
        conditions['ph'] = 12.0

    # Pressure
    pressure_match = re.search(r'(\d+\.?\d*)\s*(?:atm|bar|mpa)', text_lower)
    if pressure_match:
        conditions['pressure'] = float(pressure_match.group(1))
    elif 'high pressure' in text_lower:
        conditions['pressure'] = 10.0
    elif 'low pressure' in text_lower or 'vacuum' in text_lower:
        conditions['pressure'] = 0.1

    return conditions


def extract_application_context(text):
    """Extract application domain from text."""
    text_lower = text.lower()
    APPLICATIONS = {
        "aerospace": {"temperature": 600, "tensile_strength_min": 500, "density_max": 5.0,
                      "description": "Aerospace — high strength-to-weight, extreme heat resistance"},
        "biomedical": {"corrosion_resistance_min": 8, "breathability_min": 3,
                       "description": "Biomedical — biocompatibility, corrosion resistance, body-safe"},
        "automotive": {"tensile_strength_min": 300, "density_max": 8.0,
                       "description": "Automotive — strength, wear resistance, moderate weight"},
        "marine": {"corrosion_resistance_min": 9, "ph": 7.5,
                   "description": "Marine / Underwater — extreme corrosion & salt resistance"},
        "electronics": {"electrical_conductivity_min": 1e6,
                        "description": "Electronics — high electrical conductivity, thermal management"},
        "construction": {"tensile_strength_min": 200, "hardness_min": 5,
                         "description": "Construction / Structural — high strength and hardness"},
        "textile": {"breathability_min": 6, "elongation_pct_min": 30,
                    "description": "Textile / Wearable — breathability, flexibility, comfort"},
        "catalysis": {"temperature": 350, "ph": 2.0,
                      "description": "Catalysis / Chemical — thermal stability in reactive environments"},
        "energy": {"thermal_conductivity_min": 100,
                   "description": "Energy / Thermal Management — heat dissipation capability"},
        "robotics": {"elongation_pct_min": 50, "fatigue_resistance_min": 8,
                     "description": "Robotics — flexibility, fatigue resistance, actuator materials"},
        "soft robotics": {"elongation_pct_min": 200, "density_max": 2.0,
                          "description": "Soft Robotics — extreme flexibility, lightweight, compliant"},
        "defense": {"tensile_strength_min": 800, "impact_resistance_min": 8,
                    "description": "Defense / Military — ballistic, impact, blast resistance"},
        "sports": {"elongation_pct_min": 30, "density_max": 3.0,
                   "description": "Sports / Athletic — lightweight, flexible, impact resistant"},
        "packaging": {"density_max": 2.0, "corrosion_resistance_min": 5,
                      "description": "Packaging — lightweight, barrier properties, low cost"},
        "semiconductor": {"electrical_conductivity_min": 1e2,
                          "description": "Semiconductor — controlled conductivity, thermal stability"},
        "nuclear": {"melting_point_min": 2500, "corrosion_resistance_min": 8,
                    "description": "Nuclear — extreme temperature & radiation resistance"},
        "food": {"biocompatibility_min": 9, "corrosion_resistance_min": 7,
                 "description": "Food / Pharmaceutical — food-safe, non-toxic, corrosion resistant"},
        "thermal insulation": {"thermal_conductivity_max": 0.1,
                               "description": "Thermal Insulation — extremely low thermal conductivity"},
        "acoustic": {"acoustic_dampening_min": 7,
                     "description": "Acoustic / Soundproofing — high sound absorption"},
        "optical": {"transparency_min": 7,
                    "description": "Optical — high transparency, optical clarity"},
    }

    # Single keyword match
    for keyword, ctx in APPLICATIONS.items():
        if keyword in text_lower:
            return ctx

    # Multi-word and synonym phrase detection
    if any(w in text_lower for w in ["water splitting", "catalyst", "photocatalyst", "electrochemical"]):
        return APPLICATIONS["catalysis"]
    if any(w in text_lower for w in ["implant", "prosthe", "surgery", "in-vivo", "body", "medical device"]):
        return APPLICATIONS["biomedical"]
    if any(w in text_lower for w in ["airplane", "rocket", "spacecraft", "drone", "satellite", "hypersonic"]):
        return APPLICATIONS["aerospace"]
    if any(w in text_lower for w in ["underwater", "ocean", "sea", "subsea", "offshore", "saltwater"]):
        return APPLICATIONS["marine"]
    if any(w in text_lower for w in ["battery", "solar", "fuel cell", "heat exchanger", "thermoelectric"]):
        return APPLICATIONS["energy"]
    if any(w in text_lower for w in ["building", "structural", "bridge", "skyscraper", "infrastructure"]):
        return APPLICATIONS["construction"]
    if any(w in text_lower for w in ["wearable", "clothing", "garment", "fabric", "apparel", "suit"]):
        return APPLICATIONS["textile"]
    if any(w in text_lower for w in ["robot arm", "actuator", "servo", "mechanical arm"]):
        return APPLICATIONS["robotics"]
    if any(w in text_lower for w in ["soft robot", "gripper", "pneumatic", "peristaltic"]):
        return APPLICATIONS["soft robotics"]
    if any(w in text_lower for w in ["armor", "bulletproof", "blast", "military", "ballistic"]):
        return APPLICATIONS["defense"]
    if any(w in text_lower for w in ["bicycle", "helmet", "ski", "tennis", "running shoe", "athletic"]):
        return APPLICATIONS["sports"]
    if any(w in text_lower for w in ["chip", "transistor", "wafer", "pcb", "microelectronics"]):
        return APPLICATIONS["semiconductor"]
    if any(w in text_lower for w in ["reactor", "nuclear", "radiation", "radioactive"]):
        return APPLICATIONS["nuclear"]
    if any(w in text_lower for w in ["food safe", "edible", "fda", "pharmaceutical", "drug delivery"]):
        return APPLICATIONS["food"]
    if any(w in text_lower for w in ["sound", "noise", "acoustic", "studio", "soundproof"]):
        return APPLICATIONS["acoustic"]
    if any(w in text_lower for w in ["lens", "optical", "fiber optic", "photonic", "window"]):
        return APPLICATIONS["optical"]
    if any(w in text_lower for w in ["insulate", "insulation", "keep warm", "cold", "cryogenic"]):
        return APPLICATIONS["thermal insulation"]

    return None



# ═══════════════════════════════════════════════════════════════════════════════
# 4. PROPERTY → GENERATION PARAMETER MAPPER
# ═══════════════════════════════════════════════════════════════════════════════

PROPERTY_UNITS = {
    "tensile_strength": "MPa",
    "elongation_pct": "%",
    "malleability": "/10",
    "lustre": "/10",
    "hardness": "Mohs",
    "compressive_strength": "MPa",
    "impact_resistance": "/10",
    "fatigue_resistance": "/10",
    "wear_resistance": "/10",
    "transparency": "/10",
    "uv_resistance": "/10",
    "corrosion_resistance": "/10",
    "chemical_stability": "/10",
    "biocompatibility": "/10",
    "water_resistance": "/10",
    "breathability": "/10",
    "density": "g/cm³",
    "melting_point": "K",
    "thermal_conductivity": "W/mK",
    "electrical_conductivity": "S/m",
    "magnetic_property": "/10",
    "acoustic_dampening": "/10",
    "cost_index": "/10",
}

PROPERTY_DISPLAY_NAMES = {
    "tensile_strength": "Tensile Strength",
    "elongation_pct": "Elongation / Flexibility",
    "malleability": "Malleability",
    "lustre": "Lustre / Shine",
    "hardness": "Hardness",
    "compressive_strength": "Compressive Strength",
    "impact_resistance": "Impact Resistance",
    "fatigue_resistance": "Fatigue Resistance",
    "wear_resistance": "Wear / Abrasion Resistance",
    "transparency": "Transparency",
    "uv_resistance": "UV Resistance",
    "corrosion_resistance": "Corrosion Resistance",
    "chemical_stability": "Chemical Stability",
    "biocompatibility": "Biocompatibility",
    "water_resistance": "Water Resistance",
    "breathability": "Breathability / Permeability",
    "density": "Density",
    "melting_point": "Melting Point",
    "thermal_conductivity": "Thermal Conductivity",
    "electrical_conductivity": "Electrical Conductivity",
    "magnetic_property": "Magnetic Property",
    "acoustic_dampening": "Acoustic Dampening",
    "cost_index": "Cost Index",
}

PROPERTY_ICONS = {
    "tensile_strength": "🔩",
    "elongation_pct": "🔄",
    "malleability": "🔨",
    "lustre": "✨",
    "hardness": "💎",
    "compressive_strength": "🏗️",
    "impact_resistance": "💥",
    "fatigue_resistance": "🔁",
    "wear_resistance": "🛡️",
    "transparency": "🔍",
    "uv_resistance": "☀️",
    "corrosion_resistance": "🧪",
    "chemical_stability": "⚗️",
    "biocompatibility": "🧬",
    "water_resistance": "💧",
    "breathability": "🌬️",
    "density": "⚖️",
    "melting_point": "🌡️",
    "thermal_conductivity": "🔥",
    "electrical_conductivity": "⚡",
    "magnetic_property": "🧲",
    "acoustic_dampening": "🔇",
    "cost_index": "💰",
}


def map_properties_to_params(extracted_properties):
    """
    Convert extracted material properties into generation parameters.
    
    Args:
        extracted_properties: dict of {property_key: target_value}
    
    Returns:
        dict of generation parameters for server.py/synthesize_molecule
    """
    params = {
        "temperature": 298,
        "pressure": 1.0,
        "ph": 7.0,
        "humidity": 50,
        "dielectric": 78.5,
        "ionic_strength": 0.1,
        "viscosity": 1.0,
        "steps": 200,
        "noise_scale": 1.0,
        "pino_weight": 0.1,
        "guidance": 1.0,
        "seed": 42,
        "bond_threshold": 1.8,
        "doping_prob": 0.15,
        "max_heavy_atoms": 9,
        "flexibility": 1.0,
        "quantum_ensemble": 0.5,
        "wave_packet": 1.0,
        "tunnelling_depth": 0.1,
    }

    props = extracted_properties

    # 1. MECHANICAL ─────────────────────────────────────────────────────────
    # High tensile strength → stronger bonds, lower noise, higher guidance
    if "tensile_strength" in props:
        ts = props["tensile_strength"]
        if ts > 1000:
            params["guidance"] = 2.0
            params["noise_scale"] = 0.7
            params["pino_weight"] = 0.2
            params["bond_threshold"] = 1.6
        elif ts > 300:
            params["guidance"] = 1.5
            params["noise_scale"] = 0.85
            params["pino_weight"] = 0.15

    # High flexibility/elongation → more quantum tunnelling, higher wave packet
    if "elongation_pct" in props:
        ep = props["elongation_pct"]
        if ep > 200:
            params["flexibility"] = 2.0
            params["wave_packet"] = 1.5
            params["tunnelling_depth"] = 0.2
            params["noise_scale"] = 1.2
        elif ep > 30:
            params["flexibility"] = 1.3
            params["wave_packet"] = 1.2

    # High hardness → tighter bonds, more PINO refinement
    if "hardness" in props:
        h = props["hardness"]
        if h > 7:
            params["pino_weight"] = 0.25
            params["bond_threshold"] = 1.5
            params["guidance"] = 2.0
        elif h > 4:
            params["pino_weight"] = 0.15

    # Malleability → increased flexibility at the cost of stability
    if "malleability" in props:
        m = props["malleability"]
        if m > 7:
            params["flexibility"] = 1.8
            params["noise_scale"] = 1.15

    # Impact/Fatigue/Wear resistance → High stability, low PDE residuals
    if any(p in props for p in ["impact_resistance", "fatigue_resistance", "wear_resistance"]):
        # These all require highly stable, well-ordered crystal-like structures
        params["pino_weight"] = 0.2
        params["steps"] = 300
        params["noise_scale"] = 0.9

    # 2. SURFACE & OPTICAL ──────────────────────────────────────────────────
    # Lustre/Transparency → Highly ordered, symmetric configurations
    if "lustre" in props or "transparency" in props:
        params["guidance"] = 2.0
        params["pino_weight"] = 0.15
        params["bond_threshold"] = 1.7 # tighter for clean surface

    # 3. CHEMICAL / ENVIRONMENTAL ───────────────────────────────────────────
    # Corrosion resistance / Chemical stability → adjust doping and pH tolerance
    if "corrosion_resistance" in props or "chemical_stability" in props or "uv_resistance" in props:
        cr = props.get("corrosion_resistance", props.get("chemical_stability", 5))
        if cr > 7:
            params["doping_prob"] = 0.25  # More diverse elements for passivating layers
            params["ionic_strength"] = 0.05

    # Biocompatibility → prioritize stable, low-energy configurations
    if "biocompatibility" in props:
        if props["biocompatibility"] > 8:
            params["pino_weight"] = 0.2
            params["max_heavy_atoms"] = 7 # simpler molecules are often less toxic

    # 4. PHYSICAL / ELECTRICAL ─────────────────────────────────────────────
    # Thermal/Electrical Conductivity → increase doping (electron/phonon carriers)
    if "electrical_conductivity" in props or "thermal_conductivity" in props:
        cond = props.get("electrical_conductivity", props.get("thermal_conductivity", 0))
        if cond > 1e6 or cond > 100:
            params["doping_prob"] = 0.3 # High doping for conductivity
            params["steps"] = 300 # more refinement for lattice order

    # Magnetic property → Higher guidance for specific orientation
    if "magnetic_property" in props:
        if props["magnetic_property"] > 7:
            params["guidance"] = 2.2 # Tight guidance

    # Acoustic Dampening → porous, heterogeneous structures
    if "acoustic_dampening" in props:
        if props["acoustic_dampening"] > 7:
            params["noise_scale"] = 1.4
            params["wave_packet"] = 1.4

    # 5. PERMEABILITY & WEIGHT ─────────────────────────────────────────────
    # Breathability → more porosity in structure
    if "breathability" in props:
        br = props["breathability"]
        if br > 5:
            params["noise_scale"] = 1.3
            params["quantum_ensemble"] = 0.7
            params["tunnelling_depth"] = 0.15

    # Density target
    if "density" in props:
        d = props["density"]
        if d < 3.0:  # lightweight
            params["max_heavy_atoms"] = 7
        elif d > 10:  # very dense
            params["doping_prob"] = 0.3
            params["max_heavy_atoms"] = 12

    # Melting point → temperature and stability
    if "melting_point" in props:
        mp = props["melting_point"]
        if mp > 2000:
            params["temperature"] = max(params["temperature"], 500)
            params["guidance"] = 2.0

    # 6. ECONOMIC ──────────────────────────────────────────────────────────
    # Cost Index → lower cost triggers simpler structures
    if "cost_index" in props:
        if props["cost_index"] < 3: # Cheap
            params["max_heavy_atoms"] = 6 # simpler to synthesize
            params["doping_prob"] = 0.05

    return params
