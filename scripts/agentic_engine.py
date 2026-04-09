"""
Agentic Conversation Engine
============================
A rule-based conversational AI engine that interprets user prompts about
material design through natural language analogies. Works like Gemini:
guides non-expert users, asks clarifying questions, and auto-maps material
properties to generation parameters.

State Machine:
    GREETING → COLLECTING → CLARIFYING → CONFIRMING → GENERATING → PRESENTING
"""

import re
import uuid
import random
from scripts.material_properties import (
    MATERIALS_DB, MATERIAL_ALIASES,
    PROPERTY_DISPLAY_NAMES, PROPERTY_UNITS, PROPERTY_ICONS, PROPERTY_DEFAULTS,
    extract_analogies, extract_numeric_conditions, extract_application_context,
    find_material_in_text, find_properties_in_text, extract_direct_statements,
    map_properties_to_params,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONVERSATION STATES
# ═══════════════════════════════════════════════════════════════════════════════

STATE_GREETING    = "greeting"
STATE_COLLECTING  = "collecting"
STATE_CLARIFYING  = "clarifying"
STATE_CONFIRMING  = "confirming"
STATE_GENERATING  = "generating"
STATE_PRESENTING  = "presenting"
STATE_ITERATING   = "iterating"

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STORE (in-memory; production would use Redis/DB)
# ═══════════════════════════════════════════════════════════════════════════════

_sessions = {}

def get_or_create_session(session_id=None):
    if session_id and session_id in _sessions:
        return _sessions[session_id]
    sid = session_id or str(uuid.uuid4())
    session = {
        "id": sid,
        "state": STATE_GREETING,
        "extracted_properties": {},   # {prop_key: value}
        "analogies": [],              # list of analogy dicts
        "conditions": {},             # {temperature, ph, pressure}
        "application": None,          # application context dict
        "raw_messages": [],           # conversation history
        "missing_info": [],           # what we still need
        "conflicts": [],              # conflicting property pairs
        "generation_params": None,    # final mapped params
        "mentioned_materials": {},    # materials found in conversation
    }
    _sessions[sid] = session
    return session

def reset_session(session_id):
    if session_id in _sessions:
        del _sessions[session_id]


# ═══════════════════════════════════════════════════════════════════════════════
# CONVERSATION LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def process_message(session_id, user_message):
    """
    Core function: process a user message and return an AI response.

    Args:
        session_id: str — session identifier
        user_message: str — the user's raw text message

    Returns:
        dict with:
            session_id, state, response (str), extracted_params (dict),
            analogies (list), conditions (dict), application (dict or None),
            suggestions (list of str), ready_to_generate (bool),
            property_cards (list of dicts for UI display)
    """
    session = get_or_create_session(session_id)
    session["raw_messages"].append({"role": "user", "text": user_message})
    msg_lower = user_message.lower().strip()

    # ── Check for short commands ─────────────────────────────────────────
    if msg_lower in ("yes", "go", "generate", "yes!", "proceed", "let's go",
                     "do it", "generate!", "yes generate", "start", "run it",
                     "make it", "synthesize", "create", "build it"):
        if session["extracted_properties"] or session["conditions"]:
            return _build_generation_ready_response(session)

    if msg_lower in ("reset", "start over", "new", "clear"):
        reset_session(session_id)
        return process_message(session_id, "")

    # ── Greeting state (only when no input is given) ────────────────────
    if msg_lower == "":
        session["state"] = STATE_COLLECTING
        return _greeting_response(session)

    # ── All other states: parse the message ──────────────────────────────────
    # 1. Extract analogies ("as strong as steel")
    new_analogies = extract_analogies(user_message)
    for a in new_analogies:
        session["analogies"].append(a)
        session["extracted_properties"][a["property"]] = a["value"]

    # 2. Extract mentioned materials (even without analogy pattern)
    mentioned = find_material_in_text(user_message)
    session["mentioned_materials"].update(mentioned)

    # 3. Extract direct property references (old: just flags)
    direct_props = find_properties_in_text(user_message)

    # 3b. Extract direct statements WITH negation handling and default values
    #     e.g. "it should not conduct electricity" → electrical_conductivity = LOW
    #     e.g. "make it breathable" → breathability = HIGH
    direct_statements = extract_direct_statements(user_message)
    for ds in direct_statements:
        prop = ds["property"]
        # Only set if not already set by a more specific analogy
        if prop not in session["extracted_properties"]:
            session["extracted_properties"][prop] = ds["value"]
            # Build a pseudo-analogy entry for display purposes
            intent_label = "HIGH" if ds["is_high"] else "LOW"
            negate_str = " (negated)" if ds["negated"] else ""
            session["analogies"].append({
                "property": prop,
                "material": None,
                "material_display": f"{intent_label} target{negate_str}",
                "value": ds["value"],
                "is_inverse": not ds["is_high"],
                "trigger": ds["trigger_phrase"],
            })

    # 4. Extract numeric conditions (temp, pH, pressure)
    new_conditions = extract_numeric_conditions(user_message)
    session["conditions"].update(new_conditions)

    # 5. Extract application context
    app_ctx = extract_application_context(user_message)
    if app_ctx:
        session["application"] = app_ctx
        # Apply application defaults to conditions
        if "temperature" in app_ctx and "temperature" not in session["conditions"]:
            session["conditions"]["temperature"] = app_ctx["temperature"]
        if "ph" in app_ctx and "ph" not in session["conditions"]:
            session["conditions"]["ph"] = app_ctx["ph"]

    # ── Detect conflicts ─────────────────────────────────────────────
    session["conflicts"] = _detect_conflicts(session["extracted_properties"])

    # ── Build the response ───────────────────────────────────────────
    has_analogies = len(session["analogies"]) > 0
    has_materials = len(session["mentioned_materials"]) > 0
    has_conditions = len(session["conditions"]) > 0
    has_direct_props = len(direct_props) > 0
    has_direct_statements = len(direct_statements) > 0
    has_any_useful_info = (has_analogies or has_materials or has_conditions
                           or has_direct_props or has_direct_statements)

    if not has_any_useful_info:
        # Didn't understand — help the user
        session["state"] = STATE_COLLECTING
        return _help_response(session, user_message)

    # If we have materials mentioned without analogies, infer their well-known properties
    if has_materials and not has_analogies and not has_direct_statements:
        for mat_key, mat_data in mentioned.items():
            # Add the material's standout properties
            standout = _get_standout_properties(mat_key)
            for prop, val in standout.items():
                if prop not in session["extracted_properties"]:
                    session["extracted_properties"][prop] = val

    # If we got useful info, check if we need more
    missing = _assess_missing_info(session)
    session["missing_info"] = missing

    if session["conflicts"]:
        session["state"] = STATE_CLARIFYING
        return _clarify_conflicts_response(session)

    # Ask for more only when we have very little info
    if len(session["extracted_properties"]) < 1 and not has_conditions:
        session["state"] = STATE_CLARIFYING
        return _ask_for_more_response(session)

    # We have enough — show confirmation
    session["state"] = STATE_CONFIRMING
    return _confirmation_response(session)


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def _greeting_response(session):
    return _build_response(session,
        response=(
            "Welcome to the **Diffusion Dynamics Material Designer**! 🧬\n\n"
            "I can help you design new materials using natural language. "
            "Just describe what you need — you can use **real-world comparisons** like:\n\n"
            "• *\"As strong as steel but as flexible as rubber\"*\n"
            "• *\"Non-corrosive like titanium, lightweight like aluminum\"*\n"
            "• *\"Breathable like linen but tough as kevlar\"*\n\n"
            "You can also mention conditions like temperature, pH, or your application area. "
            "I'll extract the exact scientific parameters and guide you through the design.\n\n"
            "**What kind of material are you looking for?**"
        ),
        suggestions=[
            "As strong as steel but flexible as rubber",
            "A lightweight heat-resistant material for aerospace",
            "Non-corrosive like titanium for acidic environments",
            "Breathable like cotton but strong as kevlar",
            "A catalyst for water splitting at 350K",
        ]
    )


def _help_response(session, user_message):
    return _build_response(session,
        response=(
            f"I understand you're looking for a material, but I need a bit more detail to extract specific properties.\n\n"
            f"**Here are some ways you can describe what you need:**\n\n"
            f"🔹 **Comparisons**: *\"As strong as steel, as light as aluminum\"*\n"
            f"🔹 **Properties**: *\"High strength, low density, corrosion resistant\"*\n"
            f"🔹 **Applications**: *\"For aerospace use at 500K\"* or *\"For biomedical implants\"*\n"
            f"🔹 **Conditions**: *\"pH 2, 350 kelvin, high pressure\"*\n\n"
            f"Think of materials you admire and tell me what specific qualities you want from each!\n\n"
            f"**Could you try describing your ideal material using comparisons?**"
        ),
        suggestions=[
            "Strong as steel, malleable as copper",
            "For underwater use, non-corrosive",
            "Lightweight and heat-resistant like titanium",
            "Flexible as rubber but hard as ceramic",
        ]
    )


def _ask_for_more_response(session):
    """We have some info but want more to give a good result."""
    existing = _format_property_cards(session)
    questions = []
    ep = session["extracted_properties"]

    if "tensile_strength" not in ep and "hardness" not in ep:
        questions.append("How **strong or hard** does this need to be? (e.g., *\"as strong as steel\"* or *\"hard as diamond\"*)")
    if "elongation_pct" not in ep and "malleability" not in ep:
        questions.append("Should it be **rigid or flexible**? (e.g., *\"flexible as rubber\"* or *\"rigid as ceramic\"*)")
    if "corrosion_resistance" not in ep:
        questions.append("Does it need to **resist corrosion**? Will it be in water, acid, or outdoor conditions?")
    if "temperature" not in session["conditions"]:
        questions.append("What **temperature range** will it operate in?")
    if session["application"] is None:
        questions.append("What's the **application**? (e.g., aerospace, biomedical, construction, catalysis)")

    question_text = "\n".join([f"• {q}" for q in questions[:3]])

    return _build_response(session,
        response=(
            f"Great start! Here's what I've picked up so far:\n\n"
            f"{existing}\n\n"
            f"To give you the best candidates, I'd love to know more:\n\n"
            f"{question_text}\n\n"
            f"Or just say **\"generate\"** if you're happy with what we have!"
        ),
        suggestions=[
            "That's enough, generate!",
            "Also make it corrosion resistant",
            "For use at 400K under high pressure",
            "It's for biomedical implants",
        ]
    )


def _clarify_conflicts_response(session):
    """Address conflicting property requirements."""
    conflict_msgs = []
    for c in session["conflicts"]:
        conflict_msgs.append(
            f"⚠️ You asked for **{c['prop1_name']}** ({c['prop1_val']}) "
            f"and **{c['prop2_name']}** ({c['prop2_val']}). "
            f"These are usually opposing — {c['explanation']}"
        )
    conflict_text = "\n".join(conflict_msgs)
    existing = _format_property_cards(session)

    return _build_response(session,
        response=(
            f"I noticed some interesting constraints:\n\n"
            f"{conflict_text}\n\n"
            f"This isn't impossible — advanced composites can achieve both! "
            f"I'll optimize for the best **compromise** between these properties.\n\n"
            f"Current extraction:\n{existing}\n\n"
            f"**Shall I proceed with generation, or would you like to adjust anything?**"
        ),
        suggestions=[
            "Go ahead and generate!",
            "Prioritize strength over flexibility",
            "Prioritize flexibility over strength",
            "Add more requirements",
        ]
    )


def _confirmation_response(session):
    """Show extracted parameters and ask for go-ahead."""
    existing = _format_property_cards(session)

    condition_lines = []
    if "temperature" in session["conditions"]:
        condition_lines.append(f"🌡️ **Temperature**: {session['conditions']['temperature']}K")
    if "ph" in session["conditions"]:
        condition_lines.append(f"🧪 **pH**: {session['conditions']['ph']}")
    if "pressure" in session["conditions"]:
        condition_lines.append(f"🔧 **Pressure**: {session['conditions']['pressure']} atm")
    condition_text = "\n".join(condition_lines) if condition_lines else "*(Standard conditions: 298K, pH 7, 1 atm)*"

    app_text = ""
    if session["application"]:
        app_text = f"\n🏗️ **Application**: {session['application'].get('description', 'General')}\n"

    param_count = len(session["extracted_properties"]) + len(session["conditions"])

    return _build_response(session,
        response=(
            f"Here's my complete analysis of your requirements:\n\n"
            f"{existing}\n\n"
            f"**Environment & Conditions:**\n{condition_text}\n"
            f"{app_text}\n"
            f"📊 **{param_count} parameters locked** — I'll generate 3 optimized candidates.\n\n"
            f"**Ready to synthesize? Just say \"Generate\"!**\n"
            f"Or tell me more to refine the design."
        ),
        ready_to_generate=True,
        suggestions=[
            "Generate!",
            "Also make it lightweight",
            "Change temperature to 500K",
            "Add corrosion resistance like titanium",
        ]
    )


def _build_generation_ready_response(session):
    """User confirmed — prepare final params and signal generation."""
    # Map properties to generation parameters
    final_params = map_properties_to_params(session["extracted_properties"])

    # Overlay explicit conditions
    final_params.update(session["conditions"])

    session["generation_params"] = final_params
    session["state"] = STATE_GENERATING

    return _build_response(session,
        response="🚀 **Initiating molecular synthesis...**\nTranslating your requirements into 3D molecular candidates via EGNN + PINO-FNO pipeline.",
        ready_to_generate=True,
        generation_params=final_params,
        trigger_generation=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# POST-GENERATION: iterative refinement
# ═══════════════════════════════════════════════════════════════════════════════

def process_iteration(session_id, user_message):
    """
    After generation, user wants to refine: "make it stronger", "less flexible", etc.
    Also handles plain statements like "it shouldn't conduct electricity".
    """
    session = get_or_create_session(session_id)
    session["state"] = STATE_ITERATING
    session["raw_messages"].append({"role": "user", "text": user_message})
    msg_lower = user_message.lower().strip()

    # ── Check for generation commands first ───────────────────────────────
    if msg_lower in ("yes", "go", "generate", "yes!", "proceed", "let's go",
                     "do it", "generate!", "yes generate", "start", "run it",
                     "make it", "synthesize", "create", "build it"):
        return _build_generation_ready_response(session)

    # ── Check for reset commands ──────────────────────────────────────────
    if msg_lower in ("reset", "start over", "new", "clear", "start new design"):
        reset_session(session_id)
        return process_message(session_id, "")

    adjustments = []

    # ── Mechanical adjustments ────────────────────────────────────────────
    if any(w in msg_lower for w in ["stronger", "tougher", "more strength"]):
        current_ts = session["extracted_properties"].get("tensile_strength", 300)
        session["extracted_properties"]["tensile_strength"] = current_ts * 1.5
        adjustments.append("⬆️ Increased tensile strength by 50%")

    if any(w in msg_lower for w in ["harder", "more rigid", "stiffer", "hardness"]):
        current_h = session["extracted_properties"].get("hardness", 5.0)
        session["extracted_properties"]["hardness"] = min(10, current_h + 2)
        adjustments.append("⬆️ Increased hardness target")

    if any(w in msg_lower for w in ["compressive", "crush resistant", "squeeze"]):
        session["extracted_properties"]["compressive_strength"] = 500
        adjustments.append("⬆️ Set high compressive strength target")

    if any(w in msg_lower for w in ["impact", "shock", "shatter", "blast"]):
        session["extracted_properties"]["impact_resistance"] = 9
        adjustments.append("⬆️ Set high impact resistance target")

    if any(w in msg_lower for w in ["fatigue", "cyclic", "repeated"]):
        session["extracted_properties"]["fatigue_resistance"] = 9
        adjustments.append("⬆️ Set high fatigue endurance target")

    if any(w in msg_lower for w in ["more flexible", "softer", "bendier", "more elastic", "more stretchy", "malleable"]):
        current_ep = session["extracted_properties"].get("elongation_pct", 25)
        session["extracted_properties"]["elongation_pct"] = current_ep * 1.5
        session["extracted_properties"]["malleability"] = 9
        adjustments.append("⬆️ Increased flexibility/malleability targets")

    if any(w in msg_lower for w in ["lighter", "less dense", "more lightweight", "reduce weight"]):
        session["extracted_properties"]["density"] = session["extracted_properties"].get("density", 5.0) * 0.7
        adjustments.append("⬇️ Reduced target density by 30%")

    if any(w in msg_lower for w in ["heavier", "more dense", "increase weight"]):
        session["extracted_properties"]["density"] = session["extracted_properties"].get("density", 5.0) * 1.4
        adjustments.append("⬆️ Increased target density by 40%")

    # ── Surface / Optical ─────────────────────────────────────────────────
    if any(w in msg_lower for w in ["shiny", "lustrous", "glossy", "more shine"]):
        session["extracted_properties"]["lustre"] = 9
        adjustments.append("⬆️ Set high lustre target")

    if any(w in msg_lower for w in ["transparent", "see through", "clear", "translucent"]):
        session["extracted_properties"]["transparency"] = 9
        adjustments.append("⬆️ Transparency set to 9/10")

    if any(w in msg_lower for w in ["opaque", "non transparent", "blocks light"]):
        session["extracted_properties"]["transparency"] = 0
        adjustments.append("⬇️ Transparency set to 0/10 (opaque)")

    if any(w in msg_lower for w in ["uv resistant", "sun resistant", "uv stable"]):
        session["extracted_properties"]["uv_resistance"] = 9
        adjustments.append("⬆️ UV resistance set to 9/10")

    # ── Corrosion / Chemical ──────────────────────────────────────────────
    if any(w in msg_lower for w in ["more corrosion", "corrosion resistant", "anti-rust", "non-corrosive"]):
        session["extracted_properties"]["corrosion_resistance"] = 9
        adjustments.append("⬆️ Set corrosion resistance to 9/10")

    if any(w in msg_lower for w in ["chemical resistant", "chemically stable", "acid resistant"]):
        session["extracted_properties"]["chemical_stability"] = 9
        adjustments.append("⬆️ Set chemical stability to 9/10")

    # ── Thermal / Temperature ─────────────────────────────────────────────
    if any(w in msg_lower for w in ["hotter", "higher temperature", "more heat resistant", "melting point"]):
        current_t = session["conditions"].get("temperature", 298)
        session["conditions"]["temperature"] = current_t + 100
        session["extracted_properties"]["melting_point"] = 2500
        adjustments.append(f"🌡️ Temperature raised and Melting Point target increased")

    if any(w in msg_lower for w in ["better heat dissipation", "more thermally conductive"]):
        session["extracted_properties"]["thermal_conductivity"] = 400
        adjustments.append("⬆️ Thermal conductivity set to HIGH")

    # ── Electrical ────────────────────────────────────────────────────────
    if any(w in msg_lower for w in ["more conductive", "conduct electricity", "electrically conductive"]):
        session["extracted_properties"]["electrical_conductivity"] = 1e7
        adjustments.append("⬆️ Electrical conductivity set to HIGH")

    if any(w in msg_lower for w in ["insulating", "insulator", "non-conductive"]):
        session["extracted_properties"]["electrical_conductivity"] = 1e-12
        adjustments.append("⬇️ Electrical conductivity set to LOW")

    # ── Physical / Special ────────────────────────────────────────────────
    if any(w in msg_lower for w in ["breathable", "permeable", "porous"]):
        session["extracted_properties"]["breathability"] = 9
        adjustments.append("⬆️ Breathability set to 9/10")

    if any(w in msg_lower for w in ["biocompatible", "body safe", "non-toxic", "surgical"]):
        session["extracted_properties"]["biocompatibility"] = 9
        adjustments.append("⬆️ Biocompatibility set to 9/10")

    if any(w in msg_lower for w in ["magnetic", "magnetizable"]):
        session["extracted_properties"]["magnetic_property"] = 9
        adjustments.append("⬆️ Magnetic property set to 9/10")

    if any(w in msg_lower for w in ["soundproof", "acoustic", "quiet"]):
        session["extracted_properties"]["acoustic_dampening"] = 9
        adjustments.append("⬆️ Acoustic dampening set to 9/10")

    # ── Economic ──────────────────────────────────────────────────────────
    if any(w in msg_lower for w in ["cheaper", "less expensive", "reduce cost", "low cost"]):
        session["extracted_properties"]["cost_index"] = 2
        adjustments.append("⬇️ Cost index reduced (targeting simpler structures)")

    # ── Also parse for new analogies & direct statements ──────────────────
    new_analogies = extract_analogies(user_message)
    for a in new_analogies:
        session["analogies"].append(a)
        session["extracted_properties"][a["property"]] = a["value"]
        adjustments.append(f"🔄 Added analogy: {a['trigger']}")

    direct_stmts = extract_direct_statements(user_message)
    for ds in direct_stmts:
        if ds["property"] not in session["extracted_properties"]:
            session["extracted_properties"][ds["property"]] = ds["value"]
            intent = "HIGH" if ds["is_high"] else "LOW"
            adjustments.append(f"🔄 Set {ds['property'].replace('_',' ')} to {intent}")

    new_conditions = extract_numeric_conditions(user_message)
    for k, v in new_conditions.items():
        session["conditions"][k] = v
        adjustments.append(f"🔄 Set {k} = {v}")

    if not adjustments:
        return _build_response(session,
            response=(
                "I can adjust properties for you! Try saying things like:\n\n"
                "• *\"Make it stronger\"* or *\"More transparent\"*\n"
                "• *\"Increase temperature to 500K\"*\n"
                "• *\"Reduce cost\"*\n\n"
                "What would you like to change?"
            ),
            suggestions=["Make it stronger", "More transparent", "Reduce cost", "Increase heat resistance"]
        )

    adj_text = "\n".join(adjustments)
    existing = _format_property_cards(session)

    return _build_response(session,
        response=(
            f"✅ **Adjustments applied:**\n\n{adj_text}\n\n"
            f"**Updated profile:**\n{existing}\n\n"
            f"Ready for the next synthesis? Say **\"Generate\"**!"
        ),
        ready_to_generate=True,
        suggestions=["Generate!", "Even more transparent", "Reduce weight", "Add corrosion resistance"]
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _format_property_cards(session):
    """Format extracted properties as nice text cards for the AI message."""
    lines = []
    for prop, val in session["extracted_properties"].items():
        icon = PROPERTY_ICONS.get(prop, "📋")
        name = PROPERTY_DISPLAY_NAMES.get(prop, prop.replace("_", " ").title())
        unit = PROPERTY_UNITS.get(prop, "")
        # Find which material this came from
        source = ""
        for a in session["analogies"]:
            if a["property"] == prop:
                source = f" *(from {a['material_display']})*"
                break
        if isinstance(val, float):
            val_str = f"{val:,.2f}" if val > 100 else f"{val:.2f}"
        else:
            val_str = str(val)
        lines.append(f"{icon} **{name}**: {val_str} {unit}{source}")
    return "\n".join(lines) if lines else "*(No properties extracted yet)*"


def _get_standout_properties(mat_key):
    """Get the 2-3 most distinctive properties of a material."""
    mat = MATERIALS_DB.get(mat_key, {})
    if not mat:
        return {}

    # Compare each property to average across all materials
    all_mats = list(MATERIALS_DB.values())
    standouts = {}
    props_to_check = ["tensile_strength", "elongation_pct", "malleability",
                      "hardness", "corrosion_resistance", "breathability",
                      "density", "melting_point", "lustre"]

    for prop in props_to_check:
        vals = [m.get(prop, 0) for m in all_mats if prop in m]
        if not vals:
            continue
        avg = sum(vals) / len(vals)
        mat_val = mat.get(prop, 0)
        # If this material is >1.5x the average, it's a standout
        if avg > 0 and mat_val / avg > 1.5:
            standouts[prop] = mat_val
        # Or if it's <0.3x the average for inverse properties
        elif avg > 0 and mat_val / avg < 0.3:
            standouts[prop] = mat_val

    # Limit to top 3
    return dict(list(standouts.items())[:3])


def _detect_conflicts(extracted_properties):
    """Detect conflicting property requirements."""
    conflicts = []
    ep = extracted_properties

    # Hard + Flexible
    if "hardness" in ep and ep["hardness"] > 7 and "elongation_pct" in ep and ep["elongation_pct"] > 100:
        conflicts.append({
            "prop1_name": "High Hardness",
            "prop1_val": f"{ep['hardness']} Mohs",
            "prop2_name": "High Flexibility",
            "prop2_val": f"{ep['elongation_pct']}%",
            "explanation": "hard materials are typically brittle. I'll aim for a composite-like balance."
        })

    # Strong + Lightweight
    if "tensile_strength" in ep and ep["tensile_strength"] > 500 and "density" in ep and ep["density"] < 2:
        conflicts.append({
            "prop1_name": "Very High Strength",
            "prop1_val": f"{ep['tensile_strength']} MPa",
            "prop2_name": "Very Low Density",
            "prop2_val": f"{ep['density']} g/cm³",
            "explanation": "the strongest materials tend to be dense. Carbon fiber composites achieve this — I'll optimize accordingly."
        })

    # Breathable + Strong
    if "breathability" in ep and ep["breathability"] > 7 and "tensile_strength" in ep and ep["tensile_strength"] > 500:
        conflicts.append({
            "prop1_name": "High Breathability",
            "prop1_val": f"{ep['breathability']}/10",
            "prop2_name": "High Strength",
            "prop2_val": f"{ep['tensile_strength']} MPa",
            "explanation": "porous structures are typically weaker. I'll try for a structured mesh approach."
        })

    return conflicts


def _assess_missing_info(session):
    """Determine what important info is still missing."""
    missing = []
    ep = session["extracted_properties"]
    cond = session["conditions"]

    if "tensile_strength" not in ep and "hardness" not in ep:
        missing.append("strength_or_hardness")
    if "elongation_pct" not in ep and "malleability" not in ep:
        missing.append("flexibility")
    if "temperature" not in cond:
        missing.append("temperature")
    if session["application"] is None:
        missing.append("application")

    return missing


def _build_response(session, response, suggestions=None, ready_to_generate=False,
                    generation_params=None, trigger_generation=False):
    """Build the standard response dict."""
    # Build property cards for frontend display
    property_cards = []
    for prop, val in session["extracted_properties"].items():
        source_material = None
        for a in session["analogies"]:
            if a["property"] == prop:
                source_material = a["material_display"]
                break
        property_cards.append({
            "property": prop,
            "display_name": PROPERTY_DISPLAY_NAMES.get(prop, prop),
            "icon": PROPERTY_ICONS.get(prop, "📋"),
            "value": val,
            "unit": PROPERTY_UNITS.get(prop, ""),
            "source_material": source_material,
        })

    session["raw_messages"].append({"role": "assistant", "text": response})

    return {
        "session_id": session["id"],
        "state": session["state"],
        "response": response,
        "extracted_properties": session["extracted_properties"],
        "analogies": session["analogies"],
        "conditions": session["conditions"],
        "application": session["application"],
        "property_cards": property_cards,
        "conflicts": session["conflicts"],
        "suggestions": suggestions or [],
        "ready_to_generate": ready_to_generate,
        "generation_params": generation_params,
        "trigger_generation": trigger_generation or False,
        "message_count": len(session["raw_messages"]),
    }
