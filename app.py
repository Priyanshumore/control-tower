"""
Integrated Manufacturing Operations Control Tower
Nestle Maggi, Sanand plant (illustrative)

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import io
import hashlib
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from engine import schema as sch
from engine.ui import Theme
from engine.agents import Overrides, apply_decision
from engine.forecasting import METHODS
from engine.bom import indented_bom, low_level_codes
from engine.capacity import available_hours
from engine.kpis import to_gantt
from engine.loader import Dataset, load_files, load_folder, period_label, remap
from engine.agentic import SUGGESTIONS, PlanningSession
from engine.pipeline import (SCENARIO_LIBRARY, Settings, build_scenario,
                             compare_runs, run_pipeline)
from engine.scheduling import DISPATCH_RULES, RULE_HELP

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "data")

# Any folder dropped into data/ becomes a selectable dataset, so a new plant can
# be added without touching the code.
PRETTY = {"nestle_maggi": "Nestle Maggi noodles and sauces",
          "vayu_mobility": "Vayu Mobility electric two wheelers"}


def bundled_datasets() -> Dict[str, str]:
    out = {}
    if os.path.isdir(DATA_ROOT):
        for name in sorted(os.listdir(DATA_ROOT)):
            path = os.path.join(DATA_ROOT, name)
            if os.path.isdir(path) and any(
                    f.lower().endswith((".csv", ".xlsx", ".xls", ".json"))
                    for f in os.listdir(path)):
                out[PRETTY.get(name, name.replace("_", " ").title())] = path
    return out

st.set_page_config(page_title="Operations Control Tower",
                   page_icon="▦", layout="wide", initial_sidebar_state="expanded")
# The workspace intentionally uses one consistent light visual language. Keep
# Streamlit's built-in appearance menu from changing only part of the page.
ui = Theme("light")
ui.inject()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
def _init():
    ss = st.session_state
    ss.setdefault("ds", None)
    ss.setdefault("ds_key", "")
    ss.setdefault("overrides", Overrides())
    ss.setdefault("decisions", [])
    ss.setdefault("scenario_result", None)
    ss.setdefault("scenario_label", "")
    ss.setdefault("cache", {})
    ss.setdefault("chat", [])
    ss.setdefault("pending_prompt", "")
    ss.setdefault("scenario_note", "")


_init()


def get_dataset() -> Optional[Dataset]:
    return st.session_state.ds


def signature(ds: Dataset, st_obj: Settings, ov: Overrides) -> str:
    import hashlib
    import json
    payload = {
        "ds": {k: hashlib.sha256(v.to_json(date_format="iso").encode()).hexdigest()
               for k, v in ds.tables.items()},
        "map": {k: {a: b for a, b in v.items()} for k, v in ds.mappings.items()},
        "set": {k: v for k, v in st_obj.__dict__.items()},
        "ov": {k: v for k, v in ov.__dict__.items()},
    }
    return hashlib.md5(json.dumps(payload, default=str, sort_keys=True).encode()).hexdigest()


def cached_session(ds: Dataset, settings: Settings, ov: Overrides) -> PlanningSession:
    """One planning session per set of inputs.

    The plan is produced by the agent roster rather than by calling the pipeline
    directly, so the capacity and material negotiations really happen on every
    run and the transcript on the Agents tab is the record of this plan.
    """
    key = signature(ds, settings, ov)
    cache = st.session_state.cache
    if key not in cache:
        if len(cache) > 4:
            cache.clear()
        session = PlanningSession(ds, settings, ov)
        session.run()
        cache[key] = session
    return cache[key]


# ---------------------------------------------------------------------------
# Sidebar: data source and planning parameters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ◈  CONTROL TOWER")
    st.caption("PLANNING WORKSPACE")
    st.caption("Light workspace theme")

    available = bundled_datasets()
    choices = list(available) + ["Attach my own files"]
    source = st.radio("Dataset", choices, label_visibility="collapsed")

    if source in available:
        key = "bundled:" + source
        if st.session_state.ds_key != key:
            st.session_state.ds = load_folder(available[source], source)
            st.session_state.ds_key = key
            st.session_state.cache = {}
            st.session_state.chat = []
            st.session_state.scenario_result = None
            st.session_state.overrides = Overrides()
    elif not available and source not in available and source != "Attach my own files":
        st.error("No bundled data found. Run: python scripts/generate_dataset.py")
    else:
        up = st.file_uploader(
            "CSV, Excel or a zip of them",
            type=["csv", "tsv", "txt", "xlsx", "xlsm", "xls", "json", "zip"],
            accept_multiple_files=True)
        if up:
            key = "up:" + ",".join(sorted(f"{f.name}:{hashlib.sha256(f.getvalue()).hexdigest()}" for f in up))
            if st.session_state.ds_key != key:
                st.session_state.ds = load_files([(f.name, f.getvalue()) for f in up], "Attached")
                st.session_state.ds_key = key
                st.session_state.cache = {}
                st.session_state.chat = []
                st.session_state.scenario_result = None
                st.session_state.overrides = Overrides()
        elif not st.session_state.ds_key.startswith("up:") and st.session_state.ds is None:
            st.info("Attach your files, or switch back to the bundled dataset.")

    ds = get_dataset()

    if ds is not None:
        found = len(ds.tables)
        st.caption(f"{found} of {len(sch.SCHEMA)} tables recognised in **{ds.name}**")
        if ds.issues:
            st.warning(f"{len(ds.issues)} data issue(s). See the Data tab.")

        st.divider()
        st.markdown("**Planning parameters**")
        horizon = st.slider("Planning horizon", 4, 24, 12, help="Periods to plan forward")
        service = st.slider("Service level %", 80.0, 99.5, 95.0, 0.5)
        fence = st.slider("Demand time fence", 0, 6, 2,
                          help="Periods. Inside the fence the schedule uses confirmed orders, falling back to forecast when no orders exist.")
        demand_rule = st.selectbox("Demand outside the fence",
                                   ["max", "forecast", "orders", "sum"], index=0,
                                   format_func=lambda v: {"max": "Greater of forecast and orders",
                                                          "forecast": "Forecast only",
                                                          "orders": "Orders only",
                                                          "sum": "Forecast plus orders"}[v])
        level = st.checkbox("Level against capacity", value=True,
                            help="Pull volume earlier to clear overloaded periods")

        st.markdown("**Shop floor**")
        rule = st.selectbox("Dispatching rule", DISPATCH_RULES,
                            index=DISPATCH_RULES.index("EDD"),
                            help="\n".join(f"{k}: {v}" for k, v in RULE_HELP.items()))
        window = st.slider("Release window", 1, 4, 1,
                           help="How many periods of orders to release to the floor")
        max_lot = st.number_input("Split orders above (units)",
                                  min_value=0.0, value=400.0, step=50.0,
                                  help="Larger orders are split into sub lots so operations "
                                       "can overlap. Set 0 to keep every order whole.")
        seq_setup = st.checkbox("Skip repeat setups", value=True,
                                help="No setup when a machine runs the same product again")

        with st.expander("Forecasting settings", expanded=True):
            manual_forecast = st.toggle("Manual forecast parameters", value=True)
            forecast_method = st.selectbox("Forecast method", ["Auto"] + METHODS)
            alpha = st.slider("Alpha · level", 0.01, 1.0, 0.2, 0.01, disabled=not manual_forecast)
            beta = st.slider("Beta · trend", 0.01, 1.0, 0.1, 0.01, disabled=not manual_forecast)
            gamma = st.slider("Gamma · seasonality", 0.01, 1.0, 0.2, 0.01, disabled=not manual_forecast)
            ma_n = st.number_input("Moving average window", 1, 52, 3, disabled=not manual_forecast)
            weights_text = st.text_input("Weights · newest first", "0.5, 0.3, 0.2")
            try:
                weights = [float(w.strip()) for w in weights_text.split(",")]
                if not weights or any(not np.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
                    raise ValueError()
            except ValueError:
                st.error("Enter comma-separated, nonnegative weights with a positive total.")
                st.stop()
            st.caption("Weights are normalized. Alpha affects smoothing methods; beta affects trend; gamma affects Holt-Winters. Auto selects the best available method.")
            season = st.number_input("Seasonal cycle (periods)", 2, 52, 13)
            holdout = st.number_input("Hold out periods", 2, 20, 8,
                                      help="Periods kept back to test the forecast on")
            criterion = st.selectbox("Selection criterion", ["MAPE", "MAD", "RMSE", "MSE"])
            mape_threshold = st.number_input("Alert above MAPE %", 1.0, 100.0, 20.0)

        with st.expander("Inventory and release policies"):
            use_mps = st.checkbox("Use MPS template quantities", False,
                                  help="Use entered quantities exactly, including zeros. Unlisted periods are calculated automatically. Capacity levelling is disabled while using the template.")
            poq = st.number_input("POQ coverage (periods)", 1, 24, 2)
            excess = st.number_input("Excess stock threshold (weeks)", 1.0, 52.0, 6.0)
            include_open = st.checkbox("Include open production orders", True)
            lot_policy = st.selectbox("Master schedule lot policy", ["Use item master", "LFL", "FOQ", "EOQ", "POQ"])

        settings = Settings(
            use_mps_input=use_mps,
            forecast_options=dict(manual=manual_forecast, alpha=alpha, beta=beta,
                                  gamma=gamma, ma_n=int(ma_n), wma_weights=weights),
            method_override={it: forecast_method for it in ds.get("demand_history")["item"].unique()},
            poq_periods=int(poq), excess_weeks=float(excess), include_open_orders=include_open,
            lot_rule_override=({it: lot_policy for it in ds.get("inventory_master")["item"]}
                               if lot_policy != "Use item master" else {}),
            horizon=int(horizon), season_length=int(season), holdout=int(holdout),
            accuracy_criterion=criterion, mape_threshold=float(mape_threshold),
            service_level=float(service), demand_time_fence=int(fence),
            demand_rule=demand_rule, level_capacity=bool(level),
            release_window=int(window), max_lot=float(max_lot), dispatch_rule=rule,
            sequence_dependent_setup=bool(seq_setup),
        )

        st.divider()
        ov: Overrides = st.session_state.overrides
        if not ov.is_empty():
            st.markdown("**Active plan changes**")
            for line in ov.summary()[:6]:
                st.caption("• " + line)
            if st.button("Clear all changes"):
                st.session_state.overrides = Overrides()
                st.session_state.decisions = []
                st.rerun()
    else:
        settings = Settings()

if ds is None:
    st.title("Integrated Manufacturing Operations Control Tower")
    st.info("Choose a dataset in the control panel on the left to begin.")
    st.stop()

if not ds.has("demand_history") or not ds.has("inventory_master"):
    st.title("Integrated Manufacturing Operations Control Tower")
    st.error("This dataset needs at least a demand history and an inventory master. "
             "Detected: " + (", ".join(sch.SCHEMA[t].label for t in ds.tables) or "nothing"))
    for i in ds.issues:
        st.caption("• " + i)
    st.stop()

plan_key = signature(ds, settings, st.session_state.overrides)
if st.session_state.get("active_plan_key") != plan_key:
    st.session_state.scenario_result = None
    st.session_state.chat = []
    st.session_state.active_plan_key = plan_key
session = cached_session(ds, settings, st.session_state.overrides)
res = session.plan
S = res["kpi_summary"]
p0 = res["first_period"]
periods = res["periods"]


def plabel(p) -> str:
    return period_label(ds, p)


# ---------------------------------------------------------------------------
# Header ribbon
# ---------------------------------------------------------------------------
high = S["exceptions_high"]
over = S["overloaded_periods"]
otd = S["otd_pct"]

if high == 0 and over == 0 and (otd == 100 or np.isnan(otd)):
    verdict, vsub, vcol = "Plan is executable", "No high severity exceptions this run", ui.GREEN
elif high > 0 and over > 0:
    verdict, vsub, vcol = "Plan needs intervention", f"{high} high severity exceptions and {over} overloaded work centre periods", ui.RED
elif high > 0:
    verdict, vsub, vcol = "Plan is at risk", f"{high} high severity exceptions to clear", ui.AMBER
else:
    verdict, vsub, vcol = "Plan is tight", f"{over} overloaded work centre periods", ui.AMBER

# styled on the element itself rather than through an h1 selector, so it cannot
# resize any other heading on the page
st.markdown(
    f"<div style='font-size:2.1rem;font-weight:650;line-height:1.35;"
    f"margin:0.2rem 0 0.1rem 0;color:{ui.TEXT}'>From demand to delivery.</div>"
    f"<div style='font-size:0.85rem;line-height:1.5;color:{ui.MUTED};"
    f"margin-bottom:0.5rem'>{ui.esc(ds.name)} &middot; periods {p0} to "
    f"{p0 + len(periods) - 1}</div>",
    unsafe_allow_html=True)

ui.ribbon(verdict, vsub, vcol, [
    (f"{S['forecast_mape']:.1f}%" if not np.isnan(S["forecast_mape"]) else "n/a",
     "Forecast error (MAPE)", ui.TEXT),
    (f"{S['mps_units']:,.0f}", "Units in the master schedule", ui.TEXT),
    (f"{S['peak_utilisation']:.0f}%", "Peak work centre load",
     ui.RED if S["peak_utilisation"] > 100 else (ui.ACCENT if S["peak_utilisation"] > 90 else ui.GREEN)),
    (f"{otd:.0f}%" if not np.isnan(otd) else "n/a", "On time delivery",
     ui.GREEN if otd >= 95 else (ui.ACCENT if otd >= 85 else ui.RED)),
    (f"{S['exceptions_total']}", "Open exceptions",
     ui.RED if high else (ui.ACCENT if S["exceptions_total"] else ui.GREEN)),
])

TAB_NAMES = ["Control tower", "Agents", "1-2 Data", "3 Forecast", "4 Inventory", "5 MPS",
             "6 BOM", "7 MRP", "8 Release", "9 Capacity", "10 Schedule", "11 KPIs",
             "12 Exceptions", "13 Scenarios", "Reports"]
TABS = st.tabs(TAB_NAMES)

# ---------------------------------------------------------------------------
# Control tower overview
# ---------------------------------------------------------------------------
with TABS[0]:
    c1, c2 = st.columns([2, 1], gap="medium")
    with c1:
        st.markdown("##### Demand into supply, period by period")
        mps = res["mps"]
        if len(mps):
            agg = mps.groupby("period", as_index=False).agg(
                demand=("gross_requirement", "sum"), schedule=("mps_qty", "sum"),
                closing=("pab", "sum"))
            fig = go.Figure()
            fig.add_bar(x=agg["period"], y=agg["demand"], name="Gross requirement",
                        marker_color=ui.BLUE, opacity=0.85)
            fig.add_bar(x=agg["period"], y=agg["schedule"], name="Master schedule",
                        marker_color=ui.ACCENT, opacity=0.9)
            fig.add_scatter(x=agg["period"], y=agg["closing"], name="Closing stock",
                            mode="lines+markers", line=dict(color=ui.GREEN, width=2))
            fig.update_layout(barmode="group", xaxis_title="Period", yaxis_title="Units")
            st.plotly_chart(ui.theme(fig, 330, assumed_width=600), key="ct_demand")

        st.markdown("##### Work centre load against capacity")
        load = res["capacity_load"]
        if len(load):
            piv = load.pivot(index="period", columns="work_centre", values="utilisation_pct").fillna(0)
            fig = px.imshow(piv.T, color_continuous_scale=[[0, ui.PANEL_2], [0.5, "#8AC4B4"],
                                                           [0.8, "#E6BE74"], [1.0, ui.RED]],
                            aspect="auto", labels=dict(x="Period", y="", color="Load %"),
                            zmin=0, zmax=max(120, float(piv.to_numpy().max())))
            fig.update_traces(hovertemplate="%{y} period %{x}: %{z:.0f}%<extra></extra>")
            st.plotly_chart(ui.theme(fig, ui.fit_height(len(piv.columns), per=26, base=140),
                                     legend=False), key="ct_heat")
            ui.note("Colour deepens as load increases; red indicates the highest utilisation. "
                    "The work centre that stays hottest across the horizon is the constraint.")
    with c2:
        st.markdown("##### This run")
        st.metric("Items planned", f"{S['items_planned']}")
        st.metric("Planned orders raised", f"{S['planned_orders']}")
        st.metric("Schedule value", f"INR {S['mps_value'] / 1e7:,.2f} Cr")
        st.metric("Inventory on hand", f"INR {S['inventory_value'] / 1e7:,.2f} Cr")
        st.metric("Jobs on the floor", f"{S['jobs']}")
        st.metric("Makespan", f"{S['makespan_h']:,.0f} h" if not np.isnan(S["makespan_h"]) else "n/a")

    st.markdown("##### What needs attention first")
    exc = res["exceptions"]
    if len(exc):
        top = exc.head(6)
        for _, r in top.iterrows():
            col = ui.STATUS_COLOUR.get(r["severity"], ui.MUTED)
            money = f" &middot; exposure INR {r['exposure_value']:,.0f}" if r["exposure_value"] > 0 else ""
            st.markdown(
                f"<div class='panel'>{ui.pill(r['severity'], col)} "
                f"<b>{ui.esc(r['item'])}</b> &middot; {ui.esc(r['type'])} "
                f"<span style='color:{ui.MUTED}'>&middot; {ui.esc(r['agent'])}{money}</span><br>"
                f"<span class='note'>{ui.esc(r['detail'])}<br>"
                f"<b style='color:{ui.ACCENT}'>Action:</b> "
                f"{ui.esc(r['recommended_action'])}</span></div>",
                unsafe_allow_html=True)
    else:
        st.success("No exceptions raised on this run.")

# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
KIND_COLOUR = {"request": ui.BLUE, "response": ui.GREEN, "inform": ui.MUTED,
               "escalate": ui.RED, "broadcast": ui.VIOLET}


def render_reply(reply, transcript_expanded: bool = False) -> None:
    st.markdown(f"<div class='reply'><div class='hl'>{ui.esc(reply.headline)}</div></div>", unsafe_allow_html=True)
    for who, what in reply.sections:
        if what:
            st.caption(f"{who}: {what}")
    for name, df in reply.tables.items():
        if not isinstance(df, pd.DataFrame):
            continue
        st.markdown(f"**{name} · {len(df)} results**")
        if df.empty:
            st.success("No matching items found in the current plan.")
            continue
        if name == "Stockout items":
            for _, row in df.iterrows():
                st.markdown(
                    f"<div class='panel' style='border-left:4px solid {ui.RED}'>"
                    f"{ui.pill('PROJECTED STOCKOUT', ui.RED)} <b>{ui.esc(row['item'])}</b><br>"
                    f"{ui.esc(row['description'])}<br>"
                    f"<b>First stockout: {ui.esc(plabel(row['period']))}</b> · "
                    f"Shortfall: <b>{row['shortfall']:,.0f} {ui.esc(row['uom'])}</b><br>"
                    f"<span class='note'>{ui.esc(row['next_action'])}</span></div>", unsafe_allow_html=True)
        else:
            status_col = next((c for c in ("severity", "status") if c in df.columns), None)
            st.dataframe(ui.status_frame(df, status_col) if status_col else df,
                         hide_index=True, height=min(480, 40 + 35 * len(df)))
    if reply.focus_tab:
        st.info(f"Explore the full plan in the {reply.focus_tab} tab.")


def render_transcript(messages, height_note: str = "") -> None:
    if not messages:
        st.caption("No messages.")
        return
    rows = []
    for m in messages:
        col = KIND_COLOUR.get(m.kind, ui.MUTED)
        took = f" &middot; {m.elapsed_ms:.0f} ms" if m.elapsed_ms > 1 else ""
        rows.append(
            f"<div class='msg' style='border-left-color:{col}'>"
            f"<span class='who' style='color:{col}'>{ui.esc(m.sender)} &rarr; "
            f"{ui.esc(m.recipient)}</span> "
            f"<span class='tp'>{ui.esc(m.topic)}{took}</span><br>"
            f"<span class='bd'>{ui.esc(m.text)}</span></div>")
    st.markdown("".join(rows), unsafe_allow_html=True)


with TABS[1]:
    ui.step("Multi agent layer  \u00b7  a coordinator delegating to ten specialists")
    st.markdown("##### Ask the control tower")
    with st.form("agent_prompt", clear_on_submit=True):
        typed = st.text_input("Your question", value="",
                              placeholder="Why is anything running late?",
                              label_visibility="collapsed")
        sent = st.form_submit_button("Ask", type="primary")
    # Streamlit renders the submit label inside a generated wrapper. Keep the
    # label white even when its component stylesheet has a more specific rule.
    st.markdown("""<style>
    html body div.stFormSubmitButton > button,
    html body div.stFormSubmitButton > button * {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }
    </style>""", unsafe_allow_html=True)

    st.markdown("<div class='askrow'>Or try one of these</div>", unsafe_allow_html=True)
    sug_cols = st.columns(5)
    for i, sug in enumerate(SUGGESTIONS[:10]):
        if sug_cols[i % 5].button(sug, key=f"sug_{i}"):
            st.session_state.pending_prompt = sug
            st.rerun()

    prompt = st.session_state.pending_prompt or (typed if sent else "")
    st.session_state.pending_prompt = ""
    if prompt:
        reply = session.ask(prompt)
        st.session_state.chat.insert(0, {
            "prompt": prompt,
            "reply": reply,
            "messages": session.bus.transcript()[reply.from_seq:],
        })
        st.session_state.chat = st.session_state.chat[:6]

    if st.session_state.chat:
        latest = st.session_state.chat[0]
        with st.expander("Latest answer", expanded=True):
            st.caption("YOUR QUESTION")
            st.write(latest["prompt"])
            render_reply(latest["reply"])
        with st.expander("Agent reasoning and activity", expanded=False):
            render_transcript(latest["messages"])
        if len(st.session_state.chat) > 1:
            with st.expander(f"Previous answers ({len(st.session_state.chat) - 1})"):
                for turn in st.session_state.chat[1:]:
                    st.write(f"You asked: {turn['prompt']}")
                    render_reply(turn["reply"])
    else:
        st.info("Ask a question to see the answer here. The latest result stays directly below the chat and can be collapsed.")

    st.markdown("##### The roster")
    ui.note("Every answer below is assembled from what the specialists reported. The "
            "coordinator does no planning arithmetic of its own, and the sub agents talk to "
            "each other directly: the master schedule has to ask the capacity agent for hours "
            "before it can build ahead, release asks the material agent whether components "
            "exist, and the material agent asks the supplier agent before promising an expedite.")

    roster = session.roster_frame()
    agents = {a.name: a for a in [session.coordinator] + session.sub_agents}
    for i, (_, r) in enumerate(roster.iterrows()):
        if i % 3 == 0:
            card_columns = st.columns(3)
        with card_columns[i % 3], st.container(border=True):
            if st.button(f"{r['Agent']} →", key=f"agent_card_{r['Agent']}",
                         help=r["Role"], width="stretch"):
                st.session_state.selected_agent = r["Agent"]
            st.caption(r["Role"])
            st.caption(f"{r['Owns']} · {r['Questions answered']} questions answered")
    selected_agent = st.session_state.get("selected_agent")
    if selected_agent in agents:
        agent = agents[selected_agent]
        with st.container(border=True):
            st.markdown(f"### {agent.name} agent")
            st.write(agent.role)
            st.caption(f"Ownership: {agent.owns}")
            st.markdown("**What this agent can do**")
            st.write(" · ".join(topic.replace(".", " / ").replace("_", " ") for topic in agent.skills))
            output_keys = {
                "Coordinator": ["kpi_summary"], "Data": [], "Forecast": ["forecast"],
                "Inventory": ["inventory_params"], "Capacity": ["capacity_load"],
                "MPS": ["mps"], "Supplier": [], "Material": ["mrp"],
                "Scheduling": ["schedule"], "Exceptions": ["exceptions"], "Scenario": []}
            st.markdown("**Current plan output**")
            if agent.name == "Data":
                st.dataframe(pd.DataFrame([{"Table": k, "Rows": len(v)} for k, v in ds.tables.items()]), hide_index=True)
            elif agent.name == "Supplier":
                st.dataframe(ds.get("supplier_leadtime"), hide_index=True)
            elif agent.name == "Scenario":
                st.info("Use Scenarios to choose a disruption, run it, and compare the resulting plan.")
            else:
                for output_key in output_keys.get(agent.name, []):
                    value = res.get(output_key)
                    if isinstance(value, pd.DataFrame):
                        st.dataframe(value, hide_index=True, height=240)
                    elif output_key == "kpi_summary":
                        st.dataframe(pd.DataFrame(value.items(), columns=["Metric", "Value"]), hide_index=True)
            st.markdown("**Activity on this plan**")
            activity = [m for m in session.bus.log if agent.name in (m.sender, m.recipient)]
            render_transcript(activity[-30:])
            st.caption("The latest 30 exchanges are shown. Agents with no messages have not been consulted yet.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Messages this run", f"{len(session.bus.log)}")
    c2.metric("Capacity grants", f"{session.capacity.grants}",
              help="Hours the capacity agent released to the master schedule during levelling")
    c3.metric("Hours negotiated", f"{session.capacity.hours_granted:,.1f} h")
    c4.metric("Component checks", f"{session.material.checks}",
              help=f"{session.material.blocked} came back short")

    with st.expander(f"Transcript of this planning run ({len(session.bus.log)} messages)"):
        ui.note("This is the conversation that produced the plan on the other tabs, not a "
                "description of it written afterwards. Repeated identical exchanges are "
                "summarised after the third.")
        render_transcript(session.bus.transcript(80))

    st.markdown("##### Who answers what")
    skills = pd.DataFrame([
        {"Agent": a.name, "Owns": a.owns, "Role": a.role,
         "Topics": ", ".join(t.split(".", 1)[-1] for t in a.skills)}
        for a in session.sub_agents])
    st.dataframe(skills, hide_index=True)


# ---------------------------------------------------------------------------
# 1-2 Data and mapping
# ---------------------------------------------------------------------------
with TABS[2]:
    ui.step("Steps 1 and 2  ·  understand the assignment inputs and prepare the dataset")
    st.markdown("##### Every table the control tower can read")
    ui.note("The engine works on canonical column names. Any dataset can be attached: files are "
            "matched to tables by name and by content, then columns are matched to canonical "
            "fields by name, synonym and token. Anything mismatched can be corrected below and "
            "the whole plan recomputes.")

    rows = []
    for tname in sch.TABLE_ORDER:
        tbl = sch.SCHEMA[tname]
        present = ds.has(tname)
        rows.append({
            "Table": tbl.label,
            "Required": "Yes" if tbl.required else "Optional",
            "Detected": "Yes" if present else "No",
            "Source file": ds.sources.get(tname, ""),
            "Rows": len(ds.tables[tname]) if present else 0,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    if ds.unmatched:
        st.warning("Not recognised: " + ", ".join(ds.unmatched))
    if ds.issues:
        with st.expander(f"{len(ds.issues)} data quality note(s)", expanded=True):
            for i in ds.issues:
                st.caption("• " + i)

    st.markdown("##### Check or correct the column mapping")
    tname = st.selectbox("Table", [t for t in sch.TABLE_ORDER if ds.has(t)],
                         format_func=lambda t: sch.SCHEMA[t].label)
    if tname:
        raw = ds.raw[tname]
        cur = ds.mappings[tname]
        cols = ["(not mapped)"] + list(raw.columns)
        new_map, changed = {}, False
        grid = st.columns(3)
        for i, f in enumerate(sch.SCHEMA[tname].fields):
            with grid[i % 3]:
                idx = cols.index(cur[f.name]) if cur.get(f.name) in cols else 0
                pick = st.selectbox(
                    f"{f.name}{' *' if f.required else ''}", cols, index=idx,
                    key=f"map_{tname}_{f.name}")
                new_map[f.name] = None if pick == "(not mapped)" else pick
                if new_map[f.name] != cur.get(f.name):
                    changed = True
        if changed and st.button("Apply mapping and replan"):
            remap(ds, tname, new_map)
            st.session_state.cache = {}
            st.rerun()

        with st.expander("Edit planning data", expanded=True):
            st.caption("Edit any source value, including demand, stock, BOM quantities, supplier lead times, routing, capacity, receipts and orders. Apply to recompute every module. Changes stay in this session; download a copy to keep them.")
            with st.form(f"edit_source_{st.session_state.ds_key}_{tname}"):
                edited = st.data_editor(raw, num_rows="dynamic", hide_index=True, width="stretch",
                                        key=f"source_editor_{st.session_state.ds_key}_{tname}")
                apply_data = st.form_submit_button("Apply data and replan", type="primary")
            if apply_data:
                import copy
                candidate = copy.deepcopy(ds)
                candidate.raw[tname] = edited.copy()
                try:
                    remap(candidate, tname, cur)
                    problems = sch.validate(tname, candidate.tables[tname])
                    if not candidate.has("demand_history") or not candidate.has("inventory_master"):
                        problems.append("Demand history and inventory master cannot be empty.")
                    if problems:
                        st.error("Please correct the data: " + "; ".join(problems))
                    else:
                        run_pipeline(candidate, settings, st.session_state.overrides)
                        st.session_state.ds = candidate
                        st.session_state.cache = {}
                        st.rerun()
                except (ValueError, KeyError, TypeError, IndexError, ZeroDivisionError) as exc:
                    st.error(f"The edits could not produce a valid plan: {exc}")
            st.download_button("Download this source table", raw.to_csv(index=False),
                               file_name=f"{tname}.csv", mime="text/csv")

        st.markdown("###### Source file as delivered")
        st.dataframe(raw.head(12), hide_index=True)
        st.markdown("###### After mapping")
        st.dataframe(ds.tables[tname].head(12), hide_index=True)

# ---------------------------------------------------------------------------
# 3 Forecasting
# ---------------------------------------------------------------------------
with TABS[3]:
    ui.step("Step 3  ·  demand forecasting engine")
    f = res["forecast_run"]
    sel, acc, fit = f["selected"], f["accuracy"], f["fit"]

    c1, c2 = st.columns([1.35, 1], gap="medium")
    with c1:
        st.markdown("##### Method chosen for each product")
        st.dataframe(sel.rename(columns={
            "item": "Item", "method": "Method", "params": "Parameters",
            "MAPE": "MAPE %", "MAD": "MAD", "RMSE": "RMSE", "Bias": "Bias",
            "Tracking_Signal": "Tracking signal", "selection": "Basis"}).round(2),
            hide_index=True)
        ui.note("Manual mode uses your smoothing constants and window. Automatic mode searches parameter candidates. Scores measure forecasts against a held-out history block; the selected model is then refitted on all history. Constant future values are expected for level-only methods. Unavailable methods fall back to the best valid method.")
    with c2:
        st.markdown("##### Error by method")
        item_sel = st.selectbox("Product", sorted(fit["item"].unique()), key="fc_item")
        a = acc[acc["item"] == item_sel].sort_values("MAPE")
        fig = px.bar(a, x="MAPE", y="method", orientation="h", text="MAPE")
        fig.update_traces(marker_color=[ui.ACCENT if m == sel.loc[sel["item"] == item_sel, "method"].iloc[0]
                                        else ui.BLUE for m in a["method"]],
                          texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
        fig.update_layout(xaxis_title="Hold out MAPE %", yaxis_title="")
        st.plotly_chart(ui.theme(fig, ui.fit_height(len(a), per=34, base=130),
                                 legend=False, right_pad=64), key="fc_bar")

    st.markdown("##### Actual against forecast")
    g = fit[fit["item"] == item_sel].sort_values("period")
    fig = go.Figure()
    fig.add_scatter(x=g["period"], y=g["actual"], name="Actual demand",
                    mode="lines+markers", line=dict(color=ui.TEXT, width=2),
                    marker=dict(size=5))
    hist = g[g["type"] == "History"]
    fut = g[g["type"] == "Forecast"]
    fig.add_scatter(x=hist["period"], y=hist["fitted"], name="Fitted",
                    mode="lines", line=dict(color=ui.BLUE, width=1.6, dash="dot"))
    fig.add_scatter(x=fut["period"], y=fut["fitted"], name="Forecast",
                    mode="lines+markers", line=dict(color=ui.ACCENT, width=2.4))
    fig.add_vline(x=p0 - 0.5, line=dict(color=ui.LINE, width=1, dash="dash"))
    fig.update_layout(xaxis_title="Period", yaxis_title="Units")
    st.plotly_chart(ui.theme(fig, 380, assumed_width=860), key="fc_line")

    with st.expander("Full accuracy table and the forecast values"):
        st.dataframe(acc.round(3), hide_index=True)
        st.dataframe(res["forecast"].pivot(index="period", columns="item", values="forecast"))

# ---------------------------------------------------------------------------
# 4 Inventory
# ---------------------------------------------------------------------------
with TABS[4]:
    ui.step("Step 4  ·  inventory planning")
    par = res["inventory_params"]
    ui.formula("Safety stock = z x sigma x sqrt(lead time)&nbsp;&nbsp;|&nbsp;&nbsp;"
               "Reorder point = average demand x lead time + safety stock&nbsp;&nbsp;|&nbsp;&nbsp;"
               "EOQ = sqrt(2 x annual demand x ordering cost / holding cost per unit)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Items", f"{len(par)}")
    c2.metric("Inventory value", f"INR {par['inventory_value'].sum() / 1e7:,.2f} Cr")
    c3.metric("Below safety stock", f"{int((par['on_hand'] < par['safety_stock_master']).sum())}")
    c4.metric("A class items", f"{int((par['abc_class'] == 'A').sum())}")

    show = par[["item", "description", "item_type", "abc_class", "on_hand", "uom",
                "avg_demand_per_period", "sd_demand", "lead_time", "safety_stock_master",
                "safety_stock_statistical", "reorder_point", "eoq", "lot_rule",
                "weeks_of_supply", "inventory_value"]].copy()
    show.columns = ["Item", "Description", "Type", "ABC", "On hand", "UoM", "Avg demand",
                    "Std dev", "Lead time", "Safety stock", "Statistical SS", "Reorder point",
                    "EOQ", "Lot rule", "Weeks of cover", "Value INR"]
    st.dataframe(show.round(2), hide_index=True, height=320)
    ui.note("Statistical safety stock is what the demand variability and lead time would justify at "
            "the chosen service level. Where it sits far below the master safety stock, the plant "
            "is holding more buffer than the variability calls for.")

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("##### Cover against the buffer")
        d = par[par["avg_demand_per_period"] > 0].copy()
        fig = px.scatter(d, x="weeks_of_supply", y="item", color="abc_class",
                         size="inventory_value", size_max=26,
                         color_discrete_map={"A": ui.ACCENT, "B": ui.BLUE, "C": ui.MUTED})
        fig.update_layout(xaxis_title="Weeks of cover on hand", yaxis_title="")
        st.plotly_chart(ui.theme(fig, ui.fit_height(d["item"].nunique())), key="inv_scatter")
    with c2:
        st.markdown("##### Finished goods if nothing is produced")
        proj = res["inventory_projection"]
        if len(proj):
            fig = go.Figure()
            for i, it in enumerate(sorted(proj["item"].unique())):
                gg = proj[proj["item"] == it]
                fig.add_scatter(x=gg["period"], y=gg["closing"], name=it, mode="lines+markers",
                                line=dict(width=2, color=ui.SERIES[i % len(ui.SERIES)]))
            fig.add_hline(y=0, line=dict(color=ui.RED, width=1, dash="dash"))
            fig.update_layout(xaxis_title="Period", yaxis_title="Closing stock")
            st.plotly_chart(ui.theme(fig, ui.fit_height(
                res["inventory_params"]["item"].nunique())), key="inv_proj")
            ui.note("This is the gap the master schedule has to close. Crossing zero is a stockout.")

    with st.expander("Period by period projection and the exceptions raised"):
        st.dataframe(ui.status_frame(res["inventory_projection"]), hide_index=True)
        st.dataframe(res["inventory_exceptions"], hide_index=True)

# ---------------------------------------------------------------------------
# 5 MPS
# ---------------------------------------------------------------------------
with TABS[5]:
    ui.step("Step 5  ·  master production schedule")
    ui.formula("Projected available balance = previous balance + master schedule "
               "+ scheduled receipts - gross requirement")
    mps = res["mps"]
    feas = res["mps_feasibility"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scheduled units", f"{mps['mps_qty'].sum():,.0f}")
    c2.metric("Lots planned", f"{int((mps['mps_qty'] > 0).sum())}")
    c3.metric("Peak load", f"{S['peak_utilisation']:.0f}%")
    c4.metric("Schedule feasible", "Yes" if feas["feasible"] else "No")

    item = st.selectbox("Product", sorted(mps["item"].unique()), key="mps_item")
    g = mps[mps["item"] == item]
    show = g[["period", "forecast", "customer_orders", "gross_requirement", "demand_basis",
              "opening_pab", "scheduled_receipts", "mps_qty", "pab", "safety_stock",
              "lot_rule", "status"]].copy()
    show.columns = ["Period", "Forecast", "Customer orders", "Gross requirement", "Demand basis",
                    "Opening", "Receipts", "Master schedule", "Closing (PAB)", "Safety stock",
                    "Lot rule", "Status"]
    st.dataframe(ui.status_frame(show.round(1), "Status"), hide_index=True)

    c1, c2 = st.columns([1.4, 1], gap="medium")
    with c1:
        fig = go.Figure()
        fig.add_bar(x=g["period"], y=g["gross_requirement"], name="Gross requirement",
                    marker_color=ui.BLUE, opacity=0.8)
        fig.add_bar(x=g["period"], y=g["mps_qty"], name="Master schedule", marker_color=ui.ACCENT)
        fig.add_scatter(x=g["period"], y=g["pab"], name="Closing balance", mode="lines+markers",
                        line=dict(color=ui.GREEN, width=2.2))
        fig.add_scatter(x=g["period"], y=g["safety_stock"], name="Safety stock", mode="lines",
                        line=dict(color=ui.RED, width=1.4, dash="dash"))
        fig.update_layout(barmode="group", xaxis_title="Period", yaxis_title="Units")
        st.plotly_chart(ui.theme(fig, 350), key="mps_chart")
    with c2:
        st.markdown("##### Available to promise")
        atp = res["mps_atp"]
        a = atp[atp["item"] == item]
        if len(a):
            st.dataframe(a.rename(columns={"period": "Period", "supply": "Supply",
                                           "committed_orders": "Committed", "atp": "ATP"})
                         [["Period", "Supply", "Committed", "ATP"]].round(0),
                         hide_index=True, height=300)
            ui.note("What sales can still promise from each lot before the next one lands.")

    if len(res["mps_moves"]):
        st.markdown("##### Capacity levelling that was applied")
        ui.note("Volume is only ever pulled earlier, never pushed later, so customer due dates "
                "are protected. Every move is checked against all work centres on the routing.")
        st.dataframe(res["mps_moves"], hide_index=True)

    if len(feas["issues"]):
        st.markdown("##### Feasibility check")
        st.dataframe(ui.status_frame(feas["issues"], "severity"), hide_index=True)
    else:
        st.success("The master schedule passes the balance, safety stock and capacity checks.")

# ---------------------------------------------------------------------------
# 6 BOM
# ---------------------------------------------------------------------------
with TABS[6]:
    ui.step("Step 6  ·  bill of material explosion")
    bom = ds.get("bom")
    if not len(bom):
        st.info("No bill of material in this dataset, so the plan runs at finished goods level only.")
    else:
        llc = low_level_codes(bom, list(ds.get("inventory_master")["item"]))
        c1, c2, c3 = st.columns(3)
        c1.metric("Parent to component links", f"{len(bom)}")
        c2.metric("Structure depth", f"{max(llc.values()) + 1} levels")
        c3.metric("Shared components",
                  f"{int((bom.groupby('component')['parent'].nunique() > 1).sum())}")

        c1, c2 = st.columns([1, 1.25], gap="medium")
        with c1:
            parent = st.selectbox("Explode", sorted(bom["parent"].unique()), key="bom_parent")
            qty = st.number_input("For how many parents", 1.0, 1e9, 1000.0, step=100.0)
            ind = pd.DataFrame(indented_bom(bom, parent, qty))
            if len(ind):
                st.dataframe(ind[["indent", "level", "qty_per", "scrap_pct", "extended_qty", "uom"]]
                             .rename(columns={"indent": "Structure", "level": "Level",
                                              "qty_per": "Qty per parent", "scrap_pct": "Scrap %",
                                              "extended_qty": "Extended qty", "uom": "UoM"}),
                             hide_index=True, height=330)
                ui.note("Extended quantity carries the scrap allowance at every level, so the number "
                        "at the bottom is what has to be bought, not what ends up in the pack.")
        with c2:
            st.markdown("##### Gross requirements from the master schedule")
            ex = res["bom_explosion"]
            if len(ex):
                piv = ex.pivot_table(index="item", columns="period",
                                     values="gross_requirement", aggfunc="sum").fillna(0)
                st.dataframe(piv.round(0), height=330)
                lvl = ex.groupby("level", as_index=False)["gross_requirement"].sum()
                fig = px.bar(lvl, x="level", y="gross_requirement",
                             labels={"level": "BOM level", "gross_requirement": "Units required"})
                fig.update_traces(marker_color=ui.ACCENT)
                st.plotly_chart(ui.theme(fig, 220, legend=False), key="bom_lvl")

# ---------------------------------------------------------------------------
# 7 MRP
# ---------------------------------------------------------------------------
with TABS[7]:
    ui.step("Step 7  ·  material requirements planning")
    ui.formula("Net requirement = gross requirement - scheduled receipts - opening available "
               "+ safety stock&nbsp;&nbsp;|&nbsp;&nbsp;planned release = planned receipt offset "
               "back by the lead time")
    mrp = res["mrp"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Items planned", f"{mrp['item'].nunique()}")
    c2.metric("Planned orders", f"{int((mrp['planned_order_receipt'] > 0).sum())}")
    c3.metric("Past due releases",
              f"{int((mrp['status'] == 'Past due release').sum())}")
    c4.metric("Shortages", f"{int((mrp['projected_available'] < 0).sum())}")

    c1, c2 = st.columns([1, 3], gap="medium")
    with c1:
        lvl_pick = st.selectbox("BOM level", ["All"] + sorted(mrp["level"].unique().tolist()),
                                key="mrp_lvl")
        pool = mrp if lvl_pick == "All" else mrp[mrp["level"] == lvl_pick]
        item = st.selectbox("Item", sorted(pool["item"].unique()), key="mrp_item")
    with c2:
        g = mrp[mrp["item"] == item]
        show = g[["period", "gross_requirement", "scheduled_receipts",
                  "projected_available_start", "net_requirement", "planned_order_receipt",
                  "release_period", "projected_available", "safety_stock", "status"]].copy()
        show.columns = ["Period", "Gross req", "Scheduled receipts", "Opening available",
                        "Net req", "Planned receipt", "Release in period", "Projected available",
                        "Safety stock", "Status"]
        st.dataframe(ui.status_frame(show.round(1), "Status"), hide_index=True)

    row = mrp[mrp["item"] == item].iloc[0]
    ui.note(f"Lead time {row['lead_time']:.0f} period(s), lot rule {row['lot_rule']}, "
            f"level {row['level']}. A receipt needed in period p is released in period p minus "
            f"the lead time, which is why releases can fall before the horizon starts.")

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        fig = go.Figure()
        fig.add_bar(x=g["period"], y=g["gross_requirement"], name="Gross requirement",
                    marker_color=ui.BLUE)
        fig.add_bar(x=g["period"], y=g["planned_order_receipt"], name="Planned receipt",
                    marker_color=ui.ACCENT)
        fig.add_scatter(x=g["period"], y=g["projected_available"], name="Projected available",
                        mode="lines+markers", line=dict(color=ui.GREEN, width=2))
        fig.update_layout(barmode="group", xaxis_title="Period", yaxis_title="Units")
        st.plotly_chart(ui.theme(fig, 320), key="mrp_chart")
    with c2:
        st.markdown("##### Where this requirement comes from")
        peg = res["mrp_pegging"]
        pp = peg[peg["component"] == item] if len(peg) else pd.DataFrame()
        if len(pp):
            st.dataframe(pp.rename(columns={"period": "Period", "parent": "Driven by",
                                            "qty": "Quantity", "qty_per": "Qty per",
                                            "parent_release_qty": "Parent release",
                                            "scrap_pct": "Scrap %"})
                         [["Period", "Driven by", "Parent release", "Qty per", "Scrap %", "Quantity"]]
                         .round(2), hide_index=True, height=300)
        else:
            st.caption("This item has independent demand, so its requirement comes from the "
                       "master schedule rather than a parent.")

    with st.expander("Full MRP output (the MRP_Output.csv shape)"):
        st.dataframe(res["mrp_output"].round(2), hide_index=True, height=400)
    if len(res["mrp_exceptions"]):
        st.markdown("##### Material exceptions")
        st.dataframe(ui.status_frame(res["mrp_exceptions"], "severity"), hide_index=True)

# ---------------------------------------------------------------------------
# 8 Order release
# ---------------------------------------------------------------------------
with TABS[8]:
    ui.step("Step 8  ·  production order release")
    rel = res["order_release"]
    if not len(rel):
        st.info("No finished goods orders fall inside the release window.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Candidate orders", f"{len(rel)}")
        c2.metric("Released", f"{int((rel['status'] == 'Released').sum())}")
        c3.metric("Held", f"{int((rel['status'] == 'Held').sum())}")

        show = rel[["order_id", "item", "qty", "release_period", "due_period",
                    "status", "constraint", "reason"]].copy()
        show.columns = ["Order", "Item", "Quantity", "Release period", "Due period",
                        "Status", "Constraint", "Why"]
        st.dataframe(ui.status_frame(show.round(1), "Status"), hide_index=True)
        ui.note("An order is only released once every component on its bill is available at the "
                "release period. Held orders name the component that is holding them up, which is "
                "the line the buyer has to chase.")

        by = rel.groupby("constraint", as_index=False).size()
        fig = px.bar(by, x="size", y="constraint", orientation="h",
                     labels={"size": "Orders", "constraint": ""})
        fig.update_traces(marker_color=[ui.GREEN if c == "None" else ui.RED for c in by["constraint"]])
        st.plotly_chart(ui.theme(fig, ui.fit_height(len(by), per=40, base=120),
                                 legend=False), key="rel_bar")

# ---------------------------------------------------------------------------
# 9 Capacity and job list
# ---------------------------------------------------------------------------
with TABS[9]:
    ui.step("Step 9  ·  routing, work centre capacity and the executable job list")
    av = available_hours(res["machine_capacity"])
    ui.formula("Available hours = machines x hours per shift x shifts per day x days per week "
               "x efficiency")
    st.dataframe(av.rename(columns={
        "work_centre": "Work centre", "description": "Description", "num_machines": "Machines",
        "hours_per_shift": "Hours per shift", "shifts_per_day": "Shifts", "days_per_week": "Days",
        "efficiency": "Efficiency %", "machine_hours": "Hours per machine",
        "gross_hours": "Gross hours", "available_hours": "Available hours"}).round(1),
        hide_index=True)

    st.markdown("##### Routing")
    rt = ds.get("routing")
    if len(rt):
        st.dataframe(rt.rename(columns={
            "item": "Item", "op_seq": "Op", "operation": "Operation",
            "work_centre": "Work centre", "setup_min": "Setup min",
            "run_min": "Run min per unit"}), hide_index=True, height=260)

    st.markdown("##### Executable job list")
    jl = res["job_list"]
    if len(jl):
        c1, c2, c3 = st.columns(3)
        c1.metric("Jobs", f"{len(jl)}")
        c2.metric("Operations", f"{int(jl['operations'].sum())}")
        c3.metric("Work content", f"{jl['total_processing_h'].sum():,.0f} h")
        st.dataframe(jl.rename(columns={
            "job": "Job", "item": "Item", "qty": "Qty", "operations": "Ops",
            "total_processing_h": "Processing h", "arrival_period": "Arrives",
            "due_period": "Due", "priority": "Priority", "work_centres": "Route"}),
            hide_index=True, height=320)
        ui.note("Large orders are split into sub lots so the shop can overlap operations, which is "
                "what lets a downstream work centre start before the whole batch is finished.")
    else:
        st.info("No jobs to schedule. Widen the release window or check held orders in step 8.")

# ---------------------------------------------------------------------------
# 10 Schedule
# ---------------------------------------------------------------------------
with TABS[10]:
    ui.step("Step 10  ·  shop floor scheduling simulation")
    sched = res["schedule"]
    if not len(sched):
        st.info("Nothing to schedule on this run.")
    else:
        st.markdown(f"##### Gantt under {res['dispatch_rule']}")
        ui.note(RULE_HELP.get(res["dispatch_rule"], ""))
        colour_by = st.radio("Colour by", ["Product", "Job"], horizontal=True, key="gantt_colour")
        origin = ds.period_dates.get(p0, pd.Timestamp("2026-01-05"))
        g = to_gantt(sched, origin, settings.period_minutes)
        fig = px.timeline(g, x_start="Start", x_end="Finish", y="machine",
                          color="item" if colour_by == "Product" else "job",
                          hover_data=["job", "operation", "qty", "setup_min", "proc_min"])
        fig.update_yaxes(autorange="reversed", title="")
        fig.update_layout(xaxis_title="")
        n_series = g["item"].nunique() if colour_by == "Product" else g["job"].nunique()
        st.plotly_chart(ui.theme(fig, ui.fit_height(g["machine"].nunique(), per=30, base=150),
                                 assumed_width=900), key="gantt")
        if n_series > 14:
            ui.note(f"{n_series} series would not fit a readable legend, so it is left off. "
                    f"Hover any bar for the job, product, quantity and timings.")

        k = res["schedule_kpis"]
        cols = st.columns(6)
        for c, (lab, key, unit) in zip(cols, [
                ("Makespan", "Makespan_h", "h"), ("Avg flow", "Avg_Flow_Time_h", "h"),
                ("Avg wait", "Avg_Waiting_Time_h", "h"), ("Avg tardiness", "Avg_Tardiness_h", "h"),
                ("On time", "OTD_Pct", "%"), ("Utilisation", "Utilisation_Pct", "%")]):
            c.metric(lab, f"{k.get(key, 0):,.1f}{unit}")

        c1, c2 = st.columns([1, 1], gap="medium")
        with c1:
            st.markdown("##### Work centre utilisation")
            u = res["utilisation"]
            fig = px.bar(u, x="utilisation_pct", y="work_centre", orientation="h",
                         labels={"utilisation_pct": "Utilisation %", "work_centre": ""})
            fig.update_traces(marker_color=[ui.RED if v > 90 else (ui.ACCENT if v > 75 else ui.BLUE)
                                            for v in u["utilisation_pct"]])
            st.plotly_chart(ui.theme(fig, ui.fit_height(len(u)), legend=False), key="util_bar")
        with c2:
            st.markdown("##### Job completion against due date")
            jd = res["schedule_jobs"]
            jd2 = jd.copy()
            jd2["completion_h"] = jd2["completion"] / 60
            jd2["due_h"] = jd2["due"] / 60
            fig = go.Figure()
            fig.add_bar(x=jd2["completion_h"], y=jd2["job"], orientation="h", name="Completion",
                        marker_color=[ui.GREEN if o else ui.RED for o in jd2["on_time"]])
            fig.add_scatter(x=jd2["due_h"], y=jd2["job"], mode="markers", name="Due",
                            marker=dict(symbol="line-ns", size=14, line=dict(color=ui.TEXT, width=2)))
            fig.update_layout(xaxis_title="Hours from the start of the period", yaxis_title="")
            st.plotly_chart(ui.theme(fig, ui.fit_height(len(jd2))), key="due_bar")

        problems = res["schedule_valid"]
        if problems and problems != ["no jobs"]:
            st.error("Schedule check found: " + "; ".join(problems[:4]))
        else:
            st.success("Schedule check passed: no machine runs two operations at once, "
                       "no operation starts before the one in front of it finishes.")

        with st.expander("Operation by operation schedule"):
            st.dataframe(sched.round(1), hide_index=True, height=400)

# ---------------------------------------------------------------------------
# 11 KPI comparison
# ---------------------------------------------------------------------------
with TABS[11]:
    ui.step("Step 11  ·  performance comparison across dispatching rules")
    tbl = res["rule_comparison"]
    if not len(tbl):
        st.info("No jobs available to compare rules on.")
    else:
        rank = res["rule_ranking"]
        best_rule = rank.iloc[0]["rule"]
        st.markdown(f"##### {best_rule} comes out ahead on the weighted score")
        ui.note("The weighting leans on tardiness and on time delivery, then flow time, makespan "
                "and utilisation. Change the dispatching rule in the control panel to schedule "
                "the floor with any of them.")

        show = tbl.rename(columns={
            "rule": "Rule", "Makespan_h": "Makespan h", "Avg_Flow_Time_h": "Avg flow h",
            "Avg_Waiting_Time_h": "Avg wait h", "Avg_Tardiness_h": "Avg tardiness h",
            "Max_Tardiness_h": "Max tardiness h", "Jobs_Tardy": "Jobs late",
            "OTD_Pct": "On time %", "Utilisation_Pct": "Utilisation %",
            "Avg_WIP_Jobs": "Avg WIP", "Throughput_Units_Per_Period": "Units per period",
            "Total_Setup_h": "Setup h", "Takt_Time_Min_Per_Unit": "Takt min per unit"})
        cols = ["Rule", "Makespan h", "Avg flow h", "Avg wait h", "Avg tardiness h",
                "Max tardiness h", "Jobs late", "On time %", "Utilisation %", "Avg WIP",
                "Units per period", "Setup h", "Takt min per unit"]
        st.dataframe(show[[c for c in cols if c in show.columns]].round(2), hide_index=True)

        c1, c2 = st.columns([1.3, 1], gap="medium")
        with c1:
            metric = st.selectbox("Compare on", [
                "Avg_Flow_Time_h", "Avg_Tardiness_h", "Makespan_h", "OTD_Pct",
                "Utilisation_Pct", "Avg_WIP_Jobs", "Avg_Waiting_Time_h", "Total_Setup_h"],
                format_func=lambda v: v.replace("_", " "), key="kpi_metric")
            d = tbl.sort_values(metric)
            fig = px.bar(d, x="rule", y=metric, text=metric)
            best_for = res["rule_best"].get(metric)
            fig.update_traces(marker_color=[ui.ACCENT if r == best_for else ui.BLUE for r in d["rule"]],
                              texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
            fig.update_layout(xaxis_title="", yaxis_title=metric.replace("_", " "))
            st.plotly_chart(ui.theme(fig, 350, legend=False), key="kpi_bar")
            fig.update_yaxes(rangemode="tozero")
        with c2:
            st.markdown("##### Best rule per measure")
            bl = pd.DataFrame([{"Measure": k.replace("_", " "), "Best rule": v}
                               for k, v in res["rule_best"].items()])
            st.dataframe(bl, hide_index=True, height=330)

        st.markdown("##### Weighted ranking")
        st.dataframe(rank[["rule", "score", "Avg_Tardiness_h", "OTD_Pct", "Avg_Flow_Time_h",
                           "Makespan_h", "Utilisation_Pct"]].rename(columns={
            "rule": "Rule", "score": "Score", "Avg_Tardiness_h": "Avg tardiness h",
            "OTD_Pct": "On time %", "Avg_Flow_Time_h": "Avg flow h",
            "Makespan_h": "Makespan h", "Utilisation_Pct": "Utilisation %"}).round(2),
            hide_index=True)

# ---------------------------------------------------------------------------
# 12 Exceptions and decisions
# ---------------------------------------------------------------------------
with TABS[12]:
    ui.step("Step 12  ·  exception management and managerial decision")
    exc = res["exceptions"]
    ui.note("Six agents inspect the plan, one per module. A supervisor ranks what they raise on "
            "severity blended with the money at stake. Accepting a recommendation writes a real "
            "change into the next planning run rather than just logging it.")

    if not len(exc):
        st.success("Nothing to escalate.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Open", f"{len(exc)}")
        c2.metric("High severity", f"{int((exc['severity'] == 'High').sum())}")
        c3.metric("Agents reporting", f"{exc['agent'].nunique()}")
        c4.metric("Value at stake", f"INR {exc['exposure_value'].sum() / 1e7:,.2f} Cr")

        c1, c2 = st.columns([1.1, 1], gap="medium")
        with c1:
            by_agent = exc.groupby(["agent", "severity"], as_index=False).size()
            fig = px.bar(by_agent, x="size", y="agent", color="severity", orientation="h",
                         color_discrete_map={"High": ui.RED, "Medium": ui.ACCENT, "Low": ui.BLUE},
                         labels={"size": "Exceptions", "agent": ""})
            fig.update_layout(barmode="stack")
            st.plotly_chart(ui.theme(fig, ui.fit_height(exc["agent"].nunique(), per=34, base=130)),
                            key="exc_agent")
        with c2:
            top = exc[exc["exposure_value"] > 0].head(8)
            if len(top):
                fig = px.bar(top, x="exposure_value", y="item", orientation="h",
                             labels={"exposure_value": "Exposure INR", "item": ""})
                fig.update_traces(marker_color=ui.ACCENT)
                st.plotly_chart(ui.theme(fig, ui.fit_height(len(top), per=34, base=130),
                                         legend=False), key="exc_money")

        sev_filter = st.multiselect("Severity", ["High", "Medium", "Low"], default=["High", "Medium"])
        view = exc[exc["severity"].isin(sev_filter)] if sev_filter else exc
        st.dataframe(view[["exception_id", "agent", "item", "period", "type", "severity",
                           "detail", "recommended_action", "exposure_value"]].rename(columns={
            "exception_id": "ID", "agent": "Raised by", "item": "Item", "period": "Period",
            "type": "Exception", "severity": "Severity", "detail": "Detail",
            "recommended_action": "Recommended action", "exposure_value": "Exposure INR"}),
            hide_index=True, height=300)

        st.markdown("##### Manager decision")
        if len(view):
            pick = st.selectbox("Exception", view["exception_id"].tolist(),
                                format_func=lambda i: f"{i}  {view.loc[view['exception_id'] == i, 'item'].iloc[0]}"
                                                      f"  {view.loc[view['exception_id'] == i, 'type'].iloc[0]}")
            row = view[view["exception_id"] == pick].iloc[0]
            st.markdown(f"<div class='panel'><b>{ui.esc(row['type'])}</b> on "
                        f"{ui.esc(row['item'])}<br>"
                        f"<span class='note'>{ui.esc(row['detail'])}<br>"
                        f"<b style='color:{ui.ACCENT}'>Recommended:</b> "
                        f"{ui.esc(row['recommended_action'])}</span></div>",
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                decision = st.radio("Decision", ["Accept", "Modify", "Reject"], horizontal=False)
            with c2:
                qty = st.number_input("Quantity to expedite", 0.0, 1e9,
                                      float(row.get("exposure_value", 0) and 0) or 1000.0,
                                      step=100.0,
                                      help="Used when the action is an expedite")
            with c3:
                note_txt = st.text_input("Note for the log", "")
            if st.button("Apply decision and replan", type="primary"):
                ov = st.session_state.overrides
                apply_decision(ov, row, "Reject" if decision == "Reject" else "Accept",
                               note_txt, qty)
                st.session_state.decisions.append({
                    "Exception": row["exception_id"], "Item": row["item"],
                    "Type": row["type"], "Decision": decision, "Quantity": qty,
                    "Note": note_txt})
                st.session_state.cache = {}
                st.rerun()

        if st.session_state.decisions:
            st.markdown("##### Decision log")
            st.dataframe(pd.DataFrame(st.session_state.decisions), hide_index=True)
            st.caption("Every accepted decision is applied to the plan above. "
                       "Clear them from the control panel.")

# ---------------------------------------------------------------------------
# 13 Scenarios
# ---------------------------------------------------------------------------
with TABS[13]:
    ui.step("Step 13  ·  what if and disruption simulation")
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        kind = st.selectbox("Scenario", list(SCENARIO_LIBRARY), key="scen_kind")
        st.caption(SCENARIO_LIBRARY[kind])
    with c2:
        if kind == "Demand surge":
            target = st.selectbox("Applies to", ["All finished goods"] + res["fg_items"])
            mag = st.slider("Uplift %", 5, 100, 30)
            delay = 0
        elif kind == "Supplier delay":
            items = sorted(ds.get("inventory_master")["item"].tolist())
            target = st.selectbox("Material", items,
                                  index=items.index("RM-CHILLI") if "RM-CHILLI" in items else 0)
            delay = st.slider("Delay (periods)", 1, 6, 2)
            mag = 0
        elif kind == "Machine breakdown":
            wcs = sorted(res["machine_capacity"]["work_centre"].tolist())
            target = st.selectbox("Work centre", wcs)
            mag = st.slider("Hours down", 4, 120, 48)
            delay = 0
        elif kind == "Rush order":
            target = st.selectbox("Product", res["fg_items"])
            mag = st.slider("Size as % of a normal period", 10, 200, 80)
            delay = 0
        else:
            wcs = sorted(res["machine_capacity"]["work_centre"].tolist())
            target = st.selectbox("Work centre", wcs)
            mag = st.slider("Extra capacity %", 25, 200, 100)
            delay = 0
    with c3:
        st.write("")
        st.write("")
        if st.button("Run this scenario", type="primary"):
            # the scenario agent owns this, and reports back through the bus
            answer = session.coordinator.ask(
                "Scenario", "scenario.run", f"run {kind}",
                kind=kind, magnitude=float(mag), target=target, delay=int(delay))
            st.session_state.scenario_result = session._scenario_cache[kind]["result"]
            st.session_state.scenario_label = kind
            st.session_state.scenario_note = answer.get("summary", "")
        if st.button("Clear scenario"):
            st.session_state.scenario_result = None

    scen = st.session_state.scenario_result
    if scen is not None and st.session_state.get("scenario_note"):
        st.markdown(f"<div class='reply'><div class='hl'>"
                    f"{ui.esc(st.session_state['scenario_note'])}</div></div>",
                    unsafe_allow_html=True)
    if scen is None:
        st.info("Pick a disruption and run it. The baseline stays on screen for comparison.")
    else:
        label = st.session_state.scenario_label
        st.markdown(f"##### Baseline against {label.lower()}")
        for line in scen["overrides"].summary():
            st.caption("• " + line)
        cmp_df = compare_runs(res, scen, label)

        def _colour(v):
            return {"Worse": f"color: {ui.RED}", "Better": f"color: {ui.GREEN}"}.get(v, "")

        st.dataframe(cmp_df.style.map(_colour, subset=["Direction"]), hide_index=True)

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown("###### Work centre load")
            bl = res["capacity_load"].groupby("period", as_index=False)["utilisation_pct"].max()
            sl = scen["capacity_load"].groupby("period", as_index=False)["utilisation_pct"].max()
            fig = go.Figure()
            fig.add_scatter(x=bl["period"], y=bl["utilisation_pct"], name="Baseline",
                            mode="lines+markers", line=dict(color=ui.BLUE, width=2))
            fig.add_scatter(x=sl["period"], y=sl["utilisation_pct"], name=label,
                            mode="lines+markers", line=dict(color=ui.ACCENT, width=2))
            fig.add_hline(y=100, line=dict(color=ui.RED, width=1, dash="dash"))
            fig.update_layout(xaxis_title="Period", yaxis_title="Peak utilisation %")
            st.plotly_chart(ui.theme(fig, 300), key="scen_load")
        with c2:
            st.markdown("###### Master schedule volume")
            bm = res["mps"].groupby("period", as_index=False)["mps_qty"].sum()
            sm = scen["mps"].groupby("period", as_index=False)["mps_qty"].sum()
            fig = go.Figure()
            fig.add_bar(x=bm["period"], y=bm["mps_qty"], name="Baseline", marker_color=ui.BLUE)
            fig.add_bar(x=sm["period"], y=sm["mps_qty"], name=label, marker_color=ui.ACCENT)
            fig.update_layout(barmode="group", xaxis_title="Period", yaxis_title="Units")
            st.plotly_chart(ui.theme(fig, 300), key="scen_mps")

        st.markdown("###### New exceptions this scenario creates")
        b_ids = set(zip(res["exceptions"]["item"], res["exceptions"]["type"])) if len(res["exceptions"]) else set()
        se = scen["exceptions"]
        new = se[[(i, t) not in b_ids for i, t in zip(se["item"], se["type"])]] if len(se) else se
        if len(new):
            st.dataframe(new[["agent", "item", "type", "severity", "detail",
                              "recommended_action"]], hide_index=True, height=240)
        else:
            st.caption("No new exception types appear under this scenario.")

# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
with TABS[14]:
    ui.step("Step 13.4  ·  deliverables")
    st.markdown("##### Export the whole run")
    frames = {
        "Summary": pd.DataFrame([res["kpi_summary"]]).T.reset_index().rename(
            columns={"index": "Measure", 0: "Value"}),
        "Forecast": res["forecast"],
        "Forecast_accuracy": res["forecast_run"]["accuracy"],
        "Forecast_selected": res["forecast_run"]["selected"],
        "Inventory_parameters": res["inventory_params"],
        "Inventory_projection": res["inventory_projection"],
        "MPS": res["mps"],
        "MPS_ATP": res["mps_atp"],
        "Capacity_load": res["capacity_load"],
        "BOM_explosion": res["bom_explosion"],
        "MRP_output": res["mrp_output"],
        "MRP_pegging": res["mrp_pegging"],
        "Order_release": res["order_release"],
        "Job_list": res["job_list"],
        "Schedule": res["schedule"],
        "Job_results": res["schedule_jobs"],
        "Rule_comparison": res["rule_comparison"],
        "Utilisation": res["utilisation"],
        "Exceptions": res["exceptions"],
    }
    if st.session_state.decisions:
        frames["Decision_log"] = pd.DataFrame(st.session_state.decisions)
    if st.session_state.scenario_result is not None:
        frames["Scenario_comparison"] = compare_runs(
            res, st.session_state.scenario_result, st.session_state.scenario_label)

    c1, c2 = st.columns(2)
    with c1:
        ui.download_frames(frames, "maggi_control_tower_run.xlsx",
                           "Download the full run as one workbook")
    with c2:
        pick = st.selectbox("Or a single table as CSV", list(frames))
        st.download_button(f"Download {pick}.csv",
                           frames[pick].to_csv(index=False).encode(),
                           file_name=f"{pick}.csv", mime="text/csv")

    st.markdown("##### How the modules connect")
    ui.note("Every number on every tab comes from one pass of the same pipeline, so a change to any "
            "parameter in the control panel flows all the way through to the Gantt chart.")
    flow = pd.DataFrame([
        {"Step": "3 Forecast", "Reads": "Demand history", "Writes": "Statistical forecast per period"},
        {"Step": "4 Inventory", "Reads": "Forecast, item master, BOM", "Writes": "Safety stock, reorder point, EOQ, projected position"},
        {"Step": "5 MPS", "Reads": "Forecast, customer orders, stock", "Writes": "Master schedule, PAB, ATP, capacity check"},
        {"Step": "6 BOM", "Reads": "Master schedule, bill of material", "Writes": "Gross requirements by level"},
        {"Step": "7 MRP", "Reads": "Gross requirements, stock, lead times", "Writes": "Planned receipts and releases"},
        {"Step": "8 Release", "Reads": "Planned releases, component availability", "Writes": "Released and held works orders"},
        {"Step": "9 Capacity", "Reads": "Released orders, routing, machines", "Writes": "Executable job list"},
        {"Step": "10 Schedule", "Reads": "Job list, dispatching rule", "Writes": "Operation schedule and Gantt"},
        {"Step": "11 KPIs", "Reads": "Schedule under every rule", "Writes": "Comparison and best rule"},
        {"Step": "12 Exceptions", "Reads": "Output of every module", "Writes": "Ranked actions and decisions"},
        {"Step": "13 Scenarios", "Reads": "Baseline plus a disruption", "Writes": "Side by side impact"},
    ])
    st.dataframe(flow, hide_index=True)

    st.markdown("##### Data note")
    ui.note("The bundled Nestle Maggi dataset is synthetic. Product names and the process flow are "
            "modelled on publicly understood instant noodle and ketchup manufacturing; every "
            "quantity, cost, lead time and demand figure is generated from a seeded random model "
            "for teaching purposes and is not Nestle company data.")
