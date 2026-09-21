"""
The control tower run: forecasting to scheduling in one pass.

Everything the dashboard shows comes from a single call to `run_pipeline`, so
the baseline plan and any scenario are produced by exactly the same code with
different settings and overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import (agents, bom as bom_mod, capacity as cap_mod, forecasting as fc_mod,
               inventory as inv_mod, kpis as kpi_mod, mps as mps_mod, mrp as mrp_mod,
               scheduling as sch_mod)
from .agents import Overrides
from .loader import Dataset


@dataclass
class Settings:
    use_mps_input: bool = False
    horizon: int = 12
    season_length: int = 13
    holdout: int = 8
    accuracy_criterion: str = "MAPE"
    mape_threshold: float = 20.0
    service_level: float = 95.0
    demand_time_fence: int = 2
    demand_rule: str = "max"
    poq_periods: int = 2
    level_capacity: bool = True
    release_window: int = 1
    max_lot: float = 400.0
    dispatch_rule: str = "EDD"
    dispatch_rules: List[str] = field(default_factory=lambda: list(sch_mod.DISPATCH_RULES))
    sequence_dependent_setup: bool = True
    period_minutes: float = 8640.0
    excess_weeks: float = 6.0
    include_open_orders: bool = True
    forecast_options: dict = field(default_factory=dict)
    method_override: Dict[str, str] = field(default_factory=dict)
    lot_rule_override: Dict[str, str] = field(default_factory=dict)


def infer_period_minutes(machine_capacity: pd.DataFrame) -> float:
    """One period of operating time, taken from the busiest work centre calendar."""
    av = cap_mod.available_hours(machine_capacity)
    if len(av) == 0:
        return 8640.0
    return float(av["machine_hours"].max()) * 60.0


def run_pipeline(ds: Dataset, settings: Optional[Settings] = None,
                 ov: Optional[Overrides] = None, hooks: object = None) -> Dict[str, object]:
    """Run every module in order.

    `hooks` is optional. When a multi agent session supplies one, the decision
    points inside the run are answered by the agents that own them instead of by
    the defaults, and each question and answer is recorded. The arithmetic is
    identical either way.
    """
    st = settings or Settings()
    ov = ov or Overrides()
    _h = lambda name, default=None: getattr(hooks, name, default) if hooks else default
    _notify = _h("notify") or (lambda *a, **k: None)
    out: Dict[str, object] = {"settings": st, "overrides": ov, "warnings": []}

    dh = ds.get("demand_history")
    inv_master = ds.get("inventory_master")
    bom = ds.get("bom")
    orders = ds.get("customer_orders")
    routing = ds.get("routing")
    machine = ds.get("machine_capacity")
    supplier = ds.get("supplier_leadtime")
    sr = ds.get("scheduled_receipts")
    open_orders = ds.get("production_orders")

    if len(machine):
        st.period_minutes = infer_period_minutes(machine)

    # capacity overrides act on the machine master before anything else uses it
    if ov.capacity_multiplier and len(machine):
        machine = machine.copy()
        machine["num_machines"] = machine.apply(
            lambda r: float(r["num_machines"]) * ov.capacity_multiplier.get(r["work_centre"], 1.0), axis=1)
    out["machine_capacity"] = machine

    p0 = ds.plan_start
    periods = list(range(p0, p0 + st.horizon))
    out["periods"] = periods
    out["first_period"] = p0

    # ---------------------------------------------------------------- module 3
    f = fc_mod.run_forecast(dh, st.horizon, p0, st.season_length, st.holdout,
                            st.accuracy_criterion, st.method_override, st.mape_threshold, st.forecast_options)
    forecast = f["forecast"].copy()
    if ov.demand_multiplier and len(forecast):
        forecast["forecast"] = forecast.apply(
            lambda r: r["forecast"] * ov.demand_multiplier.get(r["item"], 1.0), axis=1).round(2)
    out["forecast_run"] = f
    out["forecast"] = forecast
    _notify("forecast", out)

    # customer orders can also be scaled by a demand scenario
    if ov.demand_multiplier and len(orders):
        orders = orders.copy()
        orders["qty"] = orders.apply(
            lambda r: r["qty"] * ov.demand_multiplier.get(r["item"], 1.0), axis=1)
    if ov.rush_orders:
        extra = pd.DataFrame([{
            "item": r["item"], "period": int(r["due_period"]), "qty": float(r["qty"]),
            "order_id": r.get("order_id", "RUSH"), "customer": r.get("customer", "Rush order"),
            "priority": 1, "due_date": pd.NaT,
        } for r in ov.rush_orders])
        orders = pd.concat([orders, extra], ignore_index=True) if len(orders) else extra
    out["customer_orders"] = orders

    # expedited receipts join the scheduled receipts file
    if ov.extra_receipts:
        extra = pd.DataFrame([{"item": r["item"],
                               "period": p0 if r.get("period") is None else max(int(r["period"]), p0),
                               "qty": float(r["qty"]), "receipt_id": "EXP", "supplier": ""}
                              for r in ov.extra_receipts if float(r.get("qty", 0)) > 0])
        if len(extra):
            sr = pd.concat([sr, extra], ignore_index=True) if len(sr) else extra
    out["scheduled_receipts"] = sr

    # lead time changes
    lt_override: Dict[str, float] = dict(ov.lead_time_override)
    if ov.lead_time_delta and len(inv_master):
        base = dict(zip(inv_master["item"], inv_master["lead_time"]))
        for it, d in ov.lead_time_delta.items():
            lt_override[it] = max(0.0, float(base.get(it, 1)) + float(d))
    out["lead_time_override"] = lt_override

    # ---------------------------------------------------------------- module 4
    params = inv_mod.compute_parameters(inv_master, dh, bom, forecast, st.service_level)
    fg_items = _finished_goods(dh, bom, inv_master)
    out["fg_items"] = fg_items
    projection = inv_mod.project_inventory(fg_items, inv_master, forecast, orders, sr,
                                           periods, st.demand_rule)
    inv_exc = inv_mod.inventory_exceptions(params, projection, st.excess_weeks)
    out["inventory_params"] = params
    out["inventory_projection"] = projection
    out["inventory_exceptions"] = inv_exc

    # ---------------------------------------------------------------- module 5
    mps_res = mps_mod.run_mps(fg_items, forecast, orders, inv_master, sr, periods, params,
                              st.demand_time_fence, st.poq_periods, st.demand_rule,
                              st.lot_rule_override,
                              ds.get("mps_input") if st.use_mps_input else None)
    grid = mps_res["grid"]
    feas_before = mps_mod.check_feasibility(grid, routing, machine)
    moves = pd.DataFrame()
    if st.level_capacity and not st.use_mps_input and len(routing) and len(machine):
        lv = mps_mod.level_capacity(grid, routing, machine,
                                    ledger_factory=_h("ledger_factory"))
        grid, moves = lv["grid"], lv["moves"]
    feas = mps_mod.check_feasibility(grid, routing, machine)
    out["mps"] = grid
    out["mps_atp"] = mps_res["atp"]
    out["mps_feasibility"] = feas
    out["mps_feasibility_before"] = feas_before
    out["mps_moves"] = moves
    out["capacity_load"] = feas["load"]
    _notify("mps", out)

    # ------------------------------------------------------------- modules 6, 7
    out["bom_explosion"] = bom_mod.explode(grid, bom, "mps_qty")
    mrp_res = mrp_mod.run_mrp(grid, bom, inv_master, sr, periods, params,
                              st.poq_periods, lt_override)
    out["mrp"] = mrp_res["plan"]
    out["mrp_output"] = mrp_res["output"]
    out["mrp_pegging"] = mrp_res["pegging"]
    mrp_exc = mrp_mod.mrp_exceptions(mrp_res["plan"], inv_master, supplier, p0)
    out["mrp_exceptions"] = mrp_exc
    _notify("mrp", out)

    # ---------------------------------------------------------------- module 8
    cand = sch_mod.candidate_orders(mrp_res["plan"], fg_items, p0, st.release_window)
    released = sch_mod.check_material_availability(cand, bom, mrp_res["plan"],
                                                   feas["load"], routing, p0,
                                                   material_probe=_h("material_probe"),
                                                   capacity_probe=_h("capacity_probe"))
    if len(released) and ov.priority_items:
        released["priority"] = released["item"].map(
            lambda i: 1 if i in ov.priority_items else 2)
    out["order_candidates"] = cand
    out["order_release"] = released
    _notify("release", out)

    # ------------------------------------------------------------ modules 9, 10
    open_df = open_orders if (st.include_open_orders and len(open_orders)) else None
    ready = released[released["status"] == "Released"] if len(released) else released
    jobs = sch_mod.build_jobs(ready, routing, machine, p0, st.period_minutes,
                              st.max_lot, open_df)
    out["jobs"] = jobs
    out["job_list"] = sch_mod.job_list_frame(jobs, st.period_minutes, p0)

    rule = ov.dispatch_rule or st.dispatch_rule
    comparison = kpi_mod.compare_rules(jobs, machine, st.dispatch_rules,
                                       st.period_minutes, st.sequence_dependent_setup,
                                       ov.breakdowns)
    out["rule_comparison"] = comparison["table"]
    out["rule_best"] = comparison["best"]
    out["rule_ranking"] = kpi_mod.overall_recommendation(comparison["table"])

    run = comparison["runs"].get(rule)
    if run is None:
        run = sch_mod.simulate(jobs, machine, rule, st.sequence_dependent_setup, ov.breakdowns)
    out["schedule"] = run["schedule"]
    out["schedule_jobs"] = run["jobs"]
    out["dispatch_rule"] = rule
    out["schedule_kpis"] = kpi_mod.schedule_kpis(run["schedule"], run["jobs"], machine,
                                                 st.period_minutes)
    out["utilisation"] = kpi_mod.utilisation_by_wc(run["schedule"], machine)
    out["schedule_valid"] = sch_mod.validate_schedule(run["schedule"], jobs) if jobs else ["no jobs"]

    # --------------------------------------------------------------- module 12
    sched_exc = agents.schedule_exceptions(run["jobs"], out["utilisation"],
                                           st.period_minutes, p0)
    sup_exc = agents.supplier_exceptions(supplier, mrp_res["plan"], p0)
    costs = dict(zip(inv_master["item"], inv_master["unit_cost"])) if len(inv_master) else {}
    out["exceptions"] = agents.collect(f["exceptions"], inv_exc, feas["issues"],
                                       mrp_exc, sched_exc, sup_exc, costs)

    out["kpi_summary"] = summarise(out)
    return out


def _finished_goods(dh: pd.DataFrame, bom: pd.DataFrame,
                    inv_master: pd.DataFrame) -> List[str]:
    """Items with independent demand that are never a component of anything."""
    demanded = list(dict.fromkeys(dh["item"].tolist())) if len(dh) else []
    components = set(bom["component"]) if len(bom) else set()
    fg = [i for i in demanded if i not in components]
    if not fg:
        if len(inv_master) and "item_type" in inv_master.columns:
            fg = inv_master.loc[inv_master["item_type"].astype(str).str.upper().str.startswith("FG"),
                                "item"].tolist()
    return fg or demanded


def summarise(res: Dict[str, object]) -> Dict[str, object]:
    """Headline numbers for the control tower landing page."""
    sel = res["forecast_run"]["selected"]
    params = res["inventory_params"]
    mps = res["mps"]
    mrp = res["mrp"]
    load = res["capacity_load"]
    k = res.get("schedule_kpis", {}) or {}
    exc = res.get("exceptions", pd.DataFrame())

    inv_value = float(params["inventory_value"].sum()) if len(params) else 0.0
    plan_value = 0.0
    if len(mps) and len(params):
        cost = dict(zip(params["item"], params["unit_cost"]))
        plan_value = float(sum(r["mps_qty"] * cost.get(r["item"], 0) for _, r in mps.iterrows()))

    return {
        "forecast_mape": round(float(sel["MAPE"].mean()), 2) if len(sel) else np.nan,
        "items_planned": int(mrp["item"].nunique()) if len(mrp) else 0,
        "planned_orders": int((mrp["planned_order_receipt"] > 0).sum()) if len(mrp) else 0,
        "mps_units": round(float(mps["mps_qty"].sum()), 0) if len(mps) else 0,
        "mps_value": round(plan_value, 0),
        "inventory_value": round(inv_value, 0),
        "peak_utilisation": round(float(load["utilisation_pct"].max()), 1) if len(load) else 0.0,
        "overloaded_periods": int((load["status"] == "Overloaded").sum()) if len(load) else 0,
        "otd_pct": k.get("OTD_Pct", np.nan),
        "makespan_h": k.get("Makespan_h", np.nan),
        "avg_flow_h": k.get("Avg_Flow_Time_h", np.nan),
        "avg_tardiness_h": k.get("Avg_Tardiness_h", np.nan),
        "utilisation_pct": k.get("Utilisation_Pct", np.nan),
        "jobs": k.get("Jobs_Total", 0),
        "exceptions_total": int(len(exc)),
        "exceptions_high": int((exc["severity"] == "High").sum()) if len(exc) else 0,
    }


# ---------------------------------------------------------------------------
# Module 13: scenario simulation
# ---------------------------------------------------------------------------
SCENARIO_LIBRARY = {
    "Demand surge": "Festive uplift across the noodle range",
    "Supplier delay": "A key raw material supplier slips by two periods",
    "Machine breakdown": "One machine at the bottleneck is down for part of the week",
    "Rush order": "A large unplanned order lands with a tight due date",
    "Capacity expansion": "An extra machine is added at the bottleneck",
}


def compare_runs(base: Dict[str, object], scen: Dict[str, object],
                 label: str = "Scenario") -> pd.DataFrame:
    """13.2 impact analysis across every module."""
    b, s = base["kpi_summary"], scen["kpi_summary"]
    metrics = [
        ("Average forecast MAPE %", "forecast_mape", "lower"),
        ("Master schedule units", "mps_units", "info"),
        ("Master schedule value INR", "mps_value", "info"),
        ("Planned orders", "planned_orders", "info"),
        ("Peak work centre utilisation %", "peak_utilisation", "lower"),
        ("Overloaded work centre periods", "overloaded_periods", "lower"),
        ("Jobs scheduled", "jobs", "info"),
        ("Makespan hours", "makespan_h", "lower"),
        ("Average flow time hours", "avg_flow_h", "lower"),
        ("Average tardiness hours", "avg_tardiness_h", "lower"),
        ("On time delivery %", "otd_pct", "higher"),
        ("Shop floor utilisation %", "utilisation_pct", "higher"),
        ("Open exceptions", "exceptions_total", "lower"),
        ("High severity exceptions", "exceptions_high", "lower"),
    ]
    rows = []
    for name, key, better in metrics:
        bv, sv = b.get(key, np.nan), s.get(key, np.nan)
        try:
            delta = float(sv) - float(bv)
        except (TypeError, ValueError):
            delta = np.nan
        if better == "info" or np.isnan(delta) or delta == 0:
            verdict = "No change" if delta == 0 else "Context"
        elif better == "lower":
            verdict = "Worse" if delta > 0 else "Better"
        else:
            verdict = "Better" if delta > 0 else "Worse"
        rows.append({"Metric": name, "Baseline": bv, label: sv,
                     "Change": round(delta, 2) if not np.isnan(delta) else np.nan,
                     "Direction": verdict})
    return pd.DataFrame(rows)


def build_scenario(kind: str, ds: Dataset, res: Dict[str, object],
                   magnitude: float = 25.0, target: Optional[str] = None,
                   periods_delay: int = 2) -> Overrides:
    """Turn a named disruption into concrete overrides."""
    ov = Overrides()
    fg = res.get("fg_items", [])
    load = res.get("capacity_load", pd.DataFrame())
    bottleneck = None
    if len(load):
        by_wc = load.groupby("work_centre")["utilisation_pct"].max().sort_values(ascending=False)
        bottleneck = by_wc.index[0] if len(by_wc) else None

    if kind == "Demand surge":
        items = [target] if target and target != "All finished goods" else fg
        for i in items:
            ov.demand_multiplier[i] = 1 + magnitude / 100.0
        ov.notes.append(f"Demand raised by {magnitude:.0f}% on " +
                        (target if target and target != "All finished goods" else "every finished good"))
    elif kind == "Supplier delay":
        item = target or _longest_lead_item(ds, res.get("inventory_params"))
        if item:
            ov.lead_time_delta[item] = periods_delay
            ov.notes.append(f"{item} lead time extended by {periods_delay} period(s)")
    elif kind == "Machine breakdown":
        wc = target or bottleneck
        if wc:
            ov.breakdowns[wc] = magnitude * 60.0
            ov.notes.append(f"{wc} down for {magnitude:.0f} h from the start of the week")
    elif kind == "Rush order":
        item = target or (fg[0] if fg else None)
        if item:
            base = res["forecast"]
            qty = float(base.loc[base["item"] == item, "forecast"].mean()) * magnitude / 100.0
            ov.rush_orders.append({"item": item, "qty": round(qty, 0),
                                   "due_period": res["first_period"], "order_id": "RUSH-001"})
            ov.priority_items.append(item)
            ov.notes.append(f"Rush order for {qty:,.0f} of {item} due in the current period")
    elif kind == "Capacity expansion":
        wc = target or bottleneck
        if wc:
            ov.capacity_multiplier[wc] = 1 + magnitude / 100.0
            ov.notes.append(f"Capacity at {wc} raised by {magnitude:.0f}%")
    return ov


def _longest_lead_item(ds: Dataset, params: Optional[pd.DataFrame] = None) -> Optional[str]:
    """The bought item most worth worrying about: longest lead time first, then
    the largest annual spend among those."""
    inv = ds.get("inventory_master")
    if not len(inv):
        return None
    buy = inv[inv["make_buy"].astype(str).str.lower().str.startswith("buy")] if "make_buy" in inv else inv
    if not len(buy):
        buy = inv
    max_lt = buy["lead_time"].max()
    longest = buy[buy["lead_time"] >= max_lt]
    if params is not None and len(params):
        spend = dict(zip(params["item"], params["annual_usage_value"]))
        longest = longest.assign(_spend=longest["item"].map(lambda i: spend.get(i, 0.0)))
        longest = longest.sort_values("_spend", ascending=False)
    return str(longest.iloc[0]["item"])
