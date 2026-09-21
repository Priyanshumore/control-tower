"""
Modules 8, 9 and 10: order release, executable job list and shop floor
scheduling simulation.

Time model
----------
The simulation clock runs in plant operating minutes from the start of the
first planning period. One period is `PERIOD_MINUTES` of operating time. A
work centre that runs fewer shifts than the plant reference calendar has its
processing time stretched by an availability factor, and every work centre has
its processing time stretched by its efficiency factor, so the detailed
schedule and the rough cut capacity check tell the same story.

    effective minutes = (setup + qty x run) / (efficiency x availability)

Schedule generation is non delay: no machine is left idle while an operation
that could start on it is waiting. Among the operations that could start at the
earliest possible moment, the dispatching rule picks the winner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import capacity as cap_mod

DISPATCH_RULES = ["FCFS", "SPT", "LPT", "EDD", "CR", "SLACK", "Priority"]

RULE_HELP = {
    "FCFS": "First come first served: the job that arrived earliest goes first",
    "SPT": "Shortest processing time: the quickest operation goes first",
    "LPT": "Longest processing time: the longest operation goes first",
    "EDD": "Earliest due date: the job due soonest goes first",
    "CR": "Critical ratio: time remaining divided by work remaining, lowest first",
    "SLACK": "Least slack: due date minus now minus work remaining, lowest first",
    "Priority": "Customer priority first, ties broken by due date",
}


# ---------------------------------------------------------------------------
# Module 8: production order release
# ---------------------------------------------------------------------------
def candidate_orders(mrp_plan: pd.DataFrame, fg_items: List[str],
                     first_period: int, release_window: int = 2) -> pd.DataFrame:
    """8.1 candidate orders from the finished goods planned releases."""
    if len(mrp_plan) == 0:
        return pd.DataFrame(columns=["order_id", "item", "qty", "release_period", "due_period"])
    rel = mrp_plan[(mrp_plan["item"].isin(fg_items)) &
                   (mrp_plan["planned_order_receipt"] > 0)].copy()
    rel["release_period"] = rel["release_period"].fillna(first_period)
    rel["release_period"] = rel["release_period"].clip(lower=first_period)
    window = rel[rel["release_period"] < first_period + release_window]

    rows = []
    for n, (_, r) in enumerate(window.sort_values(["release_period", "item"]).iterrows(), start=1):
        rows.append({
            "order_id": f"PO-{first_period}-{n:03d}",
            "item": r["item"],
            "qty": float(r["planned_order_receipt"]),
            "release_period": int(r["release_period"]),
            "due_period": int(r["period"]),
            "source": "MRP planned order release",
        })
    return pd.DataFrame(rows)


def check_material_availability(orders: pd.DataFrame, bom: pd.DataFrame,
                                mrp_plan: pd.DataFrame,
                                load: pd.DataFrame,
                                routing: pd.DataFrame,
                                first_period: int,
                                material_probe=None,
                                capacity_probe=None) -> pd.DataFrame:
    """8.2 to 8.4: are all components there, and if not what is the constraint?

    The two probes let another party answer the questions instead of reading the
    frames directly, which is how the release step asks the material and capacity
    agents rather than assuming.
    """
    if len(orders) == 0:
        return orders

    avail = {}
    if len(mrp_plan):
        for _, r in mrp_plan.iterrows():
            avail[(r["item"], int(r["period"]))] = {
                "projected": float(r["projected_available"]),
                "status": r["status"],
            }

    over_cells = set()
    if len(load):
        for _, r in load[load["status"] == "Overloaded"].iterrows():
            over_cells.add((r["work_centre"], int(r["period"])))

    def _ask_material(component, period, need):
        if material_probe is not None:
            return material_probe(component, period, need)
        return avail.get((component, int(period)))

    def _ask_capacity(work_centre, period):
        if capacity_probe is not None:
            return bool(capacity_probe(work_centre, period))
        return (work_centre, int(period)) in over_cells

    out = []
    for _, o in orders.iterrows():
        p = int(o["release_period"])
        shortages, waiting = [], []
        comps = bom[bom["parent"] == o["item"]] if len(bom) else pd.DataFrame()
        for _, c in comps.iterrows():
            need = float(o["qty"]) * float(c["qty_per"]) * (1 + float(c.get("scrap_pct", 0) or 0) / 100)
            info = _ask_material(c["component"], p, need)
            if info is None:
                continue
            if info["projected"] < -0.001:
                shortages.append(f"{c['component']} short {abs(info['projected']):,.0f}")
            elif info["status"] == "Past due release":
                waiting.append(c["component"])
            elif info["projected"] < need * 0.0:
                shortages.append(c["component"])

        wcs = set(routing.loc[routing["item"] == o["item"], "work_centre"]) if len(routing) else set()
        cap_blocked = [wc for wc in wcs if _ask_capacity(wc, int(o["due_period"]))]

        if shortages:
            status, constraint = "Held", "Material constrained"
            reason = "; ".join(shortages[:3])
        elif waiting:
            status, constraint = "Held", "Waiting for supplier"
            reason = "Inbound not yet received: " + ", ".join(waiting[:3])
        elif cap_blocked:
            status, constraint = "Released", "Capacity constrained"
            reason = "Runs through an overloaded work centre: " + ", ".join(sorted(cap_blocked))
        else:
            status, constraint = "Released", "None"
            reason = "All components available and capacity clear"

        rec = o.to_dict()
        rec.update({"status": status, "constraint": constraint, "reason": reason})
        out.append(rec)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Module 9: executable job list
# ---------------------------------------------------------------------------
@dataclass
class Op:
    job: str
    item: str
    seq: int
    name: str
    wc: str
    setup: float
    run_per_unit: float
    qty: float
    proc: float = 0.0


@dataclass
class Job:
    job_id: str
    item: str
    qty: float
    arrival: float
    due: float
    priority: int
    ops: List[Op] = field(default_factory=list)

    @property
    def total_proc(self) -> float:
        return sum(o.proc for o in self.ops)


def build_jobs(released: pd.DataFrame, routing: pd.DataFrame,
               machine_capacity: pd.DataFrame, first_period: int,
               period_minutes: float, max_lot: float = 0.0,
               existing_orders: Optional[pd.DataFrame] = None) -> List[Job]:
    """9.3 build the executable job list, optionally splitting large orders."""
    eff = cap_mod.efficiency_map(machine_capacity)
    av = cap_mod.available_hours(machine_capacity)
    # a work centre manned fewer hours than the busiest one has its processing
    # time stretched, because the simulation clock runs continuously
    ref_hours = float(av["machine_hours"].max()) if len(av) else 144.0
    avail_frac = {r["work_centre"]: min(1.0, float(r["machine_hours"]) / ref_hours)
                  for _, r in av.iterrows()} if len(av) else {}

    rows = []
    if existing_orders is not None and len(existing_orders):
        for _, r in existing_orders.iterrows():
            rows.append({
                "order_id": r.get("order_id"), "item": r.get("item"),
                "qty": float(r.get("qty", 0) or 0),
                "release_period": int(r.get("release_period") or first_period),
                "due_period": int(r.get("due_period") or first_period),
                "priority": int(r.get("priority", 1) or 1),
                "source": "Open works order",
            })
    if len(released):
        for _, r in released.iterrows():
            rows.append({
                "order_id": r["order_id"], "item": r["item"], "qty": float(r["qty"]),
                "release_period": int(r["release_period"]),
                "due_period": int(r["due_period"]),
                "priority": int(r.get("priority", 2) or 2),
                "source": r.get("source", "MRP"),
            })

    jobs: List[Job] = []
    for r in rows:
        ops_df = routing[routing["item"] == r["item"]].sort_values("op_seq")
        if not len(ops_df):
            continue
        lots = _split(r["qty"], max_lot)
        for li, q in enumerate(lots, start=1):
            jid = r["order_id"] if len(lots) == 1 else f"{r['order_id']}-{li}"
            arrival = max(0.0, (r["release_period"] - first_period) * period_minutes)
            due = (r["due_period"] - first_period + 1) * period_minutes
            job = Job(jid, r["item"], q, arrival, due, r["priority"])
            for _, op in ops_df.iterrows():
                wc = op["work_centre"]
                factor = max(0.05, eff.get(wc, 1.0) * avail_frac.get(wc, 1.0))
                setup = float(op.get("setup_min", 0) or 0) / factor
                run = float(op.get("run_min", 0) or 0) / factor
                o = Op(jid, r["item"], int(op["op_seq"]), str(op.get("operation", "")),
                       wc, setup, run, q)
                o.proc = setup + run * q
                job.ops.append(o)
            jobs.append(job)
    return jobs


def _split(qty: float, max_lot: float) -> List[float]:
    if not max_lot or max_lot <= 0 or qty <= max_lot:
        return [qty]
    n = int(np.ceil(qty / max_lot))
    base = qty / n
    return [round(base, 2)] * n


def job_list_frame(jobs: List[Job], period_minutes: float, first_period: int) -> pd.DataFrame:
    rows = []
    for j in jobs:
        rows.append({
            "job": j.job_id, "item": j.item, "qty": j.qty,
            "operations": len(j.ops),
            "total_processing_h": round(j.total_proc / 60, 2),
            "arrival_period": first_period + int(j.arrival // period_minutes),
            "due_period": first_period + int(np.ceil(j.due / period_minutes)) - 1,
            "priority": j.priority,
            "work_centres": " -> ".join(o.wc for o in j.ops),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Module 10: dispatching simulation
# ---------------------------------------------------------------------------
def simulate(jobs: List[Job], machine_capacity: pd.DataFrame, rule: str = "FCFS",
             sequence_dependent_setup: bool = True,
             breakdowns: Optional[Dict[str, float]] = None) -> Dict[str, object]:
    """Run one dispatching rule and return the detailed schedule.

    breakdowns maps a work centre to the number of minutes one machine is
    unavailable from time zero, used by the disruption scenarios.
    """
    if not jobs:
        empty = pd.DataFrame()
        return {"schedule": empty, "jobs": empty, "rule": rule}

    n_mach = cap_mod.machines_map(machine_capacity)
    wcs = sorted({o.wc for j in jobs for o in j.ops})
    breakdowns = breakdowns or {}

    machine_free: Dict[str, List[float]] = {}
    machine_last_item: Dict[str, List[Optional[str]]] = {}
    for wc in wcs:
        m = max(1, int(n_mach.get(wc, 1)))
        free = [0.0] * m
        if wc in breakdowns and m > 0:
            free[0] = float(breakdowns[wc])          # one machine comes back later
        machine_free[wc] = free
        machine_last_item[wc] = [None] * m

    state = {j.job_id: {"next_op": 0, "ready": j.arrival, "job": j} for j in jobs}
    remaining = {j.job_id: j.total_proc for j in jobs}
    rows = []
    guard = 0
    total_ops = sum(len(j.ops) for j in jobs)

    while guard < total_ops + 5:
        guard += 1
        # every operation that is next in line for its job
        cands = []
        for jid, st in state.items():
            j = st["job"]
            if st["next_op"] >= len(j.ops):
                continue
            op = j.ops[st["next_op"]]
            free = machine_free[op.wc]
            mi = int(np.argmin(free))
            est = max(st["ready"], free[mi])
            cands.append((est, mi, op, j, st))
        if not cands:
            break

        t_star = min(c[0] for c in cands)
        eligible = [c for c in cands if c[0] <= t_star + 1e-9]
        est, mi, op, j, st = _pick(eligible, rule, t_star, remaining)

        free = machine_free[op.wc]
        mi = int(np.argmin(free))
        start = max(st["ready"], free[mi])
        setup = op.setup
        if sequence_dependent_setup and machine_last_item[op.wc][mi] == op.item:
            setup = 0.0
        proc = setup + op.run_per_unit * op.qty
        end = start + proc

        rows.append({
            "job": j.job_id, "item": j.item, "op_seq": op.seq, "operation": op.name,
            "work_centre": op.wc, "machine": f"{op.wc}-M{mi + 1}",
            "queue_start": st["ready"], "start": start, "end": end,
            "setup_min": round(setup, 2), "run_min": round(proc - setup, 2),
            "proc_min": round(proc, 2), "wait_min": round(start - st["ready"], 2),
            "qty": op.qty, "due": j.due, "priority": j.priority,
        })

        free[mi] = end
        machine_last_item[op.wc][mi] = op.item
        st["ready"] = end
        st["next_op"] += 1
        remaining[j.job_id] = max(0.0, remaining[j.job_id] - op.proc)

    sched = pd.DataFrame(rows)
    jobs_df = _job_summary(jobs, sched)
    return {"schedule": sched, "jobs": jobs_df, "rule": rule}


def _pick(eligible, rule: str, now: float, remaining: Dict[str, float]):
    """Apply the dispatching rule to the operations that can start now."""
    rule = (rule or "FCFS").upper()

    def key(c):
        est, mi, op, j, st = c
        proc = op.proc
        rem = remaining.get(j.job_id, proc)
        if rule == "SPT":
            return (proc, j.due, j.job_id)
        if rule == "LPT":
            return (-proc, j.due, j.job_id)
        if rule == "EDD":
            return (j.due, proc, j.job_id)
        if rule == "CR":
            slack = j.due - now
            cr = slack / rem if rem > 0 else 0.0
            return (cr, j.due, j.job_id)
        if rule == "SLACK":
            return (j.due - now - rem, j.due, j.job_id)
        if rule == "PRIORITY":
            return (j.priority, j.due, j.job_id)
        return (j.arrival, j.job_id)          # FCFS

    return min(eligible, key=key)


def _job_summary(jobs: List[Job], sched: pd.DataFrame) -> pd.DataFrame:
    if len(sched) == 0:
        return pd.DataFrame()
    last = sched.groupby("job")["end"].max()
    first = sched.groupby("job")["start"].min()
    rows = []
    for j in jobs:
        if j.job_id not in last.index:
            continue
        comp = float(last[j.job_id])
        flow = comp - j.arrival
        lateness = comp - j.due
        rows.append({
            "job": j.job_id, "item": j.item, "qty": j.qty,
            "arrival": j.arrival, "start": float(first[j.job_id]),
            "completion": comp, "due": j.due,
            "processing_min": round(j.total_proc, 2),
            "flow_min": round(flow, 2),
            "waiting_min": round(flow - j.total_proc, 2),
            "lateness_min": round(lateness, 2),
            "tardiness_min": round(max(0.0, lateness), 2),
            "on_time": bool(lateness <= 0),
            "priority": j.priority,
        })
    return pd.DataFrame(rows)


def validate_schedule(sched: pd.DataFrame, jobs: List[Job]) -> List[str]:
    """Self check: no machine runs two operations at once and no job overtakes
    its own predecessor."""
    problems = []
    if len(sched) == 0:
        return ["schedule is empty"]

    for m, g in sched.groupby("machine"):
        g = g.sort_values("start")
        ends = g["end"].to_numpy()
        starts = g["start"].to_numpy()
        for i in range(1, len(g)):
            if starts[i] < ends[i - 1] - 1e-6:
                problems.append(f"overlap on {m} between rows {i - 1} and {i}")

    for jid, g in sched.groupby("job"):
        g = g.sort_values("op_seq")
        ends = g["end"].to_numpy()
        starts = g["start"].to_numpy()
        for i in range(1, len(g)):
            if starts[i] < ends[i - 1] - 1e-6:
                problems.append(f"precedence broken on job {jid} at operation {i}")

    op_count = sum(len(j.ops) for j in jobs)
    if len(sched) != op_count:
        problems.append(f"scheduled {len(sched)} operations against {op_count} expected")

    by_job = {j.job_id: j for j in jobs}
    for jid, g in sched.groupby("job"):
        j = by_job.get(jid)
        if j is not None and g["start"].min() < j.arrival - 1e-6:
            problems.append(f"job {jid} started before it arrived")
    return problems
