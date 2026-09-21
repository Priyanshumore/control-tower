"""
Module 11: performance measures and rule comparison.

  Makespan          last completion minus the first arrival
  Flow time         completion minus arrival, averaged over jobs
  Waiting time      flow time minus processing time
  Lateness          completion minus due date, signed
  Tardiness         lateness floored at zero
  On time delivery  share of jobs finishing on or before the due date
  Utilisation       busy minutes divided by machine minutes over the makespan
  Throughput        jobs completed per period of operating time
  Average WIP       total flow time divided by makespan (Little's law)
  Takt time         operating time available divided by units required
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import capacity as cap_mod
from .scheduling import Job, simulate


def schedule_kpis(sched: pd.DataFrame, jobs_df: pd.DataFrame,
                  machine_capacity: pd.DataFrame,
                  period_minutes: float,
                  demand_units: Optional[float] = None) -> Dict[str, float]:
    if len(sched) == 0 or len(jobs_df) == 0:
        return {}

    start = float(min(jobs_df["arrival"].min(), sched["start"].min()))
    makespan = float(sched["end"].max() - start)
    n = len(jobs_df)

    n_mach = cap_mod.machines_map(machine_capacity)
    used_wcs = sorted(sched["work_centre"].unique())
    machine_minutes = sum(max(1, int(n_mach.get(wc, 1))) for wc in used_wcs) * makespan
    busy = float(sched["proc_min"].sum())

    total_flow = float(jobs_df["flow_min"].sum())
    units = float(jobs_df["qty"].sum())

    return {
        "Makespan_h": round(makespan / 60, 2),
        "Avg_Flow_Time_h": round(float(jobs_df["flow_min"].mean()) / 60, 2),
        "Max_Flow_Time_h": round(float(jobs_df["flow_min"].max()) / 60, 2),
        "Avg_Waiting_Time_h": round(float(jobs_df["waiting_min"].mean()) / 60, 2),
        "Avg_Lateness_h": round(float(jobs_df["lateness_min"].mean()) / 60, 2),
        "Avg_Tardiness_h": round(float(jobs_df["tardiness_min"].mean()) / 60, 2),
        "Max_Tardiness_h": round(float(jobs_df["tardiness_min"].max()) / 60, 2),
        "Jobs_Tardy": int((jobs_df["tardiness_min"] > 0).sum()),
        "Jobs_Total": int(n),
        "OTD_Pct": round(float(jobs_df["on_time"].mean()) * 100, 1),
        "Utilisation_Pct": round(busy / machine_minutes * 100, 1) if machine_minutes > 0 else 0.0,
        "Throughput_Jobs_Per_Period": round(n / (makespan / period_minutes), 2) if makespan > 0 else 0.0,
        "Throughput_Units_Per_Period": round(units / (makespan / period_minutes), 1) if makespan > 0 else 0.0,
        "Avg_WIP_Jobs": round(total_flow / makespan, 2) if makespan > 0 else 0.0,
        "Takt_Time_Min_Per_Unit": round(makespan / demand_units, 3) if demand_units else (
            round(makespan / units, 3) if units > 0 else 0.0),
        "Total_Setup_h": round(float(sched["setup_min"].sum()) / 60, 2),
        "Setup_Share_Pct": round(float(sched["setup_min"].sum()) / busy * 100, 1) if busy > 0 else 0.0,
    }


def utilisation_by_wc(sched: pd.DataFrame, machine_capacity: pd.DataFrame) -> pd.DataFrame:
    if len(sched) == 0:
        return pd.DataFrame(columns=["work_centre", "busy_h", "utilisation_pct"])
    n_mach = cap_mod.machines_map(machine_capacity)
    start = float(sched["start"].min())
    makespan = float(sched["end"].max() - start)
    rows = []
    for wc, g in sched.groupby("work_centre"):
        m = max(1, int(n_mach.get(wc, 1)))
        busy = float(g["proc_min"].sum())
        rows.append({
            "work_centre": wc, "machines": m,
            "busy_h": round(busy / 60, 2),
            "setup_h": round(float(g["setup_min"].sum()) / 60, 2),
            "idle_h": round(max(0.0, (makespan * m - busy)) / 60, 2),
            "utilisation_pct": round(busy / (makespan * m) * 100, 1) if makespan > 0 else 0.0,
            "operations": int(len(g)),
        })
    return pd.DataFrame(rows).sort_values("utilisation_pct", ascending=False)


def compare_rules(jobs: List[Job], machine_capacity: pd.DataFrame, rules: List[str],
                  period_minutes: float, sequence_dependent_setup: bool = True,
                  breakdowns: Optional[Dict[str, float]] = None) -> Dict[str, object]:
    """11.1 to 11.3: run every rule and pick the best on each criterion."""
    results, kpi_rows, detail = {}, [], {}
    for rule in rules:
        r = simulate(jobs, machine_capacity, rule, sequence_dependent_setup, breakdowns)
        k = schedule_kpis(r["schedule"], r["jobs"], machine_capacity, period_minutes)
        if not k:
            continue
        row = {"rule": rule}
        row.update(k)
        kpi_rows.append(row)
        results[rule] = r
        detail[rule] = r["schedule"]

    table = pd.DataFrame(kpi_rows)
    best = {}
    if len(table):
        lower_better = ["Makespan_h", "Avg_Flow_Time_h", "Avg_Waiting_Time_h",
                        "Avg_Tardiness_h", "Max_Tardiness_h", "Jobs_Tardy",
                        "Avg_WIP_Jobs", "Total_Setup_h"]
        higher_better = ["OTD_Pct", "Utilisation_Pct", "Throughput_Jobs_Per_Period",
                         "Throughput_Units_Per_Period"]
        for c in lower_better:
            if c in table.columns:
                best[c] = table.loc[table[c].idxmin(), "rule"]
        for c in higher_better:
            if c in table.columns:
                best[c] = table.loc[table[c].idxmax(), "rule"]
    return {"table": table, "best": best, "runs": results, "schedules": detail}


def overall_recommendation(table: pd.DataFrame,
                           weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Rank the rules on a weighted normalised score, so one rule can be named."""
    if len(table) == 0:
        return table
    weights = weights or {"Avg_Tardiness_h": 0.3, "OTD_Pct": 0.25,
                          "Avg_Flow_Time_h": 0.2, "Makespan_h": 0.15,
                          "Utilisation_Pct": 0.1}
    higher_better = {"OTD_Pct", "Utilisation_Pct", "Throughput_Jobs_Per_Period",
                     "Throughput_Units_Per_Period"}
    t = table.copy()
    score = pd.Series(0.0, index=t.index)
    for col, w in weights.items():
        if col not in t.columns:
            continue
        v = t[col].astype(float)
        rng = v.max() - v.min()
        norm = pd.Series(1.0, index=t.index) if rng == 0 else (
            (v - v.min()) / rng if col in higher_better else (v.max() - v) / rng)
        score += w * norm
    t["score"] = score.round(3)
    return t.sort_values("score", ascending=False)


def to_gantt(sched: pd.DataFrame, origin: pd.Timestamp, period_minutes: float) -> pd.DataFrame:
    """Map simulation minutes onto wall clock time for the Gantt chart."""
    if len(sched) == 0:
        return sched
    g = sched.copy()
    g["Start"] = origin + pd.to_timedelta(g["start"], unit="m")
    g["Finish"] = origin + pd.to_timedelta(g["end"], unit="m")
    g["Resource"] = g["machine"]
    g["Task"] = g["job"] + "  " + g["item"]
    g["Duration_h"] = (g["end"] - g["start"]) / 60
    return g
