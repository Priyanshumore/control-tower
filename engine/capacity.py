"""
Module 9 support: work centre capacity.

Available hours per period = machines x hours per shift x shifts per day
                             x days per week x efficiency.

The same efficiency factor inflates processing time on the shop floor, so the
rough cut load and the detailed schedule stay consistent with each other.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd


def available_hours(machine_capacity: pd.DataFrame) -> pd.DataFrame:
    if len(machine_capacity) == 0:
        # keep the full column shape so every downstream map still works
        return pd.DataFrame(columns=["work_centre", "description", "num_machines",
                                     "hours_per_shift", "shifts_per_day", "days_per_week",
                                     "efficiency", "eff_frac", "machine_hours",
                                     "gross_hours", "available_hours"])
    df = machine_capacity.copy()
    for c, d in [("num_machines", 1), ("hours_per_shift", 8), ("shifts_per_day", 1),
                 ("days_per_week", 6), ("efficiency", 100)]:
        if c not in df.columns:
            df[c] = d
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(d)
    df["efficiency"] = df["efficiency"].where(df["efficiency"] > 0, 100)
    # a value such as 0.9 is read as a fraction, 90 as a percentage
    df["eff_frac"] = df["efficiency"].map(lambda v: v if v <= 1.5 else v / 100.0)
    # hours one machine is manned per period, independent of how many there are
    df["machine_hours"] = df["hours_per_shift"] * df["shifts_per_day"] * df["days_per_week"]
    df["gross_hours"] = df["num_machines"] * df["machine_hours"]
    df["available_hours"] = df["gross_hours"] * df["eff_frac"]
    if "description" not in df.columns:
        df["description"] = df["work_centre"]
    return df[["work_centre", "description", "num_machines", "hours_per_shift",
               "shifts_per_day", "days_per_week", "efficiency", "eff_frac",
               "machine_hours", "gross_hours", "available_hours"]]


def capacity_map(machine_capacity: pd.DataFrame) -> Dict[str, float]:
    av = available_hours(machine_capacity)
    if not len(av) or "available_hours" not in av.columns:
        return {}
    return dict(zip(av["work_centre"], av["available_hours"]))


def efficiency_map(machine_capacity: pd.DataFrame) -> Dict[str, float]:
    av = available_hours(machine_capacity)
    if not len(av) or "eff_frac" not in av.columns:
        return {}
    return dict(zip(av["work_centre"], av["eff_frac"]))


def machines_map(machine_capacity: pd.DataFrame) -> Dict[str, int]:
    av = available_hours(machine_capacity)
    return {k: max(1, int(v)) for k, v in zip(av["work_centre"], av["num_machines"])}


def load_from_schedule(schedule: pd.DataFrame, routing: pd.DataFrame,
                       machine_capacity: pd.DataFrame,
                       qty_col: str = "mps_qty") -> pd.DataFrame:
    """Rough cut capacity: hours loaded on each work centre in each period."""
    if len(schedule) == 0 or len(routing) == 0:
        return pd.DataFrame(columns=["work_centre", "period", "load_hours",
                                     "available_hours", "utilisation_pct", "status"])
    eff = efficiency_map(machine_capacity)
    cap = capacity_map(machine_capacity)

    rows = []
    for _, s in schedule.iterrows():
        q = float(s.get(qty_col, 0) or 0)
        if q <= 0:
            continue
        ops = routing[routing["item"] == s["item"]]
        for _, op in ops.iterrows():
            wc = op["work_centre"]
            e = eff.get(wc, 1.0)
            minutes = (float(op.get("setup_min", 0) or 0)
                       + q * float(op.get("run_min", 0) or 0)) / max(e, 0.01)
            rows.append({"work_centre": wc, "period": int(s["period"]),
                         "item": s["item"], "load_hours": minutes / 60.0})

    if not rows:
        return pd.DataFrame(columns=["work_centre", "period", "load_hours",
                                     "available_hours", "utilisation_pct", "status"])

    detail = pd.DataFrame(rows)
    agg = detail.groupby(["work_centre", "period"], as_index=False)["load_hours"].sum()
    agg["available_hours"] = agg["work_centre"].map(cap).fillna(0.0)
    agg["utilisation_pct"] = agg.apply(
        lambda r: (r["load_hours"] / r["available_hours"] * 100) if r["available_hours"] > 0 else 0.0,
        axis=1)
    agg["status"] = agg["utilisation_pct"].map(
        lambda u: "Overloaded" if u > 100.5 else ("Tight" if u > 90 else "OK"))
    agg["load_hours"] = agg["load_hours"].round(2)
    agg["available_hours"] = agg["available_hours"].round(2)
    agg["utilisation_pct"] = agg["utilisation_pct"].round(1)
    return agg.sort_values(["period", "work_centre"])
