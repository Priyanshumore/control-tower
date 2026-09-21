"""
Module 12: exception management and managerial decision.

The control tower runs a small set of specialist agents, one per planning
module. Each agent inspects only its own module's output, raises exceptions in
a common shape and proposes an action. A supervisor agent then scores every
exception on severity and financial exposure and returns a single ranked action
list, which is what a plant manager actually wants to look at.

Every recommendation maps to a concrete override, so accepting a recommendation
changes the next planning run rather than just being logged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

SEVERITY_SCORE = {"High": 3.0, "Medium": 2.0, "Low": 1.0}

COLUMNS = ["exception_id", "agent", "item", "period", "type", "severity",
           "detail", "recommended_action", "action_code", "action_payload",
           "exposure_value", "score"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def collect(forecast_exc: pd.DataFrame,
            inventory_exc: pd.DataFrame,
            mps_exc: pd.DataFrame,
            mrp_exc: pd.DataFrame,
            schedule_exc: pd.DataFrame,
            supplier_exc: pd.DataFrame,
            unit_costs: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Supervisor agent: merge, price and rank every exception."""
    unit_costs = unit_costs or {}
    frames = [
        ("Forecast agent", forecast_exc),
        ("Inventory agent", inventory_exc),
        ("Master schedule agent", mps_exc),
        ("Material agent", mrp_exc),
        ("Shop floor agent", schedule_exc),
        ("Supplier agent", supplier_exc),
    ]
    rows = []
    n = 0
    for agent, df in frames:
        if df is None or len(df) == 0:
            continue
        for _, r in df.iterrows():
            n += 1
            item = str(r.get("item", ""))
            sev = str(r.get("severity", "Medium"))
            qty = float(r.get("qty", 0) or 0)
            exposure = qty * float(unit_costs.get(item, 0) or 0)
            action_code, payload = _map_action(str(r.get("type", "")), r)
            rows.append({
                "exception_id": f"EX-{n:03d}",
                "agent": agent,
                "item": item,
                "period": r.get("period", None),
                "type": r.get("type", ""),
                "severity": sev,
                "detail": r.get("detail", ""),
                "recommended_action": r.get("recommended_action", ""),
                "action_code": action_code,
                "action_payload": payload,
                "exposure_value": round(exposure, 2),
                "score": SEVERITY_SCORE.get(sev, 1.0),
            })
    if not rows:
        return _empty()

    out = pd.DataFrame(rows)
    # blend severity with financial exposure so the ranking is not purely categorical
    if out["exposure_value"].max() > 0:
        norm = out["exposure_value"] / out["exposure_value"].max()
        out["score"] = (out["score"] + norm).round(3)
    return out.sort_values(["score", "severity"], ascending=[False, True]).reset_index(drop=True)


def _map_action(exc_type: str, row) -> tuple:
    """Translate an exception into an executable override."""
    t = exc_type.lower()
    item = str(row.get("item", ""))
    period = row.get("period", None)
    if "past due" in t or "release immediately" in t:
        return "expedite", {"item": item, "period": period}
    if "shortage" in t or "below safety stock" in t or "stockout" in t:
        return "expedite", {"item": item, "period": period}
    if "capacity overload" in t:
        return "reschedule", {"work_centre": item, "period": period}
    if "critical component" in t or "supplier" in t:
        return "alternate_supplier", {"item": item}
    if "tardy" in t or "late" in t:
        return "change_priority", {"item": item}
    if "forecast" in t:
        return "review_forecast", {"item": item}
    if "excess" in t:
        return "hold_replenishment", {"item": item}
    return "review", {"item": item}


# ---------------------------------------------------------------------------
# Shop floor and supplier agents
# ---------------------------------------------------------------------------
def schedule_exceptions(jobs_df: pd.DataFrame, util: pd.DataFrame,
                        period_minutes: float, first_period: int,
                        util_high: float = 90.0) -> pd.DataFrame:
    rows = []
    if jobs_df is not None and len(jobs_df):
        late = jobs_df[jobs_df["tardiness_min"] > 0]
        for _, r in late.iterrows():
            rows.append({
                "item": r["item"], "period": first_period,
                "type": "Late job", "severity": "High" if r["tardiness_min"] > period_minutes / 2 else "Medium",
                "detail": f"Job {r['job']} finishes {r['tardiness_min'] / 60:,.1f} h after its due date",
                "recommended_action": "Raise its dispatch priority or move it to an alternate work centre",
                "qty": r.get("qty", 0),
            })
    if util is not None and len(util):
        for _, r in util[util["utilisation_pct"] >= util_high].iterrows():
            rows.append({
                "item": r["work_centre"], "period": first_period,
                "type": "Work centre saturated", "severity": "Medium",
                "detail": f"{r['work_centre']} runs at {r['utilisation_pct']:.0f}% with "
                          f"{r['setup_h']:.1f} h lost to setups",
                "recommended_action": "Group similar products to cut changeovers or add a shift",
            })
    return pd.DataFrame(rows)


def supplier_exceptions(supplier: pd.DataFrame, mrp_plan: pd.DataFrame,
                        first_period: int, reliability_floor: float = 88.0) -> pd.DataFrame:
    rows = []
    if supplier is None or len(supplier) == 0:
        return pd.DataFrame(rows)
    ordered = set(mrp_plan.loc[mrp_plan["planned_order_receipt"] > 0, "item"]) if len(mrp_plan) else set()
    sup = supplier.copy()
    sup["reliability"] = pd.to_numeric(sup["reliability"], errors="coerce").fillna(100)
    for _, r in sup.iterrows():
        if r["item"] not in ordered:
            continue
        if r["reliability"] < reliability_floor:
            rows.append({
                "item": r["item"], "period": first_period,
                "type": "Supplier reliability", "severity": "Medium",
                "detail": f"{r.get('supplier_name') or r['supplier']} delivers on time "
                          f"{r['reliability']:.0f}% of the time",
                "recommended_action": "Qualify a second source or add lead time buffer",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 12.3 and 12.4 manager decision
# ---------------------------------------------------------------------------
@dataclass
class Overrides:
    """Everything a manager decision or a scenario can change about the plan."""
    demand_multiplier: Dict[str, float] = field(default_factory=dict)
    lead_time_override: Dict[str, float] = field(default_factory=dict)
    lead_time_delta: Dict[str, float] = field(default_factory=dict)
    extra_receipts: List[dict] = field(default_factory=list)
    capacity_multiplier: Dict[str, float] = field(default_factory=dict)
    breakdowns: Dict[str, float] = field(default_factory=dict)
    rush_orders: List[dict] = field(default_factory=list)
    priority_items: List[str] = field(default_factory=list)
    dispatch_rule: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not any([self.demand_multiplier, self.lead_time_override, self.lead_time_delta,
                        self.extra_receipts, self.capacity_multiplier, self.breakdowns,
                        self.rush_orders, self.priority_items, self.dispatch_rule])

    def summary(self) -> List[str]:
        out = []
        for k, v in self.demand_multiplier.items():
            out.append(f"Demand for {k} scaled to {v:.0%}")
        for k, v in self.lead_time_delta.items():
            out.append(f"Lead time for {k} moved by {v:+.0f} period(s)")
        for k, v in self.lead_time_override.items():
            out.append(f"Lead time for {k} set to {v:.0f} period(s)")
        for r in self.extra_receipts:
            when = "the first planning period" if r.get("period") is None else f"period {r['period']}"
            out.append(f"Expedited receipt of {r['qty']:,.0f} {r['item']} into {when}")
        for k, v in self.capacity_multiplier.items():
            out.append(f"Capacity at {k} scaled to {v:.0%}")
        for k, v in self.breakdowns.items():
            out.append(f"{k} unavailable for {v / 60:,.1f} h")
        for r in self.rush_orders:
            out.append(f"Rush order {r['qty']:,.0f} of {r['item']} due period {r['due_period']}")
        if self.priority_items:
            out.append("Priority raised for " + ", ".join(self.priority_items))
        if self.dispatch_rule:
            out.append(f"Dispatch rule set to {self.dispatch_rule}")
        return out + self.notes


def _safe_period(value) -> Optional[int]:
    """Exception rows that describe a condition rather than a bucket carry no
    period. Blank, NaN and unparseable values all mean 'as soon as possible'."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def apply_decision(ov: Overrides, exception_row: pd.Series, decision: str,
                   note: str = "", expedite_qty: float = 0.0) -> Overrides:
    """12.3 and 12.4: turn an accepted recommendation into a plan change."""
    if decision == "Reject":
        ov.notes.append(f"{exception_row['exception_id']} rejected. {note}".strip())
        return ov

    code = exception_row.get("action_code", "review")
    item = exception_row.get("item", "")
    period = _safe_period(exception_row.get("period", None))

    if code == "expedite" and item:
        qty = expedite_qty
        # period None means "as early as the plan allows"; the pipeline lands it
        # in the first planning period
        ov.extra_receipts.append({"item": item, "period": period, "qty": float(qty)})
        ov.notes.append(f"{exception_row['exception_id']} accepted: expedite {qty:,.0f} of {item}. {note}".strip())
    elif code == "alternate_supplier" and item:
        ov.lead_time_delta[item] = ov.lead_time_delta.get(item, 0) - 1
        ov.notes.append(f"{exception_row['exception_id']} accepted: alternate supplier for {item}, "
                        f"lead time cut by one period. {note}".strip())
    elif code == "change_priority" and item:
        if item not in ov.priority_items:
            ov.priority_items.append(item)
        ov.notes.append(f"{exception_row['exception_id']} accepted: priority raised for {item}. {note}".strip())
    elif code == "reschedule":
        ov.notes.append(f"{exception_row['exception_id']} accepted: schedule levelled around "
                        f"{item} in period {period}. {note}".strip())
    elif code == "hold_replenishment" and item:
        ov.notes.append(f"{exception_row['exception_id']} accepted: replenishment held for {item}. {note}".strip())
    else:
        ov.notes.append(f"{exception_row['exception_id']} noted for review. {note}".strip())
    return ov
