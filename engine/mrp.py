"""
Module 7: material requirements planning.

Standard time phased record per item, processed in ascending low level code:

  Net requirement    = Gross requirement - Scheduled receipts - Prior available
                       + Safety stock          (floored at zero)
  Planned receipt    = lot size rule applied to the net requirement
  Planned release    = Planned receipt offset backwards by the lead time
  Projected available= Prior available + Scheduled receipts + Planned receipt
                       - Gross requirement

A parent's planned order releases become the gross requirements of its
components, which is why the low level code order matters.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .bom import low_level_codes
from .mps import apply_lot_rule


def run_mrp(mps_grid: pd.DataFrame,
            bom: pd.DataFrame,
            inventory_master: pd.DataFrame,
            scheduled_receipts: pd.DataFrame,
            periods: List[int],
            params: Optional[pd.DataFrame] = None,
            poq_periods: int = 2,
            lead_time_override: Optional[Dict[str, float]] = None,
            include_fg_from_mps: bool = True) -> Dict[str, pd.DataFrame]:
    """Explode the master schedule into a full material plan."""
    inv = inventory_master.set_index("item")
    all_items = list(inventory_master["item"])
    llc = low_level_codes(bom, all_items)
    eoq_map = dict(zip(params["item"], params["eoq"])) if params is not None and len(params) else {}
    lead_time_override = lead_time_override or {}

    sr = {}
    if len(scheduled_receipts):
        for (it, p), q in scheduled_receipts.groupby(["item", "period"])["qty"].sum().items():
            sr[(it, int(p))] = float(q)

    # gross requirements start empty and are filled level by level
    gross: Dict[tuple, float] = {}
    pegging_rows: List[dict] = []

    # the finished goods plan is the master schedule itself
    fg_release: Dict[str, Dict[int, float]] = {}
    if include_fg_from_mps and len(mps_grid):
        for _, r in mps_grid.iterrows():
            q = float(r["mps_qty"])
            if q > 0:
                fg_release.setdefault(r["item"], {})[int(r["period"])] = q

    order = sorted(set(all_items) | set(bom["parent"]) | set(bom["component"]) if len(bom) else set(all_items),
                   key=lambda i: (llc.get(i, 0), i))

    records: List[dict] = []
    releases: Dict[str, Dict[int, float]] = {}

    for item in order:
        row = inv.loc[item] if item in inv.index else None
        lt = float(lead_time_override.get(item,
                   (row.get("lead_time", 1) if row is not None else 1) or 1))
        lt = max(0, int(round(lt)))
        ss = float((row.get("safety_stock", 0) if row is not None else 0) or 0)
        oh = float((row.get("on_hand", 0) if row is not None else 0) or 0)
        rule = str((row.get("lot_rule", "LFL") if row is not None else "LFL") or "LFL")
        lot = float((row.get("lot_size", 0) if row is not None else 0) or 0)
        eoq = float(eoq_map.get(item, 0) or 0)
        item_type = str(row.get("item_type", "") if row is not None else "")

        is_fg = item in fg_release
        bal = oh
        needs = [gross.get((item, p), 0.0) for p in periods]

        for idx, p in enumerate(periods):
            gr = float(gross.get((item, p), 0.0))
            recv = float(sr.get((item, p), 0.0))

            if is_fg:
                # the master schedule already decided the quantity and timing
                planned_receipt = float(fg_release[item].get(p, 0.0))
                net = max(0.0, gr - recv - bal + ss)
            else:
                available_before = bal + recv - gr
                net = max(0.0, ss - available_before)
                planned_receipt = apply_lot_rule(net, rule, lot, eoq, poq_periods,
                                                 needs[idx + 1:]) if net > 0 else 0.0

            proj = bal + recv + planned_receipt - gr
            release_period = p - lt

            records.append({
                "item": item, "level": llc.get(item, 0), "item_type": item_type,
                "period": int(p),
                "gross_requirement": round(gr, 2),
                "scheduled_receipts": round(recv, 2),
                "projected_available_start": round(bal, 2),
                "net_requirement": round(net, 2),
                "planned_order_receipt": round(planned_receipt, 2),
                "planned_order_release": round(planned_receipt, 2) if planned_receipt > 0 else 0.0,
                "release_period": int(release_period) if planned_receipt > 0 else None,
                "projected_available": round(proj, 2),
                "safety_stock": round(ss, 2),
                "lead_time": lt, "lot_rule": rule,
                "status": ("Past due release" if planned_receipt > 0 and release_period < min(periods) - 1
                           else ("Release immediately" if planned_receipt > 0 and release_period < min(periods)
                           else ("Shortage" if proj < -0.001
                                 else ("Below safety stock" if proj < ss - 0.001 else "OK")))),
            })

            if planned_receipt > 0:
                releases.setdefault(item, {})
                releases[item][int(release_period)] = releases[item].get(int(release_period), 0.0) + planned_receipt

            bal = proj

        # push this item's releases down to its components
        if len(bom):
            kids = bom[bom["parent"] == item]
            for rel_p, qty in releases.get(item, {}).items():
                for _, k in kids.iterrows():
                    scrap = float(k.get("scrap_pct", 0) or 0) / 100.0
                    need = qty * float(k["qty_per"]) * (1 + scrap)
                    key = (k["component"], int(rel_p))
                    gross[key] = gross.get(key, 0.0) + need
                    pegging_rows.append({
                        "component": k["component"], "period": int(rel_p),
                        "qty": round(need, 3), "parent": item,
                        "parent_release_qty": round(qty, 2),
                        "qty_per": float(k["qty_per"]),
                        "scrap_pct": float(k.get("scrap_pct", 0) or 0),
                    })

    plan = pd.DataFrame(records)
    plan = plan.sort_values(["level", "item", "period"]).reset_index(drop=True)
    return {
        "plan": plan,
        "pegging": pd.DataFrame(pegging_rows),
        "releases": releases,
        "output": _mrp_output(plan),
    }


def _mrp_output(plan: pd.DataFrame) -> pd.DataFrame:
    """7.4 MRP_Output.csv shape: item, period, GR, SR, PAB, NR, PORct, PORel."""
    if len(plan) == 0:
        return plan
    out = plan[["item", "level", "period", "gross_requirement", "scheduled_receipts",
                "projected_available", "net_requirement", "planned_order_receipt",
                "release_period", "planned_order_release", "status"]].copy()
    out.columns = ["Item", "Level", "Period", "GR", "SR", "PAB", "NR",
                   "PORct", "Release_Period", "PORel", "Status"]
    return out


def mrp_exceptions(plan: pd.DataFrame, inventory_master: pd.DataFrame,
                   supplier: pd.DataFrame, first_period: int,
                   critical_threshold: float = 0.0) -> pd.DataFrame:
    """7.5 material exceptions: past due releases, shortages, critical items."""
    rows = []
    if len(plan) == 0:
        return pd.DataFrame(rows)

    rel = plan[(plan["planned_order_receipt"] > 0) & (plan["release_period"].notna())]
    past = rel[rel["release_period"] < first_period]
    for _, r in past.iterrows():
        late_by = first_period - int(r["release_period"])
        if late_by <= 1:
            rows.append({
                "item": r["item"], "period": int(r["period"]), "type": "Release immediately",
                "severity": "Medium",
                "detail": f"{r['planned_order_receipt']:,.0f} is needed in period {int(r['period'])} "
                          f"and the lead time means it has to go out this period",
                "recommended_action": "Release the order today to protect the due date",
                "qty": float(r["planned_order_receipt"]),
            })
        else:
            rows.append({
                "item": r["item"], "period": int(r["period"]), "type": "Past due release",
                "severity": "High",
                "detail": f"{r['planned_order_receipt']:,.0f} needed in period {int(r['period'])} "
                          f"should have been released in period {int(r['release_period'])}, "
                          f"{late_by} periods ago",
                "recommended_action": "Expedite with the supplier or split the order",
                "qty": float(r["planned_order_receipt"]),
            })

    short = plan[plan["projected_available"] < -0.001]
    for item, g in short.groupby("item"):
        p = int(g["period"].min())
        rows.append({
            "item": item, "period": p, "type": "Material shortage", "severity": "High",
            "detail": f"Projected available goes negative in period {p} "
                      f"({g['projected_available'].min():,.0f})",
            "recommended_action": "Expedite inbound supply or reschedule the parent order",
            "qty": round(abs(float(g["projected_available"].min())), 2),
        })

    # supplier risk: long lead time and weak reliability on items being ordered
    if len(supplier):
        sup = supplier.copy()
        sup["reliability"] = pd.to_numeric(sup["reliability"], errors="coerce").fillna(100)
        risky = sup[(sup["reliability"] < 90) | (pd.to_numeric(sup["lead_time"], errors="coerce") >= 4)]
        ordered = set(rel["item"])
        for _, r in risky.iterrows():
            if r["item"] in ordered:
                rows.append({
                    "item": r["item"], "period": first_period, "type": "Critical component",
                    "severity": "Medium",
                    "detail": f"Supplied by {r.get('supplier_name') or r['supplier']} at "
                              f"{r['reliability']:.0f}% reliability and {r['lead_time']:.0f} period lead time",
                    "recommended_action": "Dual source, hold extra buffer or place the order early",
                })
    return pd.DataFrame(rows)
