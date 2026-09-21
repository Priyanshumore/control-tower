"""
Module 4: inventory planning.

Computes replenishment parameters for every item and projects the finished
goods position over the planning horizon on a "do nothing" basis, so the
exceptions it raises are exactly the gaps the master schedule has to close.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .bom import explosion_order

Z = NormalDist()


def z_for(service_level_pct: float) -> float:
    p = min(max(service_level_pct / 100.0, 0.50), 0.9999)
    return float(Z.inv_cdf(p))


def compute_parameters(inventory_master: pd.DataFrame,
                       demand_history: pd.DataFrame,
                       bom: pd.DataFrame,
                       forecast: pd.DataFrame,
                       service_level: float = 95.0,
                       periods_per_year: int = 52) -> pd.DataFrame:
    """Replenishment parameters per item.

    Independent demand items are sized off their own history. Dependent items
    are sized off exploded average parent usage so that raw and packaging
    materials also get a defensible reorder point.
    """
    z = z_for(service_level)

    stats: Dict[str, Dict[str, float]] = {}
    if len(demand_history):
        g = demand_history.groupby("item")["demand"]
        for item, s in g:
            stats[item] = {"avg": float(s.mean()), "sd": float(s.std(ddof=1)) if len(s) > 1 else 0.0}

    # explode average usage down the BOM so components inherit a demand rate
    usage = {k: v["avg"] for k, v in stats.items()}
    sd_map = {k: v["sd"] for k, v in stats.items()}
    if len(bom):
        for parent in explosion_order(bom, list(inventory_master["item"])):
            pu = usage.get(parent, 0.0)
            psd = sd_map.get(parent, 0.0)
            rows = bom[bom["parent"] == parent]
            for _, r in rows.iterrows():
                q = float(r["qty_per"]) * (1 + float(r.get("scrap_pct", 0) or 0) / 100.0)
                usage[r["component"]] = usage.get(r["component"], 0.0) + pu * q
                # variance adds through the structure
                sd_map[r["component"]] = float(np.sqrt(
                    sd_map.get(r["component"], 0.0) ** 2 + (psd * q) ** 2))

    rows = []
    for _, r in inventory_master.iterrows():
        item = r["item"]
        d = float(usage.get(item, 0.0))
        sd = float(sd_map.get(item, 0.0))
        lt = max(0.0, float(r.get("lead_time", 1) or 0))
        cost = float(r.get("unit_cost", 0) or 0)
        hold_rate = float(r.get("holding_rate", 0.2) or 0.2)
        order_cost = float(r.get("ordering_cost", 0) or 0)
        oh = float(r.get("on_hand", 0) or 0)
        ss_master = float(r.get("safety_stock", 0) or 0)

        ss_calc = z * sd * np.sqrt(max(lt, 1.0))
        rop = d * lt + max(ss_master, 0.0)
        annual_demand = d * periods_per_year
        holding_per_unit = cost * hold_rate
        eoq = float(np.sqrt(2 * annual_demand * order_cost / holding_per_unit)) if (
            annual_demand > 0 and order_cost > 0 and holding_per_unit > 0) else 0.0

        rows.append({
            "item": item,
            "description": r.get("description", ""),
            "item_type": r.get("item_type", ""),
            "uom": r.get("uom", ""),
            "on_hand": oh,
            "avg_demand_per_period": round(d, 2),
            "sd_demand": round(sd, 2),
            "lead_time": lt,
            "safety_stock_master": ss_master,
            "safety_stock_statistical": round(ss_calc, 2),
            "reorder_point": round(rop, 2),
            "eoq": round(eoq, 2),
            "lot_rule": r.get("lot_rule", "LFL"),
            "lot_size": float(r.get("lot_size", 0) or 0),
            "unit_cost": cost,
            "inventory_value": round(oh * cost, 2),
            "annual_usage_value": round(annual_demand * cost, 2),
            "weeks_of_supply": round(oh / d, 2) if d > 0 else np.nan,
            "position_vs_ss": round(oh - ss_master, 2),
            "supplier": r.get("supplier", ""),
        })

    df = pd.DataFrame(rows)
    if len(df):
        df = _abc(df)
    return df


def _abc(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("annual_usage_value", ascending=False).copy()
    total = d["annual_usage_value"].sum()
    if total <= 0:
        d["abc_class"] = "C"
        return d.sort_index()
    d["cum_pct"] = d["annual_usage_value"].cumsum() / total * 100
    d["abc_class"] = np.where(d["cum_pct"] <= 80, "A", np.where(d["cum_pct"] <= 95, "B", "C"))
    return d.sort_index()


def project_inventory(items: list,
                      inventory_master: pd.DataFrame,
                      forecast: pd.DataFrame,
                      customer_orders: pd.DataFrame,
                      scheduled_receipts: pd.DataFrame,
                      periods: list,
                      demand_rule: str = "max") -> pd.DataFrame:
    """Time phased projection with no new supply, to expose the gaps."""
    inv = inventory_master.set_index("item")
    fc = forecast.set_index(["item", "period"])["forecast"].to_dict() if len(forecast) else {}
    if len(customer_orders):
        co = (customer_orders.groupby(["item", "period"])["qty"].sum()).to_dict()
    else:
        co = {}
    sr = (scheduled_receipts.groupby(["item", "period"])["qty"].sum().to_dict()
          if len(scheduled_receipts) else {})

    rows = []
    for item in items:
        if item not in inv.index:
            continue
        bal = float(inv.loc[item].get("on_hand", 0) or 0)
        ss = float(inv.loc[item].get("safety_stock", 0) or 0)
        for p in periods:
            f = float(fc.get((item, p), 0.0))
            o = float(co.get((item, p), 0.0))
            if demand_rule == "forecast":
                dem = f
            elif demand_rule == "orders":
                dem = o
            elif demand_rule == "sum":
                dem = f + o
            else:
                dem = max(f, o)
            rec = float(sr.get((item, p), 0.0))
            begin = bal
            end = begin + rec - dem
            status = "OK"
            if end < 0:
                status = "Stockout"
            elif end < ss:
                status = "Below safety stock"
            rows.append({
                "item": item, "period": int(p),
                "opening": round(begin, 2), "forecast": round(f, 2),
                "customer_orders": round(o, 2), "gross_requirement": round(dem, 2),
                "scheduled_receipts": round(rec, 2),
                "closing": round(end, 2), "safety_stock": round(ss, 2),
                "status": status,
            })
            bal = end
    return pd.DataFrame(rows)


def inventory_exceptions(params: pd.DataFrame, projection: pd.DataFrame,
                         excess_weeks: float = 6.0) -> pd.DataFrame:
    """4.4 identify inventory exceptions."""
    rows = []
    for _, r in params.iterrows():
        if r["on_hand"] < r["safety_stock_master"]:
            gap = r["safety_stock_master"] - r["on_hand"]
            rows.append({
                "item": r["item"], "type": "Below safety stock",
                "severity": "High" if r["on_hand"] < 0.5 * r["safety_stock_master"] else "Medium",
                "detail": f"On hand {r['on_hand']:,.0f} against safety stock "
                          f"{r['safety_stock_master']:,.0f}, short by {gap:,.0f} {r['uom']}",
                "recommended_action": "Expedite the open order or raise a replenishment now",
                "qty": round(gap, 2),
            })
        elif r["on_hand"] <= r["reorder_point"] and r["reorder_point"] > 0:
            rows.append({
                "item": r["item"], "type": "At reorder point",
                "severity": "Low",
                "detail": f"On hand {r['on_hand']:,.0f} has reached the reorder point "
                          f"{r['reorder_point']:,.0f}",
                "recommended_action": "Release a planned order this period",
            })
        if (r["weeks_of_supply"] is not None and not pd.isna(r["weeks_of_supply"])
                and r["weeks_of_supply"] > excess_weeks and r["avg_demand_per_period"] > 0):
            rows.append({
                "item": r["item"], "type": "Excess inventory",
                "severity": "Low",
                "detail": f"{r['weeks_of_supply']:.1f} weeks of supply on hand, worth "
                          f"INR {r['inventory_value']:,.0f}",
                "recommended_action": "Hold replenishment and let the cover run down",
            })

    if len(projection):
        for item, g in projection.groupby("item"):
            bad = g[g["status"] == "Stockout"]
            if len(bad):
                first = int(bad["period"].min())
                rows.append({
                    "item": item, "type": "Projected stockout",
                    "severity": "High",
                    "detail": f"Projected to run out in period {first} with no new supply, "
                              f"short {abs(bad['closing'].min()):,.0f}",
                    "recommended_action": "Cover the gap in the master production schedule",
                    "qty": round(abs(float(bad["closing"].min())), 2),
                })
            else:
                low = g[g["status"] == "Below safety stock"]
                if len(low):
                    rows.append({
                        "item": item, "type": "Projected safety stock breach",
                        "severity": "Medium",
                        "detail": f"Cover falls below safety stock from period {int(low['period'].min())}",
                        "recommended_action": "Bring the master schedule forward for this item",
                    })
    return pd.DataFrame(rows)
