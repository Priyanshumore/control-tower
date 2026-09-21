"""
Module 5: master production schedule.

  PAB(t) = PAB(t-1) + MPS(t) + scheduled receipts(t) - gross requirement(t)

Gross requirement follows the demand time fence convention: inside the fence
the schedule is driven by confirmed customer orders only, outside it by the
greater of forecast and orders, so the same demand is never counted twice.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import capacity as cap_mod

LOT_RULES = ["LFL", "FOQ", "EOQ", "POQ"]


def apply_lot_rule(need: float, rule: str, lot_size: float, eoq: float,
                   poq_periods: int = 2, future_need: Optional[List[float]] = None) -> float:
    """Convert a net requirement into an order quantity."""
    rule = (rule or "LFL").upper().strip()
    if need <= 0:
        return 0.0
    if rule in ("FOQ", "FIXED", "FIXEDORDERQTY"):
        if lot_size and lot_size > 0:
            return float(np.ceil(need / lot_size) * lot_size)
        return need
    if rule == "EOQ":
        q = eoq if eoq and eoq > 0 else lot_size
        if q and q > 0:
            return float(np.ceil(need / q) * q)
        return need
    if rule in ("POQ", "PERIODORDERQTY"):
        total = need + sum((future_need or [])[:max(0, poq_periods - 1)])
        return float(max(need, total))
    return float(need)          # lot for lot


def run_mps(items: List[str],
            forecast: pd.DataFrame,
            customer_orders: pd.DataFrame,
            inventory_master: pd.DataFrame,
            scheduled_receipts: pd.DataFrame,
            periods: List[int],
            params: pd.DataFrame,
            demand_time_fence: int = 2,
            poq_periods: int = 2,
            demand_rule: str = "max",
            lot_rule_override: Optional[Dict[str, str]] = None,
            manual_schedule: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
    """Build the master schedule for every finished good."""
    inv = inventory_master.set_index("item")
    eoq_map = dict(zip(params["item"], params["eoq"])) if len(params) else {}
    fc = forecast.set_index(["item", "period"])["forecast"].to_dict() if len(forecast) else {}
    co = (customer_orders.groupby(["item", "period"])["qty"].sum().to_dict()
          if len(customer_orders) else {})
    sr = (scheduled_receipts.groupby(["item", "period"])["qty"].sum().to_dict()
          if len(scheduled_receipts) else {})
    lot_rule_override = lot_rule_override or {}

    manual = (manual_schedule.groupby(["item", "period"])["mps_qty"].sum().to_dict()
              if manual_schedule is not None and len(manual_schedule) else {})
    if any(not np.isfinite(v) or v < 0 for v in manual.values()):
        raise ValueError("Manual schedule quantities must be finite and nonnegative")
    p0 = min(periods)
    grid_rows, atp_rows = [], []

    for item in items:
        if item not in inv.index:
            continue
        row = inv.loc[item]
        bal = float(row.get("on_hand", 0) or 0)
        ss = float(row.get("safety_stock", 0) or 0)
        rule = lot_rule_override.get(item) or str(row.get("lot_rule", "LFL") or "LFL")
        lot = float(row.get("lot_size", 0) or 0)
        eoq = float(eoq_map.get(item, 0) or 0)

        # future needs are used by the period order quantity rule
        needs = []
        for p in periods:
            f = float(fc.get((item, p), 0.0))
            o = float(co.get((item, p), 0.0))
            needs.append(_gross(f, o, p, p0, demand_time_fence, demand_rule))

        mps_by_period: Dict[int, float] = {}
        for idx, p in enumerate(periods):
            f = float(fc.get((item, p), 0.0))
            o = float(co.get((item, p), 0.0))
            gr = needs[idx]
            recv = float(sr.get((item, p), 0.0))
            opening = bal
            pab_before = opening + recv - gr
            need = ss - pab_before
            qty = apply_lot_rule(need, rule, lot, eoq, poq_periods, needs[idx + 1:]) if need > 0 else 0.0
            qty = float(manual.get((item, p), qty))
            pab = pab_before + qty
            mps_by_period[p] = qty

            grid_rows.append({
                "item": item, "period": int(p),
                "forecast": round(f, 2), "customer_orders": round(o, 2),
                "gross_requirement": round(gr, 2),
                "demand_basis": ("Orders (inside fence)" if o > 0 else "Forecast (no orders)") if p < p0 + demand_time_fence else {"max": "Max of forecast and orders", "sum": "Forecast plus orders", "orders": "Orders only", "forecast": "Forecast only"}.get(demand_rule, demand_rule),
                "opening_pab": round(opening, 2),
                "scheduled_receipts": round(recv, 2),
                "mps_qty": round(qty, 2),
                "pab": round(pab, 2),
                "safety_stock": round(ss, 2),
                "lot_rule": rule,
                "status": "Below safety stock" if pab < ss - 0.001 else ("Negative" if pab < 0 else "OK"),
            })
            bal = pab

        # available to promise, looking forward to the next period carrying an MPS
        supply_periods = [p for p in periods if mps_by_period.get(p, 0) > 0]
        for i, p in enumerate(periods):
            if p == p0 or mps_by_period.get(p, 0) > 0:
                nxt = next((q for q in supply_periods if q > p), None)
                span = [q for q in periods if q >= p and (nxt is None or q < nxt)]
                orders_in_span = sum(float(co.get((item, q), 0.0)) for q in span)
                supply = mps_by_period.get(p, 0.0) + float(sr.get((item, p), 0.0))
                if p == p0:
                    supply += float(inv.loc[item].get("on_hand", 0) or 0)
                atp_rows.append({"item": item, "period": int(p),
                                 "supply": round(supply, 2),
                                 "committed_orders": round(orders_in_span, 2),
                                 "atp": round(supply - orders_in_span, 2)})

    grid = pd.DataFrame(grid_rows)
    atp = pd.DataFrame(atp_rows)
    return {"grid": grid, "atp": atp}


def _gross(forecast: float, orders: float, period: int, p0: int,
           fence: int, rule: str) -> float:
    if period < p0 + fence:
        return orders if orders > 0 else forecast
    if rule == "forecast":
        return forecast
    if rule == "orders":
        return orders
    if rule == "sum":
        return forecast + orders
    return max(forecast, orders)


def check_feasibility(grid: pd.DataFrame, routing: pd.DataFrame,
                      machine_capacity: pd.DataFrame) -> Dict[str, object]:
    """5.4 and 5.5: is the master schedule executable?"""
    load = cap_mod.load_from_schedule(grid, routing, machine_capacity, qty_col="mps_qty")
    issues = []

    neg = grid[grid["pab"] < -0.001]
    for _, r in neg.iterrows():
        issues.append({"item": r["item"], "period": int(r["period"]), "type": "Negative PAB",
                       "severity": "High",
                       "detail": f"Projected available balance {r['pab']:,.0f} in period {int(r['period'])}",
                       "recommended_action": "Increase the master schedule or pull demand out"})

    viol = grid[(grid["pab"] < grid["safety_stock"] - 0.001) & (grid["pab"] >= -0.001)]
    for _, r in viol.iterrows():
        issues.append({"item": r["item"], "period": int(r["period"]),
                       "type": "Safety stock violation", "severity": "Medium",
                       "detail": f"PAB {r['pab']:,.0f} sits below safety stock {r['safety_stock']:,.0f}",
                       "recommended_action": "Add a lot in an earlier period"})

    if len(load):
        for _, r in load[load["status"] == "Overloaded"].iterrows():
            issues.append({"item": r["work_centre"], "period": int(r["period"]),
                           "type": "Capacity overload", "severity": "High",
                           "detail": f"{r['work_centre']} loaded to {r['utilisation_pct']:.0f}% "
                                     f"({r['load_hours']:,.0f} h against {r['available_hours']:,.0f} h)",
                           "recommended_action": "Level the schedule, add a shift or offload to an alternate line"})

    feasible = len(issues) == 0
    return {"feasible": feasible, "issues": pd.DataFrame(issues), "load": load}


class CapacityLedger:
    """Answers "how many hours are still free" and records what gets taken.

    The master schedule does not own capacity; the capacity function does. This
    small object is the contract between them, so the levelling loop can either
    keep its own book (the default) or defer to a capacity agent that holds the
    real one and logs every question and answer.
    """

    def __init__(self, spare: Dict[tuple, float]):
        self._spare = dict(spare)

    def spare(self, work_centre: str, period: int) -> float:
        return float(self._spare.get((work_centre, int(period)), 0.0))

    def reserve(self, work_centre: str, period: int, hours: float) -> None:
        key = (work_centre, int(period))
        self._spare[key] = self._spare.get(key, 0.0) - float(hours)

    def release(self, work_centre: str, period: int, hours: float) -> None:
        self.reserve(work_centre, period, -float(hours))


def level_capacity(grid: pd.DataFrame, routing: pd.DataFrame,
                   machine_capacity: pd.DataFrame, max_passes: int = 30,
                   ledger_factory=None) -> Dict[str, object]:
    """5.5 feedback loop: build ahead into earlier periods to clear overloads.

    Quantity is only ever moved earlier, never later, so customer service is
    protected. For each overloaded work centre and period the routine looks
    backwards for the earliest period that has genuine spare hours on that work
    centre, and moves only as much as both the excess and the spare will take.
    Anything that cannot be moved is reported as a residual overload, which is
    the honest answer when the horizon is short on total capacity.
    """
    g = grid.copy()
    log = []
    periods = sorted(g["period"].unique())
    eff = cap_mod.efficiency_map(machine_capacity)

    for _ in range(max_passes):
        load = cap_mod.load_from_schedule(g, routing, machine_capacity, qty_col="mps_qty")
        if len(load) == 0:
            break
        over = load[load["status"] == "Overloaded"].sort_values("period")
        if len(over) == 0:
            break

        # spare hours by work centre and period, including periods carrying no load
        spare = {}
        for _, r in load.iterrows():
            spare[(r["work_centre"], int(r["period"]))] = r["available_hours"] - r["load_hours"]
        cap = cap_mod.capacity_map(machine_capacity)
        for wc in cap:
            for p in periods:
                spare.setdefault((wc, p), cap[wc])
        ledger = ledger_factory(spare) if ledger_factory else CapacityLedger(spare)

        moved_any = False
        for _, o in over.iterrows():
            wc, p = o["work_centre"], int(o["period"])
            excess_h = o["load_hours"] - o["available_hours"]
            earlier = [q for q in periods if q < p and ledger.spare(wc, q) > 0.5]
            if excess_h <= 0.01 or not earlier:
                continue
            items_on_wc = set(routing.loc[routing["work_centre"] == wc, "item"])
            cand = g[(g["period"] == p) & (g["item"].isin(items_on_wc)) & (g["mps_qty"] > 0)]
            cand = cand.sort_values("mps_qty", ascending=False)

            for target in sorted(earlier, reverse=True):     # nearest earlier period first
                if excess_h <= 0.01:
                    break
                for _, c in cand.iterrows():
                    if excess_h <= 0.01 or ledger.spare(wc, target) <= 0.5:
                        break
                    run = routing[(routing["item"] == c["item"]) & (routing["work_centre"] == wc)]
                    if not len(run):
                        continue
                    per_unit_h = float(run.iloc[0].get("run_min", 0) or 0) / 60.0 / max(eff.get(wc, 1.0), 0.01)
                    if per_unit_h <= 0:
                        continue
                    gi = g[(g["item"] == c["item"]) & (g["period"] == p)].index
                    gj = g[(g["item"] == c["item"]) & (g["period"] == target)].index
                    if not len(gi) or not len(gj):
                        continue
                    available_now = float(g.loc[gi, "mps_qty"].iloc[0])
                    move_qty = float(np.floor(min(available_now,
                                                  excess_h / per_unit_h,
                                                  ledger.spare(wc, target) / per_unit_h)))
                    # moving this item also loads every other work centre on its
                    # routing, so the target period has to absorb all of them
                    item_ops = routing[routing["item"] == c["item"]]
                    per_unit_other = {}
                    for _, oprow in item_ops.iterrows():
                        owc = oprow["work_centre"]
                        h = float(oprow.get("run_min", 0) or 0) / 60.0 / max(eff.get(owc, 1.0), 0.01)
                        if h > 0:
                            per_unit_other[owc] = per_unit_other.get(owc, 0.0) + h
                    for owc, h in per_unit_other.items():
                        room = ledger.spare(owc, target)
                        move_qty = min(move_qty, max(0.0, room) / h)
                    move_qty = float(np.floor(move_qty))
                    if move_qty < 1:
                        continue
                    g.loc[gi, "mps_qty"] = g.loc[gi, "mps_qty"].to_numpy() - move_qty
                    g.loc[gj, "mps_qty"] = g.loc[gj, "mps_qty"].to_numpy() + move_qty
                    log.append({"item": c["item"], "work_centre": wc, "from_period": p,
                                "to_period": target, "qty_moved": move_qty,
                                "reason": f"{wc} over by {excess_h:,.1f} h in period {p}"})
                    excess_h -= move_qty * per_unit_h
                    for owc, h in per_unit_other.items():
                        ledger.reserve(owc, target, move_qty * h)
                        ledger.release(owc, p, move_qty * h)
                    moved_any = True

        if not moved_any:
            break
        g = recompute_pab(g)

    return {"grid": recompute_pab(g), "moves": pd.DataFrame(log)}


def recompute_pab(grid: pd.DataFrame) -> pd.DataFrame:
    """Re-run the balance recursion after the schedule has been edited."""
    out = grid.copy().sort_values(["item", "period"])
    for item, g in out.groupby("item"):
        idx = g.index
        opening = float(g.iloc[0]["opening_pab"])
        bal = opening
        for i in idx:
            r = out.loc[i]
            pab = bal + float(r["mps_qty"]) + float(r["scheduled_receipts"]) - float(r["gross_requirement"])
            out.loc[i, "opening_pab"] = round(bal, 2)
            out.loc[i, "pab"] = round(pab, 2)
            out.loc[i, "status"] = ("Below safety stock" if pab < float(r["safety_stock"]) - 0.001
                                    else ("Negative" if pab < 0 else "OK"))
            bal = pab
    return out.sort_values(["item", "period"]).reset_index(drop=True)
