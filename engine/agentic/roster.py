"""
The specialist agents.

Three of them are not just commentators. During a planning run the capacity
agent hands out the hours, the material agent answers component availability and
the supplier agent is consulted on inbound risk, all through the bus, so the
transcript is the record of decisions actually taken.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .. import mps as mps_mod
from .base import Agent, fmt, money


def _rows(df: pd.DataFrame, n: int = 8) -> List[dict]:
    return df.head(n).to_dict("records") if len(df) else []


def _pick(df: pd.DataFrame, col: str, value) -> pd.DataFrame:
    return df[df[col] == value] if len(df) and col in df.columns else pd.DataFrame()


# ---------------------------------------------------------------------------
class DataAgent(Agent):
    name = "Data"
    role = "Keeps the dataset, the column mapping and the data quality notes"
    owns = "Steps 1 and 2"
    skills = ["data.summary", "data.issues", "data.item", "data.tables"]

    def __init__(self, bus, board, ds):
        super().__init__(bus, board)
        self.ds = ds

    def do_summary(self, payload, sender):
        ds = self.ds
        return {"summary": f"{len(ds.tables)} tables recognised in {ds.name}, "
                           f"{len(ds.issues)} data note(s)",
                "tables": {t: len(ds.tables[t]) for t in ds.tables},
                "unmatched": ds.unmatched, "issues": ds.issues[:8]}

    def do_tables(self, payload, sender):
        return self.do_summary(payload, sender)

    def do_issues(self, payload, sender):
        return {"summary": f"{len(self.ds.issues)} data quality note(s)",
                "issues": self.ds.issues}

    def do_item(self, payload, sender):
        item = payload.get("item")
        inv = self.ds.get("inventory_master")
        row = _pick(inv, "item", item)
        if not len(row):
            return {"ok": False, "summary": f"{item} is not in the item master"}
        r = row.iloc[0].to_dict()
        return {"summary": f"{item} is a {r.get('item_type', 'item')} "
                           f"({r.get('make_buy', '')}), lead time {fmt(r.get('lead_time'))}",
                "master": r}


# ---------------------------------------------------------------------------
class ForecastAgent(Agent):
    name = "Forecast"
    role = "Fits six methods per product and picks the one that tests best"
    owns = "Step 3"
    skills = ["forecast.summary", "forecast.item", "forecast.accuracy", "forecast.demand"]

    def do_summary(self, payload, sender):
        sel = self.plan.get("forecast_run", {}).get("selected", pd.DataFrame())
        s = self.summary
        if not len(sel):
            return {"summary": "no forecast has been produced"}
        worst = sel.sort_values("MAPE", ascending=False).iloc[0]
        return {"summary": f"average error {fmt(s.get('forecast_mape'), '%', 2)}, "
                           f"weakest is {worst['item']} at {fmt(worst['MAPE'], '%', 1)}",
                "selected": _rows(sel), "mape": s.get("forecast_mape")}

    def do_item(self, payload, sender):
        item = payload.get("item")
        sel = self.plan.get("forecast_run", {}).get("selected", pd.DataFrame())
        row = _pick(sel, "item", item)
        if not len(row):
            return {"ok": False, "summary": f"{item} has no forecast, it is not an independent demand item"}
        r = row.iloc[0]
        fc = _pick(self.table("forecast"), "item", item)
        total = float(fc["forecast"].sum()) if len(fc) else 0.0
        return {"summary": f"{item} uses {r['method']} at {fmt(r['MAPE'], '%', 1)} error, "
                           f"{fmt(total)} units expected over the horizon",
                "method": r["method"], "mape": float(r["MAPE"]),
                "bias": float(r.get("Bias", 0)), "total_forecast": total,
                "rows": _rows(fc, 12)}

    def do_accuracy(self, payload, sender):
        acc = self.plan.get("forecast_run", {}).get("accuracy", pd.DataFrame())
        item = payload.get("item")
        if item:
            acc = _pick(acc, "item", item)
        return {"summary": f"{len(acc)} method fits on record", "rows": _rows(acc, 24)}

    def do_demand(self, payload, sender):
        """Other agents call this rather than reading the forecast table."""
        item = payload.get("item")
        fc = _pick(self.table("forecast"), "item", item)
        if not len(fc):
            return {"ok": False, "summary": f"no demand signal for {item}"}
        vals = fc["forecast"].to_numpy(dtype=float)
        return {"summary": f"mean {fmt(vals.mean())} per period, "
                           f"variability {fmt(vals.std(), '', 1)}",
                "mean": float(vals.mean()), "sd": float(vals.std()),
                "total": float(vals.sum())}


# ---------------------------------------------------------------------------
class InventoryAgent(Agent):
    name = "Inventory"
    role = "Sets safety stock, reorder points and lot sizes, and watches cover"
    owns = "Step 4"
    skills = ["inventory.summary", "inventory.item", "inventory.risk"]

    def do_summary(self, payload, sender):
        par = self.table("inventory_params")
        if not len(par):
            return {"summary": "no inventory parameters yet"}
        below = int((par["on_hand"] < par["safety_stock_master"]).sum())
        return {"summary": f"{len(par)} items worth {money(par['inventory_value'].sum())}, "
                           f"{below} below safety stock",
                "below_safety": below,
                "value": float(par["inventory_value"].sum()),
                "a_items": int((par["abc_class"] == "A").sum())}

    def do_item(self, payload, sender):
        item = payload.get("item")
        par = _pick(self.table("inventory_params"), "item", item)
        if not len(par):
            return {"ok": False, "summary": f"{item} has no inventory parameters"}
        r = par.iloc[0]
        # the buffer is only sensible against the demand signal, so ask for it
        demand = self.ask("Forecast", "forecast.demand",
                          f"what does demand for {item} look like", item=item)
        note = ""
        if demand.get("ok") and demand.get("sd", 0) > 0:
            note = (f" Demand runs {fmt(demand['mean'])} per period with a spread of "
                    f"{fmt(demand['sd'], '', 1)}.")
        return {"summary": f"{item} holds {fmt(r['on_hand'])} {r['uom']} against safety stock "
                           f"{fmt(r['safety_stock_master'])}, {fmt(r['weeks_of_supply'], '', 1)} "
                           f"weeks of cover, class {r['abc_class']}.{note}",
                "on_hand": float(r["on_hand"]),
                "safety_stock": float(r["safety_stock_master"]),
                "statistical_ss": float(r["safety_stock_statistical"]),
                "reorder_point": float(r["reorder_point"]),
                "eoq": float(r["eoq"]), "abc": r["abc_class"],
                "cover_weeks": float(r["weeks_of_supply"])}

    def do_risk(self, payload, sender):
        exc = self.table("inventory_exceptions")
        proj = self.table("inventory_projection")
        stockouts = proj[proj["closing"] < 0] if len(proj) else pd.DataFrame()
        return {"summary": f"{len(exc)} inventory exceptions, "
                           f"{stockouts['item'].nunique() if len(stockouts) else 0} products "
                           f"projected to run dry without production",
                "rows": _rows(exc)}


# ---------------------------------------------------------------------------
class CapacityAgent(Agent):
    """Owns the hours. During levelling the master schedule has to ask this
    agent for spare capacity rather than helping itself."""

    name = "Capacity"
    role = "Owns available hours, hands them out and reports the constraint"
    owns = "Steps 9 and 5.5"
    skills = ["capacity.summary", "capacity.bottleneck", "capacity.workcentre",
              "capacity.spare", "capacity.reserve"]

    def __init__(self, bus, board):
        super().__init__(bus, board)
        self._load = pd.DataFrame()
        self.grants = 0
        self.refusals = 0
        self.hours_granted = 0.0

    # -- live interface used during the run --------------------------------
    def ledger_factory(self, spare: Dict[tuple, float]):
        agent = self

        class NegotiatedLedger(mps_mod.CapacityLedger):
            def spare(self, work_centre, period):
                free = super().spare(work_centre, period)
                agent.bus.request("MPS", "Capacity", "capacity.spare",
                                  f"how many hours are free on {work_centre} in period {period}",
                                  work_centre=work_centre, period=int(period), free=round(free, 2))
                return free

            def reserve(self, work_centre, period, hours):
                super().reserve(work_centre, period, hours)
                if hours > 0:
                    agent.grants += 1
                    agent.hours_granted += float(hours)
                    agent.bus.request("MPS", "Capacity", "capacity.reserve",
                                      f"takes {hours:,.1f} h on {work_centre} in period {period}",
                                      work_centre=work_centre, period=int(period),
                                      hours=round(float(hours), 2))

        return NegotiatedLedger(spare)

    def capacity_probe(self, work_centre: str, period: int) -> bool:
        """Release asks whether a work centre is overloaded in a period."""
        over = False
        if len(self._load):
            hit = self._load[(self._load["work_centre"] == work_centre) &
                             (self._load["period"] == int(period))]
            over = bool(len(hit) and hit.iloc[0]["status"] == "Overloaded")
        return over

    def note_load(self, load: pd.DataFrame) -> None:
        self._load = load if isinstance(load, pd.DataFrame) else pd.DataFrame()

    # -- question handlers -------------------------------------------------
    def do_spare(self, payload, sender):
        return {"summary": f"{payload.get('free', 0):,.1f} h free on "
                           f"{payload.get('work_centre')} in period {payload.get('period')}",
                "free": payload.get("free")}

    def do_reserve(self, payload, sender):
        return {"summary": f"granted {payload.get('hours', 0):,.1f} h on "
                           f"{payload.get('work_centre')} in period {payload.get('period')}"}

    def do_summary(self, payload, sender):
        s = self.summary
        load = self.table("capacity_load")
        over = int((load["status"] == "Overloaded").sum()) if len(load) else 0
        return {"summary": f"peak load {fmt(s.get('peak_utilisation'), '%')} , "
                           f"{over} overloaded work centre periods, "
                           f"{self.grants} capacity grants made during levelling "
                           f"({fmt(self.hours_granted, ' h', 1)})",
                "peak": s.get("peak_utilisation"), "overloaded": over,
                "grants": self.grants}

    def do_bottleneck(self, payload, sender):
        load = self.table("capacity_load")
        if not len(load):
            return {"summary": "no capacity load has been calculated"}
        by = load.groupby("work_centre", as_index=False)["utilisation_pct"].mean()
        by = by.sort_values("utilisation_pct", ascending=False)
        top = by.iloc[0]
        util = self.table("utilisation")
        shop = ""
        if len(util):
            u = util.sort_values("utilisation_pct", ascending=False).iloc[0]
            shop = (f" On the shop floor itself the busiest is {u['work_centre']} at "
                    f"{fmt(u['utilisation_pct'], '%', 1)}.")
        return {"summary": f"the planning constraint is {top['work_centre']} averaging "
                           f"{fmt(top['utilisation_pct'], '%', 1)} of available hours.{shop}",
                "work_centre": top["work_centre"],
                "utilisation": float(top["utilisation_pct"]),
                "rows": _rows(by)}

    def do_workcentre(self, payload, sender):
        wc = payload.get("work_centre")
        load = _pick(self.table("capacity_load"), "work_centre", wc)
        if not len(load):
            return {"ok": False, "summary": f"{wc} carries no load in this plan"}
        peak = load.sort_values("utilisation_pct", ascending=False).iloc[0]
        return {"summary": f"{wc} peaks at {fmt(peak['utilisation_pct'], '%', 1)} in period "
                           f"{int(peak['period'])}, average "
                           f"{fmt(load['utilisation_pct'].mean(), '%', 1)}",
                "rows": _rows(load, 12)}


# ---------------------------------------------------------------------------
class MPSAgent(Agent):
    name = "MPS"
    role = "Turns demand into a buildable master schedule and levels it"
    owns = "Step 5"
    skills = ["mps.summary", "mps.item", "mps.moves", "mps.feasibility"]

    def do_summary(self, payload, sender):
        s = self.summary
        moves = self.table("mps_moves")
        return {"summary": f"{fmt(s.get('mps_units'))} units scheduled worth "
                           f"{money(s.get('mps_value', 0))}, {len(moves)} build ahead moves "
                           f"were needed to fit capacity",
                "units": s.get("mps_units"), "moves": len(moves)}

    def do_item(self, payload, sender):
        item = payload.get("item")
        g = _pick(self.table("mps"), "item", item)
        if not len(g):
            return {"ok": False, "summary": f"{item} is not master scheduled, it is a dependent demand item"}
        lots = int((g["mps_qty"] > 0).sum())
        return {"summary": f"{item} is scheduled {fmt(g['mps_qty'].sum())} units across {lots} lots, "
                           f"closing balance ends at {fmt(g['pab'].iloc[-1])}",
                "total": float(g["mps_qty"].sum()), "lots": lots,
                "rows": _rows(g, 12)}

    def do_moves(self, payload, sender):
        moves = self.table("mps_moves")
        if not len(moves):
            return {"summary": "no build ahead was needed, the schedule fitted as planned"}
        return {"summary": f"{len(moves)} lots pulled earlier to clear overloads, "
                           f"{fmt(moves['qty_moved'].sum())} units in total",
                "rows": _rows(moves)}

    def do_feasibility(self, payload, sender):
        feas = self.plan.get("mps_feasibility", {})
        issues = feas.get("issues", pd.DataFrame())
        if feas.get("feasible"):
            return {"summary": "the schedule passes the balance, safety stock and capacity checks"}
        return {"summary": f"the schedule has {len(issues)} open feasibility issue(s)",
                "rows": _rows(issues)}


# ---------------------------------------------------------------------------
class SupplierAgent(Agent):
    name = "Supplier"
    role = "Speaks for inbound supply: lead times, reliability and expedites"
    owns = "Step 7 support"
    skills = ["supplier.summary", "supplier.item", "supplier.quote", "supplier.reliability"]

    def __init__(self, bus, board, ds):
        super().__init__(bus, board)
        self.ds = ds

    def _row(self, item):
        sup = self.ds.get("supplier_leadtime")
        hit = _pick(sup, "item", item)
        return hit.iloc[0].to_dict() if len(hit) else None

    def do_summary(self, payload, sender):
        sup = self.ds.get("supplier_leadtime")
        if not len(sup):
            return {"summary": "no supplier master is attached"}
        weak = sup.sort_values("reliability").iloc[0]
        return {"summary": f"{sup['supplier'].nunique()} suppliers across {len(sup)} item links, "
                           f"least reliable is {weak['supplier']} at {fmt(weak['reliability'], '%')}",
                "rows": _rows(sup)}

    def do_item(self, payload, sender):
        item = payload.get("item")
        r = self._row(item)
        if r is None:
            return {"ok": False, "summary": f"{item} has no supplier on file, it looks like a made item"}
        return {"summary": f"{item} comes from {r.get('supplier')} on a "
                           f"{fmt(r.get('lead_time'))} period lead time, reliability "
                           f"{fmt(r.get('reliability'), '%')}, minimum order {fmt(r.get('moq'))}",
                "supplier": r.get("supplier"), "lead_time": float(r.get("lead_time", 1)),
                "reliability": float(r.get("reliability", 100)),
                "moq": float(r.get("moq", 0))}

    def do_reliability(self, payload, sender):
        return self.do_item(payload, sender)

    def do_quote(self, payload, sender):
        """Can inbound be pulled in, and what does it cost in risk."""
        item = payload.get("item")
        qty = float(payload.get("qty", 0) or 0)
        r = self._row(item)
        if r is None:
            return {"ok": False, "summary": f"no supplier holds {item}, it has to be made"}
        lt = float(r.get("lead_time", 1))
        rel = float(r.get("reliability", 100))
        moq = float(r.get("moq", 0))
        # an expedite is only credible from a supplier that already performs
        feasible = rel >= 90 and lt > 1
        pulled = max(1.0, lt - 1) if feasible else lt
        order_qty = max(qty, moq)
        verdict = ("can pull one period forward" if feasible else
                   "cannot commit to an earlier date")
        return {"summary": f"{r.get('supplier')} {verdict} on {item}: "
                           f"{fmt(lt)} periods becomes {fmt(pulled)}, order rounded to "
                           f"{fmt(order_qty)} for the minimum",
                "feasible": feasible, "new_lead_time": pulled,
                "order_qty": order_qty, "reliability": rel,
                "supplier": r.get("supplier")}


# ---------------------------------------------------------------------------
class MaterialAgent(Agent):
    """BOM explosion and MRP. Also answers the release step's component
    questions live, and consults the supplier agent when inbound is the issue."""

    name = "Material"
    role = "Explodes the bill, nets requirements and chases components"
    owns = "Steps 6 and 7"
    skills = ["material.summary", "material.item", "material.shortages",
              "material.pegging", "material.expedite", "material.check"]

    def __init__(self, bus, board):
        super().__init__(bus, board)
        self._mrp = pd.DataFrame()
        self.checks = 0
        self.blocked = 0

    def note_mrp(self, mrp: pd.DataFrame) -> None:
        self._mrp = mrp if isinstance(mrp, pd.DataFrame) else pd.DataFrame()

    # -- live interface used during release --------------------------------
    def material_probe(self, component: str, period: int, need: float):
        self.checks += 1
        info = None
        if len(self._mrp):
            hit = self._mrp[(self._mrp["item"] == component) &
                            (self._mrp["period"] == int(period))]
            if len(hit):
                r = hit.iloc[0]
                info = {"projected": float(r["projected_available"]), "status": r["status"]}
        short = info is not None and (info["projected"] < -0.001 or
                                      info["status"] == "Past due release")
        if short:
            self.blocked += 1
        self.bus.request("Scheduling", "Material", "material.check",
                         f"is there enough {component} in period {period}",
                         component=component, period=int(period), need=round(float(need), 1),
                         short=short)
        return info

    def do_check(self, payload, sender):
        if payload.get("short"):
            return {"summary": f"{payload.get('component')} is not covered in period "
                               f"{payload.get('period')}, the order has to wait"}
        return {"summary": f"{payload.get('component')} is covered in period {payload.get('period')}"}

    # -- question handlers -------------------------------------------------
    def do_summary(self, payload, sender):
        mrp = self.table("mrp")
        exc = self.table("mrp_exceptions")
        orders = int((mrp["planned_order_receipt"] > 0).sum()) if len(mrp) else 0
        return {"summary": f"{mrp['item'].nunique() if len(mrp) else 0} items planned, "
                           f"{orders} planned orders, {len(exc)} material exceptions. "
                           f"{self.checks} component checks were answered during release, "
                           f"{self.blocked} came back short",
                "planned_orders": orders, "exceptions": len(exc)}

    def do_item(self, payload, sender):
        item = payload.get("item")
        g = _pick(self.table("mrp"), "item", item)
        if not len(g):
            return {"ok": False, "summary": f"{item} is not in the material plan"}
        r = g.iloc[0]
        shortage = g[g["projected_available"] < -0.001]
        note = ""
        # if it is bought and it is tight, the supplier is the one to ask
        if len(shortage):
            quote = self.ask("Supplier", "supplier.item",
                             f"who supplies {item} and how reliable are they", item=item)
            if quote.get("ok"):
                note = (f" It is bought from {quote.get('supplier')} on a "
                        f"{fmt(quote.get('lead_time'))} period lead time.")
        return {"summary": f"{item} sits at BOM level {int(r['level'])}, lot rule {r['lot_rule']}, "
                           f"lead time {fmt(r['lead_time'])}. "
                           f"{int((g['planned_order_receipt'] > 0).sum())} planned orders, "
                           f"{len(shortage)} period(s) projected negative.{note}",
                "level": int(r["level"]), "shortage_periods": len(shortage),
                "rows": _rows(g, 12)}

    def do_shortages(self, payload, sender):
        mrp = self.table("mrp")
        if not len(mrp):
            return {"summary": "no material plan yet"}
        bad = mrp[(mrp["projected_available"] < -0.001) |
                  (mrp["status"] == "Past due release")]
        if not len(bad):
            return {"summary": "no material shortages, every component is covered"}
        by = bad.groupby("item", as_index=False).size().sort_values("size", ascending=False)
        return {"summary": f"{len(by)} components are short or past due, worst is "
                           f"{by.iloc[0]['item']}",
                "items": by["item"].tolist()[:10], "rows": _rows(bad, 10)}

    def do_pegging(self, payload, sender):
        item = payload.get("item")
        peg = self.table("mrp_pegging")
        p = peg[peg["component"] == item] if len(peg) else pd.DataFrame()
        if not len(p):
            return {"summary": f"{item} has independent demand, it is driven by the master schedule"}
        parents = p["parent"].unique().tolist()
        return {"summary": f"{item} is pulled by {', '.join(parents[:4])}",
                "parents": parents, "rows": _rows(p, 10)}

    def do_expedite(self, payload, sender):
        """Work out what an expedite would take, in consultation with supply."""
        item = payload.get("item")
        qty = payload.get("qty", 0)
        quote = self.ask("Supplier", "supplier.quote",
                         f"can we pull {item} in", item=item, qty=qty)
        if not quote.get("ok"):
            return {"summary": f"{item} is made in house, so the answer is a schedule change "
                               f"rather than an expedite"}
        return {"summary": quote.get("summary"), "feasible": quote.get("feasible"),
                "order_qty": quote.get("order_qty"), "supplier": quote.get("supplier")}


# ---------------------------------------------------------------------------
class SchedulingAgent(Agent):
    name = "Scheduling"
    role = "Releases orders and sequences them on the floor"
    owns = "Steps 8, 10 and 11"
    skills = ["schedule.summary", "schedule.item", "schedule.late", "schedule.rules",
              "schedule.release"]

    def do_summary(self, payload, sender):
        s = self.summary
        return {"summary": f"{s.get('jobs', 0)} jobs, makespan "
                           f"{fmt(s.get('makespan_h'), ' h', 1)}, on time delivery "
                           f"{fmt(s.get('otd_pct'), '%')}, utilisation "
                           f"{fmt(s.get('utilisation_pct'), '%', 1)} under "
                           f"{self.plan.get('dispatch_rule')}",
                "jobs": s.get("jobs"), "otd": s.get("otd_pct")}

    def do_release(self, payload, sender):
        rel = self.table("order_release")
        if not len(rel):
            return {"summary": "no orders fall inside the release window"}
        held = rel[rel["status"] == "Held"]
        return {"summary": f"{int((rel['status'] == 'Released').sum())} released, "
                           f"{len(held)} held",
                "rows": _rows(rel)}

    def do_item(self, payload, sender):
        item = payload.get("item")
        jobs = self.table("schedule_jobs")
        sched = self.table("schedule")
        mine = sched[sched["item"] == item] if len(sched) else pd.DataFrame()
        if not len(mine):
            return {"ok": False, "summary": f"{item} has no jobs on the floor this window"}
        jl = jobs[jobs["job"].isin(mine["job"].unique())] if len(jobs) else pd.DataFrame()
        late = jl[~jl["on_time"]] if len(jl) and "on_time" in jl else pd.DataFrame()
        return {"summary": f"{mine['job'].nunique()} job(s) for {item} across "
                           f"{len(mine)} operations, {len(late)} finishing late",
                "late": len(late), "rows": _rows(mine, 10)}

    def do_late(self, payload, sender):
        """Why is anything late. Ask the two agents that could be the cause."""
        jobs = self.table("schedule_jobs")
        if not len(jobs):
            return {"summary": "nothing is scheduled, so nothing is late"}
        late = jobs[~jobs["on_time"]] if "on_time" in jobs else pd.DataFrame()
        bott = self.ask("Capacity", "capacity.bottleneck", "what is the constraint")
        short = self.ask("Material", "material.shortages", "is anything short")
        if not len(late):
            return {"summary": f"every job is on time under {self.plan.get('dispatch_rule')}. "
                               f"{bott.get('summary', '')} {short.get('summary', '')}",
                    "late_jobs": 0}
        worst = late.sort_values("tardiness_min", ascending=False).iloc[0]
        return {"summary": f"{len(late)} job(s) miss their date, worst is {worst['job']} "
                           f"({worst['item']}) by {fmt(worst['tardiness_min'] / 60, ' h', 1)}. "
                           f"{bott.get('summary', '')} {short.get('summary', '')}",
                "late_jobs": len(late), "rows": _rows(late)}

    def do_rules(self, payload, sender):
        tbl = self.table("rule_comparison")
        rank = self.table("rule_ranking")
        if not len(tbl):
            return {"summary": "no jobs to compare rules on"}
        best = rank.iloc[0]["rule"] if len(rank) else ""
        bests = self.plan.get("rule_best", {})
        return {"summary": f"{best} scores highest overall. Best on flow time is "
                           f"{bests.get('Avg_Flow_Time_h')}, on tardiness "
                           f"{bests.get('Avg_Tardiness_h')}, on makespan "
                           f"{bests.get('Makespan_h')}",
                "best": best, "rows": _rows(tbl, 8)}


# ---------------------------------------------------------------------------
class ExceptionAgent(Agent):
    name = "Exceptions"
    role = "Ranks what is wrong and asks the owning agent for the remedy"
    owns = "Step 12"
    skills = ["exception.summary", "exception.list", "exception.item", "exception.remedy"]

    OWNER = {"Forecast agent": "Forecast", "Inventory agent": "Inventory",
             "Schedule agent": "MPS", "Material agent": "Material",
             "Shop floor agent": "Scheduling", "Supplier agent": "Supplier"}

    def do_summary(self, payload, sender):
        exc = self.table("exceptions")
        if not len(exc):
            return {"summary": "nothing to escalate, the plan is clean"}
        high = exc[exc["severity"] == "High"]
        return {"summary": f"{len(exc)} open exceptions, {len(high)} high severity, "
                           f"{money(exc['exposure_value'].sum())} at stake",
                "total": len(exc), "high": len(high),
                "rows": _rows(exc[["exception_id", "agent", "item", "type", "severity"]], 8)}

    def do_list(self, payload, sender):
        exc = self.table("exceptions")
        sev = payload.get("severity")
        if sev and len(exc):
            exc = exc[exc["severity"].str.lower() == str(sev).lower()]
        return {"summary": f"{len(exc)} exceptions listed", "rows": _rows(exc, 12)}

    def do_item(self, payload, sender):
        item = payload.get("item")
        exc = self.table("exceptions")
        mine = exc[exc["item"] == item] if len(exc) else pd.DataFrame()
        if not len(mine):
            return {"summary": f"nothing is flagged against {item}"}
        return {"summary": f"{len(mine)} exception(s) on {item}: "
                           f"{', '.join(mine['type'].unique()[:3])}",
                "rows": _rows(mine)}

    def do_remedy(self, payload, sender):
        """Take the top exception to the agent that owns that part of the plan."""
        exc = self.table("exceptions")
        eid = payload.get("exception_id")
        item = payload.get("item")
        if eid and len(exc):
            row = exc[exc["exception_id"] == eid]
        elif item and len(exc):
            row = exc[exc["item"] == item]
        else:
            row = exc
        if not len(row):
            return {"summary": "no exception matches that"}
        r = row.iloc[0]
        owner = self.OWNER.get(r["agent"], "Coordinator")
        proposal = {}
        if r["agent"] == "Material agent":
            proposal = self.ask("Material", "material.expedite",
                                f"what would it take to fix {r['item']}",
                                item=r["item"], qty=r.get("qty", 0))
        elif r["agent"] == "Inventory agent":
            proposal = self.ask("Inventory", "inventory.item",
                                f"where does {r['item']} stand", item=r["item"])
        elif r["agent"] == "Shop floor agent":
            proposal = self.ask("Scheduling", "schedule.late", "what is running late")
        elif r["agent"] == "Schedule agent":
            proposal = self.ask("MPS", "mps.feasibility", "does the schedule fit")
        return {"summary": f"{r['exception_id']} on {r['item']}: {r['detail']} "
                           f"Recommended action: {r['recommended_action']}. "
                           f"{owner} adds: {proposal.get('summary', 'no further detail')}",
                "exception_id": r["exception_id"], "owner": owner,
                "recommended_action": r["recommended_action"],
                "action_code": r.get("action_code", "review"),
                "item": r["item"]}
