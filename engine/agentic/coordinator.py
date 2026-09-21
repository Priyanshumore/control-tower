"""
The coordinator.

It owns the roster, drives a planning run through the sub agents, and turns a
typed request into delegation. It does no planning arithmetic itself: every
number in an answer came back from the agent that owns that part of the plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..agents import Overrides, apply_decision
from ..loader import Dataset
from ..pipeline import (SCENARIO_LIBRARY, Settings, build_scenario, compare_runs,
                        run_pipeline)
from . import language
from .base import Agent, fmt, money
from .bus import Blackboard, MessageBus
from .roster import (CapacityAgent, DataAgent, ExceptionAgent, ForecastAgent,
                     InventoryAgent, MaterialAgent, MPSAgent, SchedulingAgent,
                     SupplierAgent)


# ---------------------------------------------------------------------------
class AgentHooks:
    """What the agents plug into a planning run."""

    def __init__(self, capacity: CapacityAgent, material: MaterialAgent, bus: MessageBus):
        self.capacity = capacity
        self.material = material
        self.bus = bus
        self.ledger_factory = capacity.ledger_factory
        self.material_probe = material.material_probe
        self.capacity_probe = capacity.capacity_probe

    def notify(self, stage: str, out: Dict[str, Any]) -> None:
        """Each module reports in as it finishes, so the next agent can work."""
        if stage == "mps":
            self.capacity.note_load(out.get("capacity_load", pd.DataFrame()))
            self.bus.inform("MPS", "master schedule levelled and handed to materials",
                            topic="stage.mps")
        elif stage == "mrp":
            self.material.note_mrp(out.get("mrp", pd.DataFrame()))
            self.bus.inform("Material", "material plan ready, release can be checked",
                            topic="stage.mrp")
        elif stage == "forecast":
            self.bus.inform("Forecast", "demand signal published", topic="stage.forecast")
        elif stage == "release":
            self.bus.inform("Scheduling", "orders released, sequencing next",
                            topic="stage.release")


# ---------------------------------------------------------------------------
class ScenarioAgent(Agent):
    name = "Scenario"
    role = "Runs a disruption against the live plan and reports the damage"
    owns = "Step 13"
    skills = ["scenario.run", "scenario.list"]

    def __init__(self, bus, board, session):
        super().__init__(bus, board)
        self.session = session

    def do_list(self, payload, sender):
        return {"summary": "; ".join(f"{k}: {v}" for k, v in SCENARIO_LIBRARY.items()),
                "kinds": list(SCENARIO_LIBRARY)}

    def do_run(self, payload, sender):
        kind = payload.get("kind")
        if kind not in SCENARIO_LIBRARY:
            return {"ok": False, "summary": f"{kind} is not a scenario I know"}
        res = self.session.run_scenario(kind, payload.get("magnitude"),
                                        payload.get("target"), payload.get("delay", 2))
        cmp_df = res["comparison"]
        worse = cmp_df[cmp_df["Direction"] == "Worse"]
        better = cmp_df[cmp_df["Direction"] == "Better"]
        if not len(worse) and not len(better):
            headline = (f"{kind}: the plan absorbs it. Nothing measurable moves, which means "
                        f"the disruption lands where there is slack rather than on the "
                        f"constraint. Try a larger one, or aim it at the bottleneck.")
        else:
            headline = f"{kind}: {len(worse)} measure(s) get worse, {len(better)} improve"
        detail = []
        for _, r in worse.head(4).iterrows():
            detail.append(f"{r['Metric']} moves {r['Change']:+,.1f}")
        # the exception agent is the one who knows what is newly broken
        newx = res["new_exceptions"]
        if len(newx):
            detail.append(f"{len(newx)} new exception(s), including "
                          f"{newx.iloc[0]['item']} {newx.iloc[0]['type']}")
        return {"summary": headline + (". " + "; ".join(detail) if detail else ""),
                "comparison": cmp_df, "new_exceptions": newx,
                "scenario": kind, "overrides": res["overrides"].summary()}


# ---------------------------------------------------------------------------
@dataclass
class Reply:
    headline: str
    sections: List[Tuple[str, str]] = field(default_factory=list)
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    intent: Optional[language.Intent] = None
    from_seq: int = 0
    focus_tab: Optional[str] = None

    def text(self) -> str:
        out = [self.headline]
        for who, what in self.sections:
            out.append(f"{who}: {what}")
        return "\n".join(out)


# ---------------------------------------------------------------------------
class Coordinator(Agent):
    name = "Coordinator"
    role = "Reads the request, delegates to the specialists, assembles the answer"
    owns = "Orchestration"
    skills = ["plan.status"]

    def __init__(self, bus, board, session):
        super().__init__(bus, board)
        self.session = session

    def do_status(self, payload, sender):
        return {"summary": "the coordinator holds the plan and delegates"}

    # -- the delegation table ---------------------------------------------
    def dispatch(self, prompt: str) -> Reply:
        s = self.session
        intent = language.parse(prompt, s.item_codes(), s.work_centre_codes())
        start = len(self.bus.log)
        self.bus.inform("User", prompt, "Coordinator", topic="prompt")

        handler = getattr(self, "_on_" + intent.action, self._on_status)
        reply = handler(intent)
        if not (reply.headline or "").strip():
            # a specialist declined or failed: say so rather than showing a blank
            reply.headline = ("The specialists had nothing to report on that. "
                              "The transcript below shows what was asked.")
        reply.intent = intent
        reply.from_seq = start
        return reply

    # -- one handler per intent -------------------------------------------
    def _on_help(self, intent):
        roster = self.session.roster_frame()
        return Reply(
            headline="Ask me about any part of the plan and I will put it to the right specialist.",
            sections=[("Roster", ", ".join(f"{r['Agent']} ({r['Owns']})"
                                           for _, r in roster.iterrows()))],
            tables={"Agents": roster})

    def _on_status(self, intent):
        # a full brief means asking everyone, which is the point of the roster
        f = self.ask("Forecast", "forecast.summary", "how good is the demand signal")
        i = self.ask("Inventory", "inventory.summary", "where is stock")
        m = self.ask("MPS", "mps.summary", "what are we building")
        mat = self.ask("Material", "material.summary", "are materials covered")
        c = self.ask("Capacity", "capacity.summary", "does it fit")
        sc = self.ask("Scheduling", "schedule.summary", "how does the floor look")
        e = self.ask("Exceptions", "exception.summary", "what is broken")
        s = self.session.summary()
        verdict = self.session.verdict()[0]
        return Reply(
            headline=f"{verdict}. {e.get('summary', '')}",
            sections=[("Forecast", f.get("summary", "")), ("Inventory", i.get("summary", "")),
                      ("MPS", m.get("summary", "")), ("Material", mat.get("summary", "")),
                      ("Capacity", c.get("summary", "")), ("Scheduling", sc.get("summary", ""))],
            tables={"Top exceptions": self.session.table("exceptions").head(8)},
            focus_tab="Control tower")

    def _on_item(self, intent):
        item = intent.item
        if not item:
            return self._on_status(intent)
        parts = []
        d = self.ask("Data", "data.item", f"what is {item}", item=item)
        parts.append(("Data", d.get("summary", "")))
        for agent, topic in [("Forecast", "forecast.item"), ("Inventory", "inventory.item"),
                             ("MPS", "mps.item"), ("Material", "material.item"),
                             ("Scheduling", "schedule.item"), ("Exceptions", "exception.item")]:
            r = self.ask(agent, topic, f"what do you have on {item}", item=item)
            if r.get("summary"):
                parts.append((agent, r["summary"]))
        peg = self.ask("Material", "material.pegging", f"what pulls {item}", item=item)
        parts.append(("Material", peg.get("summary", "")))
        tables = {}
        mrp = self.session.table("mrp")
        if len(mrp) and item in set(mrp["item"]):
            tables["Material plan"] = mrp[mrp["item"] == item]
        return Reply(headline=f"Briefing on {item}, gathered from every specialist.",
                     sections=parts, tables=tables, focus_tab="7 MRP")

    def _on_forecast(self, intent):
        if intent.item:
            r = self.ask("Forecast", "forecast.item", f"how does {intent.item} forecast",
                         item=intent.item)
        else:
            r = self.ask("Forecast", "forecast.summary", "how accurate are we")
        acc = self.ask("Forecast", "forecast.accuracy", "show the method fits", item=intent.item)
        return Reply(headline=r.get("summary", ""),
                     sections=[("Forecast", acc.get("summary", ""))],
                     tables={"Method accuracy": pd.DataFrame(acc.get("rows", []))},
                     focus_tab="3 Forecast")

    def _on_inventory(self, intent):
        if intent.item:
            r = self.ask("Inventory", "inventory.item", f"where is {intent.item}",
                         item=intent.item)
        else:
            r = self.ask("Inventory", "inventory.summary", "how is stock")
        risk = self.ask("Inventory", "inventory.risk", "what is at risk")
        import re
        stockout_query = bool(re.search(r"stock[ -]?outs?|out of stock|run(?:ning)? (?:out|dry)", intent.text, re.I))
        if stockout_query:
            projection = self.session.table("inventory_projection")
            affected = projection[projection["closing"] < 0].copy() if len(projection) else pd.DataFrame()
            if intent.items and len(affected):
                affected = affected[affected["item"].isin(intent.items)]
            if len(affected):
                first = affected.sort_values("period").groupby("item", as_index=False).first()
                master = self.session.ds.get("inventory_master").set_index("item")
                first["description"] = first["item"].map(master["description"])
                first["uom"] = first["item"].map(master["uom"])
                first["shortfall"] = -first["closing"]
                first["next_action"] = "Review MPS quantities and incoming receipts before this period"
                first = first[["item", "description", "period", "shortfall", "uom", "next_action"]].sort_values("period")
            else:
                first = pd.DataFrame()
            count = len(first)
            return Reply(headline=f"{count} product{'s' if count != 1 else ''} projected to stock out without new production",
                         sections=[("Planning basis", "This is the inventory projection before new production. It is a risk the master schedule must cover, not a claim that the final production plan stocks out.")],
                         tables={"Stockout items": first}, focus_tab="4 Inventory")
        exceptions = self.session.table("inventory_exceptions")
        if intent.items and len(exceptions):
            exceptions = exceptions[exceptions["item"].isin(intent.items)]
        return Reply(headline=r.get("summary", ""), sections=[("Inventory", risk.get("summary", ""))],
                     tables={"Inventory exceptions": exceptions}, focus_tab="4 Inventory")

    def _on_mps(self, intent):
        if intent.item:
            r = self.ask("MPS", "mps.item", f"what is scheduled for {intent.item}",
                         item=intent.item)
        else:
            r = self.ask("MPS", "mps.summary", "what are we building")
        moves = self.ask("MPS", "mps.moves", "what did levelling move")
        feas = self.ask("MPS", "mps.feasibility", "does it fit")
        return Reply(headline=r.get("summary", ""),
                     sections=[("MPS", moves.get("summary", "")), ("MPS", feas.get("summary", ""))],
                     tables={"Build ahead moves": self.session.table("mps_moves")},
                     focus_tab="5 MPS")

    def _on_material(self, intent):
        if intent.item:
            r = self.ask("Material", "material.item", f"how is {intent.item} covered",
                         item=intent.item)
            peg = self.ask("Material", "material.pegging", f"what pulls {intent.item}",
                           item=intent.item)
            extra = [("Material", peg.get("summary", ""))]
        else:
            r = self.ask("Material", "material.shortages", "what is short")
            extra = [("Material", self.ask("Material", "material.summary",
                                           "how did netting go").get("summary", ""))]
        return Reply(headline=r.get("summary", ""), sections=extra,
                     tables={"Material exceptions": self.session.table("mrp_exceptions")},
                     focus_tab="7 MRP")

    def _on_supplier(self, intent):
        if intent.item:
            r = self.ask("Supplier", "supplier.item", f"who supplies {intent.item}",
                         item=intent.item)
        else:
            r = self.ask("Supplier", "supplier.summary", "how is inbound")
        return Reply(headline=r.get("summary", ""), tables={}, focus_tab="7 MRP")

    def _on_capacity(self, intent):
        if intent.work_centre:
            r = self.ask("Capacity", "capacity.workcentre",
                         f"how loaded is {intent.work_centre}", work_centre=intent.work_centre)
        else:
            r = self.ask("Capacity", "capacity.bottleneck", "where is the constraint")
        s = self.ask("Capacity", "capacity.summary", "how tight is it")
        return Reply(headline=r.get("summary", ""), sections=[("Capacity", s.get("summary", ""))],
                     tables={"Work centre load": self.session.table("capacity_load")},
                     focus_tab="9 Capacity")

    def _on_schedule(self, intent):
        if intent.item:
            r = self.ask("Scheduling", "schedule.item", f"how is {intent.item} running",
                         item=intent.item)
        else:
            r = self.ask("Scheduling", "schedule.late", "is anything late")
        rel = self.ask("Scheduling", "schedule.release", "what got released")
        return Reply(headline=r.get("summary", ""), sections=[("Scheduling", rel.get("summary", ""))],
                     tables={"Order release": self.session.table("order_release")},
                     focus_tab="10 Schedule")

    def _on_rules(self, intent):
        r = self.ask("Scheduling", "schedule.rules", "which rule wins")
        sections = []
        if intent.rule:
            sections.append(("Coordinator",
                             f"To schedule the floor with {intent.rule}, set the dispatching "
                             f"rule in the control panel and the whole plan recomputes."))
        return Reply(headline=r.get("summary", ""), sections=sections,
                     tables={"Rule comparison": self.session.table("rule_comparison")},
                     focus_tab="11 KPIs")

    def _on_exceptions(self, intent):
        r = self.ask("Exceptions", "exception.summary", "what is broken")
        lst = self.ask("Exceptions", "exception.list", "list them", severity=intent.severity)
        return Reply(headline=r.get("summary", ""), sections=[("Exceptions", lst.get("summary", ""))],
                     tables={"Exceptions": pd.DataFrame(lst.get("rows", []))},
                     focus_tab="12 Exceptions")

    def _on_remedy(self, intent):
        r = self.ask("Exceptions", "exception.remedy",
                     f"what do we do about {intent.item or 'the worst one'}",
                     item=intent.item)
        return Reply(headline=r.get("summary", ""),
                     sections=[("Coordinator",
                                "Apply it on the Exceptions tab and the plan recomputes "
                                "with the change in place.")],
                     tables={}, focus_tab="12 Exceptions")

    def _on_scenario(self, intent):
        kind = intent.scenario
        if kind is None:
            # no scenario phrase matched, so read it off whatever was named
            low = (intent.text or "").lower()
            wants_more = any(w in low for w in ("add", "extra", "more", "invest",
                                                "expand", "another shift"))
            if intent.work_centre:
                kind = "Capacity expansion" if wants_more else "Machine breakdown"
            elif intent.item:
                kind = "Rush order" if "order" in low else "Supplier delay"
            else:
                kind = "Demand surge"
        mag = intent.percent or intent.hours or (intent.numbers[0] if intent.numbers else None)
        target = intent.work_centre or intent.item
        r = self.ask("Scenario", "scenario.run", f"run {kind}",
                     kind=kind, magnitude=mag, target=target,
                     delay=int(intent.periods[0]) if intent.periods else 2)
        if not r.get("ok"):
            return Reply(headline=r.get("summary", "that scenario did not run"))
        return Reply(headline=r.get("summary", ""),
                     sections=[("Scenario", "; ".join(r.get("overrides", [])))],
                     tables={"Baseline against scenario": r.get("comparison", pd.DataFrame()),
                             "New exceptions": r.get("new_exceptions", pd.DataFrame())},
                     focus_tab="13 Scenarios")

    def _on_data(self, intent):
        r = self.ask("Data", "data.summary", "what did we load")
        iss = self.ask("Data", "data.issues", "any problems")
        return Reply(headline=r.get("summary", ""), sections=[("Data", iss.get("summary", ""))],
                     tables={}, focus_tab="1-2 Data")


# ---------------------------------------------------------------------------
class PlanningSession:
    """Holds the bus, the blackboard, the roster and the current plan."""

    def __init__(self, ds: Dataset, settings: Optional[Settings] = None,
                 overrides: Optional[Overrides] = None):
        self.ds = ds
        self.settings = settings or Settings()
        self.overrides = overrides or Overrides()
        self.bus = MessageBus()
        self.board = Blackboard()

        self.data = DataAgent(self.bus, self.board, ds)
        self.forecast = ForecastAgent(self.bus, self.board)
        self.inventory = InventoryAgent(self.bus, self.board)
        self.capacity = CapacityAgent(self.bus, self.board)
        self.mps = MPSAgent(self.bus, self.board)
        self.supplier = SupplierAgent(self.bus, self.board, ds)
        self.material = MaterialAgent(self.bus, self.board)
        self.scheduling = SchedulingAgent(self.bus, self.board)
        self.exceptions = ExceptionAgent(self.bus, self.board)
        self.scenario = ScenarioAgent(self.bus, self.board, self)
        self.coordinator = Coordinator(self.bus, self.board, self)

        self.sub_agents = [self.data, self.forecast, self.inventory, self.capacity,
                           self.mps, self.supplier, self.material, self.scheduling,
                           self.exceptions, self.scenario]
        self._scenario_cache: Dict[str, Any] = {}

    # -- running the plan --------------------------------------------------
    def run(self) -> Dict[str, Any]:
        self.bus.reset()
        self.bus.inform("Coordinator", "opening a planning run and calling the roster in order",
                        "all", topic="run.start")
        hooks = AgentHooks(self.capacity, self.material, self.bus)
        plan = run_pipeline(self.ds, self.settings, self.overrides, hooks=hooks)
        self.board.put("plan", plan, "Coordinator")
        s = plan["kpi_summary"]
        self.bus.inform("Coordinator",
                        f"run complete: {fmt(s.get('mps_units'))} units scheduled, "
                        f"{s.get('exceptions_total')} exceptions raised", "all",
                        topic="run.done")
        return plan

    def ensure(self) -> Dict[str, Any]:
        if not self.board.has("plan"):
            self.run()
        return self.board.get("plan")

    def rerun(self, settings: Optional[Settings] = None,
              overrides: Optional[Overrides] = None) -> Dict[str, Any]:
        if settings is not None:
            self.settings = settings
        if overrides is not None:
            self.overrides = overrides
        self._scenario_cache.clear()
        return self.run()

    # -- convenience -------------------------------------------------------
    @property
    def plan(self) -> Dict[str, Any]:
        return self.board.get("plan", {}) or {}

    def table(self, key: str) -> pd.DataFrame:
        v = self.plan.get(key)
        return v if isinstance(v, pd.DataFrame) else pd.DataFrame()

    def summary(self) -> Dict[str, Any]:
        return self.plan.get("kpi_summary", {}) or {}

    def verdict(self) -> Tuple[str, str]:
        s = self.summary()
        high = s.get("exceptions_high", 0)
        over = s.get("overloaded_periods", 0)
        otd = s.get("otd_pct", 100)
        otd = 100 if otd is None or (isinstance(otd, float) and pd.isna(otd)) else otd
        if high == 0 and over == 0 and otd >= 99.9:
            return "The plan is executable", "ok"
        if high and over:
            return "The plan needs intervention", "bad"
        if high:
            return "The plan is at risk", "warn"
        return "The plan is tight", "warn"

    def item_codes(self) -> List[str]:
        inv = self.ds.get("inventory_master")
        return inv["item"].astype(str).tolist() if len(inv) else []

    def work_centre_codes(self) -> List[str]:
        mc = self.plan.get("machine_capacity")
        if isinstance(mc, pd.DataFrame) and len(mc):
            return mc["work_centre"].astype(str).tolist()
        mc = self.ds.get("machine_capacity")
        return mc["work_centre"].astype(str).tolist() if len(mc) else []

    def roster_frame(self) -> pd.DataFrame:
        return pd.DataFrame([a.card() for a in [self.coordinator] + self.sub_agents])

    # -- prompts and scenarios --------------------------------------------
    def ask(self, prompt: str) -> Reply:
        self.ensure()
        return self.coordinator.dispatch(prompt)

    def run_scenario(self, kind: str, magnitude=None, target=None, delay: int = 2) -> Dict[str, Any]:
        base = self.ensure()
        defaults = {"Demand surge": 30, "Machine breakdown": 48, "Rush order": 80,
                    "Capacity expansion": 100, "Supplier delay": 0}
        mag = float(magnitude) if magnitude not in (None, 0) else float(defaults.get(kind, 30))
        sov = build_scenario(kind, self.ds, base, magnitude=mag, target=target,
                             periods_delay=int(delay))
        merged = _merge(self.overrides, sov)
        res = run_pipeline(self.ds, self.settings, merged)
        cmp_df = compare_runs(base, res, kind)
        be = base.get("exceptions", pd.DataFrame())
        se = res.get("exceptions", pd.DataFrame())
        known = set(zip(be["item"], be["type"])) if len(be) else set()
        new = se[[(i, t) not in known for i, t in zip(se["item"], se["type"])]] if len(se) else se
        out = {"result": res, "comparison": cmp_df, "new_exceptions": new, "overrides": sov,
               "kind": kind}
        self._scenario_cache[kind] = out
        return out


def _merge(base: Overrides, extra: Overrides) -> Overrides:
    return Overrides(
        demand_multiplier={**base.demand_multiplier, **extra.demand_multiplier},
        lead_time_override={**base.lead_time_override, **extra.lead_time_override},
        lead_time_delta={**base.lead_time_delta, **extra.lead_time_delta},
        extra_receipts=base.extra_receipts + extra.extra_receipts,
        capacity_multiplier={**base.capacity_multiplier, **extra.capacity_multiplier},
        breakdowns={**base.breakdowns, **extra.breakdowns},
        rush_orders=base.rush_orders + extra.rush_orders,
        priority_items=list(set(base.priority_items + extra.priority_items)),
        notes=extra.notes)
