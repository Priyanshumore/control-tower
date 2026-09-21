"""
Turning a typed request into something the coordinator can delegate.

This is a deterministic domain parser, not a language model: it runs offline,
gives the same answer twice, and can be read and corrected. It recognises the
vocabulary of the plan, which is what the questions are actually made of.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

INTENTS = {
    "help": ["help", "what can you do", "how do i", "commands", "examples", "who are you"],
    "status": ["status", "overview", "how is the plan", "summary", "health", "where do we stand",
               "brief me", "everything", "control tower"],
    "forecast": ["forecast", "demand", "mape", "accuracy", "error", "method", "seasonality",
                 "trend", "predict"],
    "inventory": ["inventory", "stock", "safety stock", "cover", "reorder", "eoq", "abc",
                  "on hand", "holding", "buffer", "stockout", "stock out", "out of stock", "running out", "run dry"],
    "mps": ["master schedule", "mps", "master production", "build ahead", "levelling",
            "leveling", "pab", "atp", "available to promise", "lot"],
    "material": ["material", "mrp", "component", "shortage", "short", "bom", "bill of material",
                 "explosion", "pegging", "raw material", "ingredient"],
    "supplier": ["supplier", "supplies", "supplied", "who makes", "who supplies", "vendor",
                 "lead time", "inbound", "procurement", "purchase order", "sourced", "sourcing",
                 "reliability", "moq"],
    "capacity": ["capacity", "bottleneck", "constraint", "work centre", "work center",
                 "utilisation", "utilization", "hours", "overload", "machine"],
    "schedule": ["schedule", "gantt", "shop floor", "job", "late", "tardy", "tardiness",
                 "makespan", "on time", "otd", "sequence", "release", "flow time"],
    "rules": ["dispatch", "rule", "fcfs", "spt", "lpt", "edd", "critical ratio", "slack",
              "compare rules", "best rule", "which rule"],
    "exceptions": ["exception", "problem", "issue", "risk", "alert", "wrong", "attention",
                   "escalate", "what should i fix", "priorit"],
    "remedy": ["fix", "remedy", "what do i do about", "recommend", "action", "resolve",
               "expedite", "solve"],
    "scenario": ["scenario", "what if", "simulate", "surge", "breakdown", "disruption",
                 "delay", "rush order", "goes down", "increases by", "drops"],
    "data": ["data", "mapping", "column", "file", "dataset", "table", "attach", "upload"],
}

SCENARIO_KINDS = {
    "Demand surge": ["surge", "demand up", "demand increase", "increases by", "spike",
                     "more demand", "higher demand", "festive", "promotion"],
    "Supplier delay": ["supplier delay", "late delivery", "lead time", "supplier late",
                       "delayed", "is delayed", "delay", "delayed shipment", "inbound delay",
                       "supplier fails", "slips"],
    "Machine breakdown": ["breakdown", "goes down", "machine down", "outage", "failure",
                          "line down", "maintenance", "broken"],
    "Rush order": ["rush", "urgent order", "expedite order", "emergency order", "big order"],
    "Capacity expansion": ["expansion", "extra capacity", "add capacity", "more machines",
                           "third shift", "add a shift", "invest"],
}

RULE_WORDS = {"fcfs": "FCFS", "first come": "FCFS", "spt": "SPT", "shortest": "SPT",
              "lpt": "LPT", "longest": "LPT", "edd": "EDD", "due date": "EDD",
              "cr": "CR", "critical ratio": "CR", "slack": "SLACK", "priority": "Priority"}


@dataclass
class Intent:
    action: str
    confidence: float = 0.0
    items: List[str] = field(default_factory=list)
    work_centres: List[str] = field(default_factory=list)
    periods: List[int] = field(default_factory=list)
    numbers: List[float] = field(default_factory=list)
    percent: Optional[float] = None
    hours: Optional[float] = None
    rule: Optional[str] = None
    scenario: Optional[str] = None
    severity: Optional[str] = None
    text: str = ""

    @property
    def item(self) -> Optional[str]:
        return self.items[0] if self.items else None

    @property
    def work_centre(self) -> Optional[str]:
        return self.work_centres[0] if self.work_centres else None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def _match_entities(text: str, known: List[str]) -> List[str]:
    """Find known codes in free text, tolerant of case and punctuation."""
    hits, low = [], _norm(text)
    squash = re.sub(r"[^a-z0-9]", "", low)
    for code in known:
        c = _norm(code)
        if c in low or re.sub(r"[^a-z0-9]", "", c) in squash:
            hits.append(code)
    if hits:
        # prefer the longest match so FG-MAG-280 beats FG-MAG
        hits.sort(key=len, reverse=True)
        keep = []
        for h in hits:
            if not any(h != k and _norm(h) in _norm(k) for k in keep):
                keep.append(h)
        return keep
    # a bare word that names a product, such as "ketchup" or "oats"
    for code in known:
        tail = _norm(code).split("-")[-1]
        if len(tail) > 3 and tail in low:
            hits.append(code)
    return hits


def parse(text: str, items: Optional[List[str]] = None,
          work_centres: Optional[List[str]] = None) -> Intent:
    low = _norm(text)
    scores: Dict[str, float] = {}
    for name, words in INTENTS.items():
        for w in words:
            if w in low:
                scores[name] = scores.get(name, 0.0) + 1.0 + 0.05 * len(w)

    found_items = _match_entities(text, items or [])
    found_wcs = _match_entities(text, work_centres or [])

    intent = Intent(action="status", text=text)
    intent.items = found_items
    intent.work_centres = found_wcs

    # periods: "period 40", "p40", "week 41"
    for m in re.finditer(r"\b(?:period|week|p|wk)\s*#?\s*(\d{1,4})\b", low):
        intent.periods.append(int(m.group(1)))
    # the other way round: "a delay of 3 periods", "2 weeks late"
    for m in re.finditer(r"\b(\d{1,4})\s*(?:period|week)s?\b", low):
        v = int(m.group(1))
        if v not in intent.periods:
            intent.periods.append(v)
    # percentages and hours
    pm = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent|per cent)", low)
    if pm:
        intent.percent = float(pm.group(1))
    hm = re.search(r"(\d+(?:\.\d+)?)\s*(?:h\b|hr|hrs|hour)", low)
    if hm:
        intent.hours = float(hm.group(1))
    intent.numbers = [float(x) for x in re.findall(r"\b\d+(?:\.\d+)?\b", low)][:4]

    # "what if" is a mode rather than a topic, so it outranks the topic word
    # that happens to appear in the same sentence
    if re.search(r"\b(what if|scenario|simulate|suppose|assume|imagine)\b", low):
        scores["scenario"] = scores.get("scenario", 0.0) + 2.5

    for word, rule in RULE_WORDS.items():
        if word in low:
            intent.rule = rule
            scores["rules"] = scores.get("rules", 0.0) + 1.2
            break
    for kind, words in SCENARIO_KINDS.items():
        if any(w in low for w in words):
            intent.scenario = kind
            scores["scenario"] = scores.get("scenario", 0.0) + 2.0
            break
    for sev in ("high", "medium", "low"):
        if re.search(rf"\b{sev}\b(?:\s+(?:severity|priority))?", low) and "exception" in low:
            intent.severity = sev.capitalize()

    # an item named with no other signal is a request for a briefing on it
    if found_items and not scores:
        scores["item"] = 2.0
    if found_wcs and not scores:
        scores["capacity"] = 2.0

    if scores:
        best = max(scores.items(), key=lambda kv: kv[1])
        intent.action, intent.confidence = best[0], best[1]
        # a question about a named item is a briefing unless it names a module
        if (found_items and intent.action == "status"):
            intent.action = "item"
    else:
        intent.action, intent.confidence = ("help" if "?" not in low and len(low) < 4
                                            else "status"), 0.0
    return intent


SUGGESTIONS = [
    "Which items are projected to stock out?",
    "How is the plan looking?",
    "What should I fix first?",
    "Why is anything running late?",
    "Where is the bottleneck?",
    "Tell me everything about FG-MAG-70",
    "Which dispatching rule should we use?",
    "What happens if demand surges 30%?",
    "What if WC-SHEET goes down for 40 hours?",
    "Are any components short?",
    "How accurate is the forecast?",
]
