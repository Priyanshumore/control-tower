"""
The contract every agent in the control tower keeps.

An agent owns one planning function, answers questions on its own topics, and is
free to consult other agents before answering. Nothing here knows about
Streamlit, so the same roster runs from a script or a notebook.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from .bus import Blackboard, MessageBus


class Agent:
    name: str = "Agent"
    role: str = ""
    owns: str = ""                 # the module of the brief this agent owns
    skills: List[str] = []         # topics it will answer

    def __init__(self, bus: MessageBus, board: Blackboard):
        self.bus = bus
        self.board = board
        self.calls = 0
        bus.register(self)

    # -- talking to the rest of the roster ---------------------------------
    def ask(self, recipient: str, topic: str, text: str = "", **payload) -> Dict[str, Any]:
        return self.bus.request(self.name, recipient, topic, text, **payload)

    def say(self, text: str, to: str = "Coordinator", **data) -> None:
        self.bus.inform(self.name, text, to, **data)

    def escalate(self, text: str, **data) -> None:
        self.bus.escalate(self.name, text, **data)

    # -- being asked -------------------------------------------------------
    def handle(self, topic: str, payload: Dict[str, Any], sender: str) -> Dict[str, Any]:
        self.calls += 1
        fn = getattr(self, "do_" + topic.split(".", 1)[-1], None)
        if fn is None:
            return {"ok": False, "summary": f"{self.name} does not handle {topic}"}
        return fn(payload, sender)

    # -- convenience -------------------------------------------------------
    @property
    def plan(self) -> Dict[str, Any]:
        return self.board.get("plan", {}) or {}

    def table(self, key: str) -> pd.DataFrame:
        v = self.plan.get(key)
        return v if isinstance(v, pd.DataFrame) else pd.DataFrame()

    @property
    def summary(self) -> Dict[str, Any]:
        return self.plan.get("kpi_summary", {}) or {}

    def status_line(self) -> str:
        return self.role

    def card(self) -> Dict[str, Any]:
        return {"Agent": self.name, "Owns": self.owns, "Role": self.role,
                "Questions answered": self.calls}


def fmt(value: float, unit: str = "", dp: int = 0) -> str:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "not available"
        return f"{value:,.{dp}f}{unit}"
    except (TypeError, ValueError):
        return str(value)


def money(value: float) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(v) >= 1e7:
        return f"INR {v / 1e7:,.2f} Cr"
    if abs(v) >= 1e5:
        return f"INR {v / 1e5:,.2f} L"
    return f"INR {v:,.0f}"
