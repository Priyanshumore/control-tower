"""
The wiring between agents: a message bus and a shared blackboard.

Every question one agent asks another goes through the bus and is recorded, so
the transcript shown in the dashboard is the actual conversation that produced
the plan rather than a description of one written afterwards.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

REQUEST = "request"
RESPONSE = "response"
INFORM = "inform"
ESCALATE = "escalate"
BROADCAST = "broadcast"


@dataclass
class Message:
    seq: int
    sender: str
    recipient: str
    kind: str
    topic: str
    text: str
    data: Dict[str, Any] = field(default_factory=dict)
    in_reply_to: Optional[int] = None
    elapsed_ms: float = 0.0

    def arrow(self) -> str:
        return f"{self.sender} -> {self.recipient}"


class MessageBus:
    """Routes messages between agents and keeps the transcript."""

    def __init__(self, verbose_topics: Optional[List[str]] = None):
        self._agents: Dict[str, Any] = {}
        self.log: List[Message] = []
        self._seq = 0
        # high frequency topics are counted rather than written out in full, so
        # one levelling pass does not bury the conversation in noise
        self.quiet_topics = set(verbose_topics or
                                ["capacity.spare", "capacity.reserve", "material.check"])
        self.counters: Dict[str, int] = {}

    # -- registration ------------------------------------------------------
    def register(self, agent) -> None:
        self._agents[agent.name] = agent

    def agent(self, name: str):
        return self._agents.get(name)

    def names(self) -> List[str]:
        return list(self._agents)

    # -- messaging ---------------------------------------------------------
    def _record(self, sender, recipient, kind, topic, text, data, in_reply_to, elapsed):
        self._seq += 1
        key = f"{sender}->{recipient}:{topic}"
        self.counters[key] = self.counters.get(key, 0) + 1
        quiet = topic in self.quiet_topics
        if quiet and self.counters[key] > 3:
            return None
        msg = Message(self._seq, sender, recipient, kind, topic,
                      text + (" (further identical exchanges summarised)"
                              if quiet and self.counters[key] == 3 else ""),
                      data or {}, in_reply_to, elapsed)
        self.log.append(msg)
        return msg

    def request(self, sender: str, recipient: str, topic: str,
                text: str = "", **payload) -> Dict[str, Any]:
        """Ask another agent something and wait for the answer."""
        target = self._agents.get(recipient)
        sent = self._record(sender, recipient, REQUEST, topic,
                            text or f"asks about {topic}", payload, None, 0.0)
        if target is None:
            self._record(recipient, sender, RESPONSE, topic,
                         "no such agent is registered", {}, None, 0.0)
            return {"ok": False, "error": f"unknown agent {recipient}",
                    "summary": f"no agent named {recipient} is registered"}
        t0 = time.perf_counter()
        try:
            reply = target.handle(topic, payload, sender)
        except Exception as exc:                      # an agent failing is news
            self._record(recipient, sender, ESCALATE, topic,
                         f"could not answer: {exc}", {}, sent.seq if sent else None, 0.0)
            return {"ok": False, "error": str(exc),
                    "summary": f"{recipient} could not answer that ({exc})"}
        dt = (time.perf_counter() - t0) * 1000
        reply = reply if isinstance(reply, dict) else {"ok": True, "value": reply}
        reply.setdefault("ok", True)
        self._record(recipient, sender, RESPONSE, topic,
                     reply.get("summary", "answered"), 
                     {k: v for k, v in reply.items() if k not in ("summary",)},
                     sent.seq if sent else None, dt)
        return reply

    def inform(self, sender: str, text: str, recipient: str = "Coordinator",
               topic: str = "note", **data) -> None:
        self._record(sender, recipient, INFORM, topic, text, data, None, 0.0)

    def escalate(self, sender: str, text: str, **data) -> None:
        self._record(sender, "Coordinator", ESCALATE, "exception", text, data, None, 0.0)

    # -- reading the transcript -------------------------------------------
    def transcript(self, limit: Optional[int] = None) -> List[Message]:
        return self.log[-limit:] if limit else list(self.log)

    def traffic(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for m in self.log:
            out[m.sender] = out.get(m.sender, 0) + 1
        return out

    def reset(self) -> None:
        self.log.clear()
        self.counters.clear()
        self._seq = 0


class Blackboard:
    """Everything the agents have established so far, in one place."""

    def __init__(self):
        self._store: Dict[str, Any] = {}
        self.history: List[str] = []

    def put(self, key: str, value: Any, by: str = "") -> None:
        self._store[key] = value
        self.history.append(f"{by} wrote {key}")

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def has(self, key: str) -> bool:
        return key in self._store

    def keys(self) -> List[str]:
        return sorted(self._store)
