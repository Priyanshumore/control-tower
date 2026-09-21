"""
Module 6: bill of material handling.

Low level coding is the backbone of the whole planning run. An item's low
level code is the deepest level at which it appears anywhere in the product
structure. Processing items in ascending low level code order guarantees that
every parent has accumulated all of its requirements before it is exploded,
which is what keeps shared components such as the tastemaker sachet correct.
"""

from __future__ import annotations

from typing import Dict, List, Set

import pandas as pd

MAX_LEVELS = 25


def low_level_codes(bom: pd.DataFrame, all_items: List[str] | None = None) -> Dict[str, int]:
    """item -> deepest level in the structure (0 for top level items)."""
    if bom is None or len(bom) == 0:
        return {i: 0 for i in (all_items or [])}

    children: Dict[str, List[str]] = {}
    for parent, g in bom.groupby("parent"):
        children[parent] = list(g["component"])

    all_nodes: Set[str] = set(bom["parent"]) | set(bom["component"])
    if all_items:
        all_nodes |= set(all_items)
    components = set(bom["component"])
    roots = [n for n in all_nodes if n not in components]

    llc: Dict[str, int] = {n: 0 for n in all_nodes}

    # breadth first relaxation: an item sits at least one level below any parent
    frontier = [(r, 0) for r in roots] or [(n, 0) for n in all_nodes]
    guard = 0
    while frontier and guard < MAX_LEVELS * max(1, len(all_nodes)):
        node, lvl = frontier.pop(0)
        guard += 1
        for c in children.get(node, []):
            if llc.get(c, 0) < lvl + 1 and lvl + 1 <= MAX_LEVELS:
                llc[c] = lvl + 1
                frontier.append((c, lvl + 1))
    return llc


def explosion_order(bom: pd.DataFrame, all_items: List[str] | None = None) -> List[str]:
    """Items sorted by low level code, so parents always come before children."""
    llc = low_level_codes(bom, all_items)
    return [i for i, _ in sorted(llc.items(), key=lambda kv: (kv[1], kv[0]))]


def indented_bom(bom: pd.DataFrame, parent: str, qty: float = 1.0,
                 level: int = 0, seen: Set[str] | None = None) -> List[dict]:
    """Classic indented bill for display, quantities scaled to `qty` parents."""
    seen = seen or set()
    rows: List[dict] = []
    if level > MAX_LEVELS or parent in seen:
        return rows
    seen = seen | {parent}
    for _, r in bom[bom["parent"] == parent].iterrows():
        scrap = float(r.get("scrap_pct", 0) or 0) / 100.0
        need = qty * float(r["qty_per"]) * (1 + scrap)
        rows.append({
            "level": level + 1,
            "indent": "    " * level + "|- " + str(r["component"]),
            "parent": parent,
            "component": r["component"],
            "qty_per": float(r["qty_per"]),
            "scrap_pct": float(r.get("scrap_pct", 0) or 0),
            "extended_qty": round(need, 4),
            "uom": r.get("uom", ""),
        })
        rows.extend(indented_bom(bom, r["component"], need, level + 1, seen))
    return rows


def explode(mps: pd.DataFrame, bom: pd.DataFrame,
            qty_col: str = "mps_qty") -> pd.DataFrame:
    """Single pass gross requirement explosion of a schedule through the BOM.

    This is the module 6 view: how much of every component the schedule needs,
    by period, ignoring stock and lead times. The time phased netting happens
    in the MRP module.
    """
    if len(mps) == 0 or len(bom) == 0:
        return pd.DataFrame(columns=["item", "period", "gross_requirement", "level", "source"])

    order = explosion_order(bom, list(mps["item"].unique()))
    llc = low_level_codes(bom, list(mps["item"].unique()))

    req: Dict[tuple, float] = {}
    for _, r in mps.iterrows():
        if float(r[qty_col]) != 0:
            key = (r["item"], int(r["period"]))
            req[key] = req.get(key, 0.0) + float(r[qty_col])

    rows = []
    for item in order:
        kids = bom[bom["parent"] == item]
        for (it, p), q in list(req.items()):
            if it != item or q == 0:
                continue
            for _, k in kids.iterrows():
                scrap = float(k.get("scrap_pct", 0) or 0) / 100.0
                need = q * float(k["qty_per"]) * (1 + scrap)
                key = (k["component"], p)
                req[key] = req.get(key, 0.0) + need
                rows.append({
                    "parent": item, "item": k["component"], "period": p,
                    "qty_per": float(k["qty_per"]), "scrap_pct": float(k.get("scrap_pct", 0) or 0),
                    "gross_requirement": round(need, 4),
                    "level": llc.get(k["component"], 1),
                })

    detail = pd.DataFrame(rows)
    if len(detail) == 0:
        return detail
    summary = (detail.groupby(["item", "period", "level"], as_index=False)["gross_requirement"]
               .sum().sort_values(["level", "item", "period"]))
    summary["gross_requirement"] = summary["gross_requirement"].round(2)
    return summary


def explode_detail(mps: pd.DataFrame, bom: pd.DataFrame, qty_col: str = "mps_qty") -> pd.DataFrame:
    """Same explosion but keeping the parent to component pegging."""
    if len(mps) == 0 or len(bom) == 0:
        return pd.DataFrame(columns=["parent", "item", "period", "gross_requirement"])
    order = explosion_order(bom, list(mps["item"].unique()))
    req: Dict[tuple, float] = {}
    for _, r in mps.iterrows():
        if float(r[qty_col]) != 0:
            key = (r["item"], int(r["period"]))
            req[key] = req.get(key, 0.0) + float(r[qty_col])
    rows = []
    for item in order:
        kids = bom[bom["parent"] == item]
        if not len(kids):
            continue
        for (it, p), q in list(req.items()):
            if it != item or q == 0:
                continue
            for _, k in kids.iterrows():
                scrap = float(k.get("scrap_pct", 0) or 0) / 100.0
                need = q * float(k["qty_per"]) * (1 + scrap)
                req[(k["component"], p)] = req.get((k["component"], p), 0.0) + need
                rows.append({"parent": item, "item": k["component"], "period": p,
                             "gross_requirement": round(need, 4)})
    return pd.DataFrame(rows)
