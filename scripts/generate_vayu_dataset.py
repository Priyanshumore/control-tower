"""
Generate a second synthetic dataset: Vayu Mobility, an electric two wheeler
assembly plant at Chakan.

This exists to prove the control tower is not tuned to one dataset. It is
deliberately unlike the Maggi data in every way that matters:

  process type   discrete assembly rather than food process
  structure      four levels with heavily shared sub assemblies
  numbering      fiscal weeks 201 to 248 rather than periods 1 to 48
  headers        engineering and works order vocabulary, not FMCG vocabulary
  constraint     end of line testing, not a mixing or sheeting line
  setups         paint booth colour changeovers dominate, so setup time matters

Vayu Mobility is a fictitious company. Every figure here is generated from a
seeded random model for teaching purposes.

Run:  python scripts/generate_vayu_dataset.py
Output: data/vayu_mobility/*.csv
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

SEED = 7
HIST = 36
PLAN = 12
FIRST_WEEK = 201                       # fiscal week numbering starts here
FIRST_PLAN_WEEK = FIRST_WEEK + HIST    # W237
LAST_WEEK = FIRST_WEEK + HIST + PLAN - 1
W201_START = date(2026, 1, 5)          # so W237 begins 2026-09-07

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(HERE), "data", "vayu_mobility")


def week_start(w: int) -> date:
    return W201_START + timedelta(weeks=w - FIRST_WEEK)


# ---------------------------------------------------------------------------
# Item master
# ---------------------------------------------------------------------------
# code, name, category, unit, lead weeks, policy, batch, std cost, source
ITEMS = [
    ("EV-CITY-2W", "Vayu City 2W electric scooter", "Finished Good", "NOS", 1, "LFL", 0, 62000, "Make"),
    ("EV-PRO-2W", "Vayu Pro 2W performance scooter", "Finished Good", "NOS", 1, "LFL", 0, 78000, "Make"),
    ("EV-CARGO-3W", "Vayu Cargo 3W load carrier", "Finished Good", "NOS", 1, "FOQ", 60, 145000, "Make"),
    ("SP-BATT-KIT", "Service battery kit 48V", "Finished Good", "NOS", 1, "EOQ", 0, 21000, "Make"),

    ("SA-FRAME-C", "Frame assembly city", "Sub Assembly", "NOS", 1, "LFL", 0, 6400, "Make"),
    ("SA-FRAME-P", "Frame assembly pro", "Sub Assembly", "NOS", 1, "LFL", 0, 7100, "Make"),
    ("SA-FRAME-G", "Frame assembly cargo", "Sub Assembly", "NOS", 1, "FOQ", 60, 12800, "Make"),
    ("SA-BATT-48", "Battery pack 48V 2.2kWh", "Sub Assembly", "NOS", 1, "LFL", 0, 18500, "Make"),
    ("SA-BATT-60", "Battery pack 60V 3.0kWh", "Sub Assembly", "NOS", 1, "LFL", 0, 24200, "Make"),
    ("SA-MOTOR-2K", "Hub motor 2.0 kW", "Sub Assembly", "NOS", 1, "LFL", 0, 9800, "Make"),
    ("SA-MOTOR-3K", "Hub motor 3.0 kW", "Sub Assembly", "NOS", 1, "LFL", 0, 13400, "Make"),
    ("SA-CONTROL", "Motor controller unit", "Sub Assembly", "NOS", 1, "POQ", 2, 4300, "Make"),
    ("SA-HARNESS", "Main wiring harness", "Sub Assembly", "NOS", 1, "LFL", 0, 1250, "Make"),

    ("PCB-CTRL", "Controller PCB populated", "Component", "NOS", 1, "FOQ", 400, 1850, "Make"),
    ("TUBE-STEEL", "Steel tube 32mm ERW", "Raw Material", "KG", 2, "EOQ", 0, 78, "Buy"),
    ("SHEET-STEEL", "Steel sheet 1.6mm CRCA", "Raw Material", "KG", 2, "EOQ", 0, 71, "Buy"),
    ("CELL-21700", "Li-ion cell 21700 5Ah", "Component", "NOS", 4, "FOQ", 20000, 210, "Buy"),
    ("BMS-BOARD", "Battery management board", "Component", "NOS", 3, "FOQ", 1500, 2100, "Buy"),
    ("CASE-ALU", "Aluminium pack casing", "Component", "NOS", 2, "EOQ", 0, 1450, "Buy"),
    ("STATOR-LAM", "Stator lamination stack", "Component", "NOS", 3, "FOQ", 1200, 1900, "Buy"),
    ("MAGNET-NDFEB", "Neodymium magnet segment", "Component", "NOS", 5, "FOQ", 8000, 96, "Buy"),
    ("BEARING-SET", "Bearing set 6203", "Component", "SET", 2, "EOQ", 0, 240, "Buy"),
    ("WIRE-CU", "Copper winding wire 1.2mm", "Raw Material", "KG", 3, "EOQ", 0, 810, "Buy"),
    ("WIRE-LOOM", "Automotive loom cable", "Raw Material", "MTR", 2, "EOQ", 0, 34, "Buy"),
    ("CONN-SET", "Connector set waterproof", "Component", "SET", 2, "FOQ", 3000, 165, "Buy"),
    ("IC-MCU", "Motor control microcontroller", "Component", "NOS", 4, "FOQ", 2500, 620, "Buy"),
    ("PCB-BLANK", "Bare 4 layer PCB", "Component", "NOS", 3, "FOQ", 2000, 210, "Buy"),
    ("CASE-PLASTIC", "Controller enclosure ABS", "Component", "NOS", 2, "EOQ", 0, 190, "Buy"),
    ("RM-PAINT", "Powder coat paint", "Raw Material", "KG", 2, "EOQ", 0, 385, "Buy"),
    ("RM-SOLDER", "Lead free solder paste", "Raw Material", "KG", 3, "EOQ", 0, 2650, "Buy"),
    ("CONS-WELDROD", "MIG welding wire", "Consumable", "KG", 2, "EOQ", 0, 168, "Buy"),
    ("CONS-FASTENER", "Fastener kit per vehicle", "Consumable", "SET", 2, "FOQ", 5000, 145, "Buy"),
    ("CONS-TAPE", "Harness wrapping tape", "Consumable", "ROL", 1, "EOQ", 0, 62, "Buy"),
    ("PK-CARTON-2W", "Export carton two wheeler", "Packaging", "NOS", 2, "FOQ", 2000, 420, "Buy"),
    ("PK-CARTON-3W", "Export carton three wheeler", "Packaging", "NOS", 2, "FOQ", 500, 760, "Buy"),
    ("PK-BOX-BATT", "Battery service box", "Packaging", "NOS", 2, "FOQ", 1500, 180, "Buy"),
]

# assembly, sub part, units per, rejection %
BOM = [
    ("EV-CITY-2W", "SA-FRAME-C", 1, 0.5), ("EV-CITY-2W", "SA-BATT-48", 1, 0.3),
    ("EV-CITY-2W", "SA-MOTOR-2K", 1, 0.4), ("EV-CITY-2W", "SA-HARNESS", 1, 0.6),
    ("EV-CITY-2W", "SA-CONTROL", 1, 0.5), ("EV-CITY-2W", "PK-CARTON-2W", 1, 0.8),
    ("EV-CITY-2W", "CONS-FASTENER", 1, 1.5),

    ("EV-PRO-2W", "SA-FRAME-P", 1, 0.5), ("EV-PRO-2W", "SA-BATT-60", 1, 0.3),
    ("EV-PRO-2W", "SA-MOTOR-3K", 1, 0.4), ("EV-PRO-2W", "SA-HARNESS", 1, 0.6),
    ("EV-PRO-2W", "SA-CONTROL", 1, 0.5), ("EV-PRO-2W", "PK-CARTON-2W", 1, 0.8),
    ("EV-PRO-2W", "CONS-FASTENER", 1, 1.5),

    ("EV-CARGO-3W", "SA-FRAME-G", 1, 0.6), ("EV-CARGO-3W", "SA-BATT-60", 2, 0.3),
    ("EV-CARGO-3W", "SA-MOTOR-3K", 1, 0.4), ("EV-CARGO-3W", "SA-HARNESS", 1, 0.6),
    ("EV-CARGO-3W", "SA-CONTROL", 1, 0.5), ("EV-CARGO-3W", "PK-CARTON-3W", 1, 0.8),
    ("EV-CARGO-3W", "CONS-FASTENER", 2, 1.5),

    ("SP-BATT-KIT", "SA-BATT-48", 1, 0.2), ("SP-BATT-KIT", "PK-BOX-BATT", 1, 0.5),

    ("SA-FRAME-C", "TUBE-STEEL", 11.5, 3.0), ("SA-FRAME-C", "SHEET-STEEL", 4.2, 4.0),
    ("SA-FRAME-C", "RM-PAINT", 0.35, 6.0), ("SA-FRAME-C", "CONS-WELDROD", 0.22, 2.0),
    ("SA-FRAME-P", "TUBE-STEEL", 12.8, 3.0), ("SA-FRAME-P", "SHEET-STEEL", 4.6, 4.0),
    ("SA-FRAME-P", "RM-PAINT", 0.38, 6.0), ("SA-FRAME-P", "CONS-WELDROD", 0.25, 2.0),
    ("SA-FRAME-G", "TUBE-STEEL", 24.0, 3.0), ("SA-FRAME-G", "SHEET-STEEL", 9.5, 4.0),
    ("SA-FRAME-G", "RM-PAINT", 0.62, 6.0), ("SA-FRAME-G", "CONS-WELDROD", 0.45, 2.0),

    ("SA-BATT-48", "CELL-21700", 48, 1.2), ("SA-BATT-48", "BMS-BOARD", 1, 0.8),
    ("SA-BATT-48", "CASE-ALU", 1, 0.5), ("SA-BATT-48", "WIRE-CU", 0.6, 2.5),
    ("SA-BATT-60", "CELL-21700", 60, 1.2), ("SA-BATT-60", "BMS-BOARD", 1, 0.8),
    ("SA-BATT-60", "CASE-ALU", 1, 0.5), ("SA-BATT-60", "WIRE-CU", 0.8, 2.5),

    ("SA-MOTOR-2K", "STATOR-LAM", 1, 0.7), ("SA-MOTOR-2K", "MAGNET-NDFEB", 12, 1.0),
    ("SA-MOTOR-2K", "WIRE-CU", 1.2, 2.5), ("SA-MOTOR-2K", "BEARING-SET", 1, 0.4),
    ("SA-MOTOR-3K", "STATOR-LAM", 1, 0.7), ("SA-MOTOR-3K", "MAGNET-NDFEB", 16, 1.0),
    ("SA-MOTOR-3K", "WIRE-CU", 1.6, 2.5), ("SA-MOTOR-3K", "BEARING-SET", 1, 0.4),

    ("SA-CONTROL", "PCB-CTRL", 1, 0.9), ("SA-CONTROL", "CONN-SET", 1, 0.6),
    ("SA-CONTROL", "CASE-PLASTIC", 1, 0.7),
    ("SA-HARNESS", "WIRE-LOOM", 3.2, 3.5), ("SA-HARNESS", "CONN-SET", 2, 0.6),
    ("SA-HARNESS", "CONS-TAPE", 0.2, 2.0),

    ("PCB-CTRL", "IC-MCU", 1, 1.1), ("PCB-CTRL", "PCB-BLANK", 1, 1.4),
    ("PCB-CTRL", "RM-SOLDER", 0.018, 5.0),
]

# code, name, stations, hours per shift, shifts, days, OEE %
WORK_CENTRES = [
    ("WC-CUT", "Tube cutting and bending", 2, 8, 2, 6, 84),
    ("WC-WELD", "Frame welding cell", 4, 8, 2, 6, 80),
    ("WC-PAINT", "Powder coat booth", 1, 8, 2, 6, 76),
    ("WC-CELL", "Cell sorting and tabbing", 2, 8, 3, 6, 88),
    ("WC-PACKASM", "Battery pack assembly", 3, 8, 2, 6, 85),
    ("WC-WIND", "Stator winding", 3, 8, 2, 6, 82),
    ("WC-MOTASM", "Motor assembly", 2, 8, 2, 6, 86),
    ("WC-SMT", "SMT and electronics", 2, 8, 2, 6, 90),
    ("WC-HARN", "Harness build", 2, 8, 2, 6, 87),
    ("WC-FINAL", "Final vehicle assembly", 3, 8, 2, 6, 83),
    ("WC-TEST", "End of line test and homologation", 2, 8, 2, 6, 79),
    ("WC-PACK", "Packing and despatch", 2, 8, 2, 6, 88),
]

# item, op number, op name, cost centre, setup mins, cycle mins per unit
ROUTING = [
    ("EV-CITY-2W", 10, "Final assembly", "WC-FINAL", 20, 6.20),
    ("EV-CITY-2W", 20, "End of line test", "WC-TEST", 12, 4.40),
    ("EV-CITY-2W", 30, "Pack and despatch", "WC-PACK", 8, 1.40),
    ("EV-PRO-2W", 10, "Final assembly", "WC-FINAL", 20, 6.90),
    ("EV-PRO-2W", 20, "End of line test", "WC-TEST", 12, 5.15),
    ("EV-PRO-2W", 30, "Pack and despatch", "WC-PACK", 8, 1.40),
    ("EV-CARGO-3W", 10, "Final assembly", "WC-FINAL", 30, 11.50),
    ("EV-CARGO-3W", 20, "End of line test", "WC-TEST", 18, 8.10),
    ("EV-CARGO-3W", 30, "Pack and despatch", "WC-PACK", 10, 2.60),
    ("SP-BATT-KIT", 10, "Kit build and pack", "WC-PACK", 10, 1.10),

    ("SA-FRAME-C", 10, "Cut and bend", "WC-CUT", 15, 1.70),
    ("SA-FRAME-C", 20, "Weld frame", "WC-WELD", 25, 4.30),
    ("SA-FRAME-C", 30, "Powder coat", "WC-PAINT", 45, 2.10),
    ("SA-FRAME-P", 10, "Cut and bend", "WC-CUT", 15, 1.85),
    ("SA-FRAME-P", 20, "Weld frame", "WC-WELD", 25, 4.60),
    ("SA-FRAME-P", 30, "Powder coat", "WC-PAINT", 45, 2.20),
    ("SA-FRAME-G", 10, "Cut and bend", "WC-CUT", 20, 3.10),
    ("SA-FRAME-G", 20, "Weld frame", "WC-WELD", 35, 8.20),
    ("SA-FRAME-G", 30, "Powder coat", "WC-PAINT", 55, 3.60),

    ("SA-BATT-48", 10, "Cell sort and tab", "WC-CELL", 20, 1.05),
    ("SA-BATT-48", 20, "Pack build and test", "WC-PACKASM", 25, 3.20),
    ("SA-BATT-60", 10, "Cell sort and tab", "WC-CELL", 20, 1.30),
    ("SA-BATT-60", 20, "Pack build and test", "WC-PACKASM", 25, 3.90),

    ("SA-MOTOR-2K", 10, "Stator winding", "WC-WIND", 30, 3.60),
    ("SA-MOTOR-2K", 20, "Rotor and housing", "WC-MOTASM", 20, 2.40),
    ("SA-MOTOR-3K", 10, "Stator winding", "WC-WIND", 30, 4.30),
    ("SA-MOTOR-3K", 20, "Rotor and housing", "WC-MOTASM", 20, 2.80),

    ("SA-CONTROL", 10, "Controller build", "WC-SMT", 35, 1.30),
    ("SA-HARNESS", 10, "Loom cut and crimp", "WC-HARN", 15, 2.00),
    ("PCB-CTRL", 10, "SMT place and reflow", "WC-SMT", 40, 0.85),
]

VENDORS = [
    ("V-101", "Deccan Cell Technologies, Hosur", ["CELL-21700"], 4, 91, 20000),
    ("V-102", "Nashik Precision Magnetics", ["MAGNET-NDFEB", "STATOR-LAM"], 5, 82, 8000),
    ("V-103", "Pune Sheet and Tube Works", ["TUBE-STEEL", "SHEET-STEEL"], 2, 95, 2500),
    ("V-104", "Coimbatore Winding Supplies", ["WIRE-CU", "WIRE-LOOM"], 3, 93, 400),
    ("V-105", "Bengaluru Circuit Systems", ["PCB-BLANK", "IC-MCU", "BMS-BOARD"], 4, 88, 2000),
    ("V-106", "Aurangabad Alu Castings", ["CASE-ALU", "CASE-PLASTIC"], 2, 94, 1000),
    ("V-107", "Chakan Fastener Company", ["CONS-FASTENER", "BEARING-SET", "CONN-SET"], 2, 96, 3000),
    ("V-108", "Vapi Coatings and Chemicals", ["RM-PAINT", "RM-SOLDER", "CONS-WELDROD"], 2, 90, 250),
    ("V-109", "Bhiwandi Packaging House", ["PK-CARTON-2W", "PK-CARTON-3W", "PK-BOX-BATT", "CONS-TAPE"], 2, 97, 500),
]

DEALERS = [("D-01", "Mumbai Metro EV Dealers"), ("D-02", "Pune West Mobility"),
           ("D-03", "Bengaluru Green Rides"), ("D-04", "Delhi NCR Fleet Services"),
           ("D-05", "Hyderabad Urban Motors")]


# ---------------------------------------------------------------------------
def build_demand(rng) -> pd.DataFrame:
    """Four demand shapes so method selection has something to choose between."""
    weeks = np.arange(FIRST_WEEK, LAST_WEEK + 1)
    rows = []
    for w in weeks:
        i = w - FIRST_WEEK                       # 0 based index

        # city scooter: steady growth with mild noise
        city = 880 + 6.2 * i + rng.normal(0, 58)

        # pro scooter: festive season lift peaking around the Diwali build
        season = 1 + 0.42 * np.sin(2 * np.pi * (i - 6) / 52) + 0.18 * np.sin(2 * np.pi * i / 13)
        pro = 430 * season + rng.normal(0, 34)

        # cargo three wheeler: fleet orders, lumpy and often zero
        cargo = 0.0
        if rng.random() < 0.55:
            cargo = rng.normal(135, 42)

        # service battery kits: erratic, with occasional warranty spikes
        spare = 250 + rng.normal(0, 46)
        if rng.random() < 0.12:
            spare *= rng.uniform(1.8, 2.6)

        for code, val in [("EV-CITY-2W", city), ("EV-PRO-2W", pro),
                          ("EV-CARGO-3W", cargo), ("SP-BATT-KIT", spare)]:
            rows.append({"Part_Code": code, "Fiscal_Week": int(w),
                         "Week_Commencing": week_start(int(w)).isoformat(),
                         "Dispatched_Units": max(0, round(float(val)))})
    return pd.DataFrame(rows)


def split_history(full: pd.DataFrame) -> pd.DataFrame:
    """Despatch history is actuals only.

    The forward weeks are generated so the dealer order book has something
    sensible to be a share of, but they are never written to file: the plant
    has not shipped them yet, and the control tower is meant to forecast them.
    """
    return full[full["Fiscal_Week"] < FIRST_PLAN_WEEK].reset_index(drop=True)


def build_orders(demand: pd.DataFrame, rng) -> pd.DataFrame:
    """Confirmed dealer orders covering the near weeks of the horizon."""
    fut = demand[demand["Fiscal_Week"] >= FIRST_PLAN_WEEK]
    rows, n = [], 0
    for _, r in fut.iterrows():
        w = int(r["Fiscal_Week"])
        horizon_pos = w - FIRST_PLAN_WEEK
        # order book is firm near in and thins out further out
        cover = max(0.0, 0.92 - 0.085 * horizon_pos)
        if cover <= 0 or r["Dispatched_Units"] <= 0:
            continue
        booked = r["Dispatched_Units"] * cover
        splits = 2 if booked > 600 else 1
        for s in range(splits):
            qty = round(booked / splits)
            if qty <= 0:
                continue
            n += 1
            d = DEALERS[rng.integers(0, len(DEALERS))]
            rows.append({
                "Sales_Order": f"SO-{25000 + n}",
                "Dealer_Code": d[0], "Dealer": d[1],
                "Part_Code": r["Part_Code"], "Fiscal_Week": w,
                "Order_Units": qty,
                "Promise_Date": week_start(w).isoformat(),
                "Urgency": int(rng.integers(1, 4)),
                "Order_Status": "Confirmed",
            })
    return pd.DataFrame(rows)


def build_item_master(demand: pd.DataFrame, rng) -> pd.DataFrame:
    """Opening stock sized off usage, with three deliberate problems seeded."""
    usage = weekly_usage(demand)
    vendor_of = {}
    for code, name, parts, lt, otp, moq in VENDORS:
        for p in parts:
            vendor_of[p] = name

    rows = []
    for (code, name, cat, unit, lt, policy, batch, cost, source) in ITEMS:
        wk = usage.get(code, 0.0)
        safety = round(wk * rng.uniform(0.9, 1.5) * max(lt, 1) ** 0.5)
        if source == "Make":
            on_hand = round(wk * rng.uniform(0.8, 1.6))
        else:
            on_hand = round(safety + wk * lt * rng.uniform(0.75, 1.25))

        # seeded exceptions
        if code == "CELL-21700":
            on_hand = round(safety * 0.38)          # the constraint material
        if code == "MAGNET-NDFEB":
            on_hand = round(safety * 0.52)          # long lead, weak vendor
        if code == "CASE-ALU":
            on_hand = round(safety * 4.1)           # money sitting idle

        rows.append({
            "Part_Code": code, "Part_Name": name, "Part_Category": cat, "Unit": unit,
            "Stock_On_Hand": int(max(on_hand, 0)), "Min_Stock": int(max(safety, 0)),
            "Lead_Time_Weeks": lt, "Order_Policy": policy, "Batch_Qty": batch,
            "Std_Cost_INR": cost,
            "Setup_Cost_INR": int(round(cost * rng.uniform(0.9, 1.6) + 900)),
            "Carrying_Rate": 0.24,
            "Vendor": vendor_of.get(code, "" if source == "Make" else "V-107"),
            "Shelf_Life_Weeks": 0 if cat != "Raw Material" else int(rng.integers(26, 105)),
            "Source": source,
        })
    return pd.DataFrame(rows)


def weekly_usage(demand: pd.DataFrame) -> dict:
    """Average weekly usage of every part, exploded down the bill.

    Parents are walked in level order so a part shared by several assemblies,
    such as the copper wire that goes into both batteries and motors, collects
    the demand from all of them.
    """
    hist = demand[demand["Fiscal_Week"] < FIRST_PLAN_WEEK]
    usage = hist.groupby("Part_Code")["Dispatched_Units"].mean().to_dict()

    children = {}
    for parent, comp, qty, scrap in BOM:
        children.setdefault(parent, []).append((comp, qty, scrap))

    level_order = ["EV-CITY-2W", "EV-PRO-2W", "EV-CARGO-3W", "SP-BATT-KIT",
                   "SA-FRAME-C", "SA-FRAME-P", "SA-FRAME-G", "SA-BATT-48", "SA-BATT-60",
                   "SA-MOTOR-2K", "SA-MOTOR-3K", "SA-CONTROL", "SA-HARNESS", "PCB-CTRL"]
    for parent in level_order:
        parent_usage = usage.get(parent, 0.0)
        for comp, qty, scrap in children.get(parent, []):
            usage[comp] = usage.get(comp, 0.0) + parent_usage * qty * (1 + scrap / 100.0)
    return usage


def build_vendor_master() -> pd.DataFrame:
    rows = []
    for code, name, parts, lt, otp, moq in VENDORS:
        for p in parts:
            rows.append({"Vendor_Code": code, "Vendor": name, "Part_Code": p,
                         "Lead_Time_Weeks": lt, "On_Time_Pct": otp, "Min_Order_Qty": moq})
    return pd.DataFrame(rows)


def build_receipts(item_master: pd.DataFrame, rng) -> pd.DataFrame:
    """Purchase orders already placed and landing in the first weeks."""
    rows, n = [], 0
    buys = item_master[item_master["Source"] == "Buy"]
    for _, r in buys.iterrows():
        if rng.random() < 0.45:
            continue
        n += 1
        qty = r["Batch_Qty"] if r["Batch_Qty"] > 0 else round(r["Min_Stock"] * rng.uniform(1.0, 2.0))
        rows.append({
            "Part_Code": r["Part_Code"],
            "Fiscal_Week": int(FIRST_PLAN_WEEK + rng.integers(0, 4)),
            "Inbound_Units": int(max(qty, 1)),
            "GRN_Ref": f"PO-{7100 + n}", "Vendor": r["Vendor"],
        })
    return pd.DataFrame(rows)


def build_works_orders(rng) -> pd.DataFrame:
    """Works orders already on the floor when the horizon opens."""
    rows = [
        ("WO-9001", "EV-CITY-2W", 320, FIRST_PLAN_WEEK, FIRST_PLAN_WEEK, 1, "Released"),
        ("WO-9002", "SA-BATT-48", 420, FIRST_PLAN_WEEK, FIRST_PLAN_WEEK, 1, "Released"),
        ("WO-9003", "SA-FRAME-C", 380, FIRST_PLAN_WEEK, FIRST_PLAN_WEEK + 1, 2, "Released"),
        ("WO-9004", "EV-PRO-2W", 180, FIRST_PLAN_WEEK + 1, FIRST_PLAN_WEEK + 1, 2, "Firm Planned"),
        ("WO-9005", "SA-MOTOR-2K", 300, FIRST_PLAN_WEEK, FIRST_PLAN_WEEK + 1, 3, "Firm Planned"),
    ]
    return pd.DataFrame(rows, columns=["Works_Order", "Part_Code", "Order_Units",
                                       "Start_Week", "Finish_Week", "Urgency", "Order_Status"])


def main() -> None:
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    full_demand = build_demand(rng)
    demand = split_history(full_demand)
    orders = build_orders(full_demand, rng)
    item_master = build_item_master(demand, rng)
    bom = pd.DataFrame([{"Assembly": p, "Sub_Part": c, "Units_Per": q,
                         "Unit": dict((i[0], i[3]) for i in ITEMS).get(c, "NOS"),
                         "Rejection_Pct": s} for p, c, q, s in BOM])
    routing = pd.DataFrame(ROUTING, columns=["Part_Code", "Opn_No", "Opn_Name",
                                             "Cost_Centre", "Setup_Mins", "Cycle_Time_Mins"])
    capacity = pd.DataFrame(WORK_CENTRES, columns=["Cost_Centre", "Cost_Centre_Name",
                                                   "No_Of_Stations", "Hours_Per_Shift",
                                                   "Shifts", "Working_Days", "OEE_Pct"])
    vendors = build_vendor_master()
    receipts = build_receipts(item_master, rng)
    works = build_works_orders(rng)
    calendar = pd.DataFrame([{"Fiscal_Week": w,
                              "Week_Commencing": week_start(w).isoformat(),
                              "Week_Ending": (week_start(w) + timedelta(days=6)).isoformat(),
                              "Bucket": "History" if w < FIRST_PLAN_WEEK else "Plan"}
                             for w in range(FIRST_WEEK, LAST_WEEK + 1)])
    build_plan = pd.DataFrame([{"Part_Code": c, "Fiscal_Week": w, "Build_Plan": ""}
                               for c in ["EV-CITY-2W", "EV-PRO-2W", "EV-CARGO-3W", "SP-BATT-KIT"]
                               for w in range(FIRST_PLAN_WEEK, LAST_WEEK + 1)])

    files = {
        "Despatch_History.csv": demand,
        "Dealer_Orders.csv": orders,
        "Part_Master.csv": item_master,
        "Bill_Of_Materials.csv": bom,
        "Vendor_Master.csv": vendors,
        "Process_Routing.csv": routing,
        "Cost_Centre_Capacity.csv": capacity,
        "Works_Orders.csv": works,
        "Inbound_Receipts.csv": receipts,
        "Build_Plan.csv": build_plan,
        "Fiscal_Calendar.csv": calendar,
    }
    for name, df in files.items():
        df.to_csv(os.path.join(OUT_DIR, name), index=False)
        print(f"  {name:<28} {len(df):>5} rows")
    print(f"\nWritten to {OUT_DIR}")
    print(f"Fiscal weeks {FIRST_WEEK} to {LAST_WEEK}, planning opens at W{FIRST_PLAN_WEEK} "
          f"({week_start(FIRST_PLAN_WEEK)})")


if __name__ == "__main__":
    main()
