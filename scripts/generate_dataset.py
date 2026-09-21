"""
Generate the synthetic Nestle Maggi dataset for the Integrated Manufacturing
Operations Control Tower.

All figures produced here are SYNTHETIC and generated from a seeded random
model. They are illustrative teaching data, not Nestle company data.

Run:  python scripts/generate_dataset.py
Output: data/nestle_maggi/*.csv
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from datetime import date, timedelta

SEED = 42
HIST_PERIODS = 36          # 36 weekly buckets of history (brief asks for 24 to 36)
PLAN_PERIODS = 12          # 12 week forward planning horizon
FIRST_PLAN_PERIOD = HIST_PERIODS + 1        # P37
LAST_PLAN_PERIOD = HIST_PERIODS + PLAN_PERIODS  # P48

# P1 starts on this Monday. P37 (first planning week) therefore starts 2026-09-07.
P1_START = date(2025, 12, 29)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(HERE), "data", "nestle_maggi")


def period_start(p: int) -> date:
    return P1_START + timedelta(weeks=p - 1)


def period_end(p: int) -> date:
    return period_start(p) + timedelta(days=6)


# ----------------------------------------------------------------------------
# 1. Item master definition
# ----------------------------------------------------------------------------
# Types: FG finished good, SF semi finished / sub assembly, RM raw material,
#        PK packaging material.

ITEMS = [
    # code, description, type, uom, lot rule, lot size, unit cost INR, supplier, shelf life wks
    ("FG-MAG-70",     "Maggi 2-Minute Masala Noodles 70g (case of 96)", "FG", "CASE", "FOQ", 1200, 1180.0, "", 36),
    ("FG-MAG-280",    "Maggi Masala Noodles Family Pack 280g (case of 24)", "FG", "CASE", "FOQ", 600, 1140.0, "", 36),
    ("FG-OAT-72",     "Maggi Oats Masala Noodles 72g (case of 96)", "FG", "CASE", "LFL", 0, 1560.0, "", 30),
    ("FG-KET-500",    "Maggi Rich Tomato Ketchup 500g (case of 24)", "FG", "CASE", "EOQ", 0, 1490.0, "", 52),

    ("SF-NCAKE-62",   "Fried noodle cake 62g", "SF", "EA", "LFL", 0, 4.90, "", 12),
    ("SF-OCAKE-64",   "Oats noodle cake 64g", "SF", "EA", "LFL", 0, 7.10, "", 12),
    ("SF-TM-8",       "Tastemaker sachet 8g filled", "SF", "EA", "FOQ", 250000, 1.85, "", 24),
    ("SF-SPICE-BLEND", "Masala spice blend", "SF", "KG", "LFL", 0, 268.0, "", 26),
    ("SF-TOM-PASTE",  "Tomato paste 28 brix", "SF", "KG", "LFL", 0, 96.0, "", 20),

    ("RM-MAIDA",      "Refined wheat flour (maida)", "RM", "KG", "EOQ", 0, 34.5, "SUP-AGRO-01", 20),
    ("RM-OATS",       "Rolled oats", "RM", "KG", "EOQ", 0, 78.0, "SUP-AGRO-02", 24),
    ("RM-PALMOIL",    "Refined palm oil", "RM", "KG", "FOQ", 20000, 112.0, "SUP-OIL-01", 26),
    ("RM-SALT",       "Iodised salt", "RM", "KG", "FOQ", 15000, 18.5, "SUP-AGRO-01", 52),
    ("RM-STARCH",     "Maize starch", "RM", "KG", "LFL", 0, 46.0, "SUP-AGRO-02", 40),
    ("RM-CHILLI",     "Red chilli powder", "RM", "KG", "EOQ", 0, 285.0, "SUP-SPICE-01", 18),
    ("RM-CORIANDER",  "Coriander powder", "RM", "KG", "EOQ", 0, 196.0, "SUP-SPICE-01", 18),
    ("RM-TURMERIC",   "Turmeric powder", "RM", "KG", "LFL", 0, 245.0, "SUP-SPICE-02", 18),
    ("RM-GARLIC-PWD", "Dehydrated garlic powder", "RM", "KG", "LFL", 0, 412.0, "SUP-SPICE-02", 16),
    ("RM-TOMATO",     "Fresh tomato grade A", "RM", "KG", "LFL", 0, 16.5, "SUP-AGRO-03", 2),
    ("RM-SUGAR",      "Refined sugar", "RM", "KG", "FOQ", 10000, 44.0, "SUP-AGRO-03", 40),

    ("PK-WRAP-70",    "Laminate wrapper 70g pack", "PK", "EA", "FOQ", 300000, 0.62, "SUP-PACK-01", 78),
    ("PK-WRAP-280",   "Laminate wrapper 280g family pack", "PK", "EA", "FOQ", 100000, 1.35, "SUP-PACK-01", 78),
    ("PK-WRAP-OAT",   "Laminate wrapper oats 72g pack", "PK", "EA", "FOQ", 120000, 0.74, "SUP-PACK-02", 78),
    ("PK-SACHET",     "Tastemaker sachet film", "PK", "EA", "FOQ", 400000, 0.21, "SUP-PACK-02", 78),
    ("PK-BOTTLE-500", "PET bottle 500g with cap", "PK", "EA", "FOQ", 60000, 6.40, "SUP-PACK-03", 104),
    ("PK-LABEL-KET",  "Ketchup label sleeve", "PK", "EA", "LFL", 0, 0.95, "SUP-PACK-03", 104),
    ("PK-CTN-96",     "Shipper carton 96 pack", "PK", "EA", "FOQ", 8000, 22.5, "SUP-PACK-01", 104),
    ("PK-CTN-24",     "Shipper carton 24 pack", "PK", "EA", "FOQ", 6000, 17.8, "SUP-PACK-01", 104),
]

FG_ITEMS = [i[0] for i in ITEMS if i[2] == "FG"]

# ----------------------------------------------------------------------------
# 2. Bill of material  (4 levels: FG -> SF -> SF -> RM)
# ----------------------------------------------------------------------------
BOM = [
    # parent, component, qty per parent, uom, scrap pct
    ("FG-MAG-70",  "SF-NCAKE-62",  96,   "EA", 0.5),
    ("FG-MAG-70",  "SF-TM-8",      96,   "EA", 0.5),
    ("FG-MAG-70",  "PK-WRAP-70",   96,   "EA", 1.5),
    ("FG-MAG-70",  "PK-CTN-96",    1,    "EA", 0.5),

    ("FG-MAG-280", "SF-NCAKE-62",  96,   "EA", 0.5),
    ("FG-MAG-280", "SF-TM-8",      96,   "EA", 0.5),
    ("FG-MAG-280", "PK-WRAP-280",  24,   "EA", 1.5),
    ("FG-MAG-280", "PK-CTN-24",    1,    "EA", 0.5),

    ("FG-OAT-72",  "SF-OCAKE-64",  96,   "EA", 0.5),
    ("FG-OAT-72",  "SF-TM-8",      96,   "EA", 0.5),
    ("FG-OAT-72",  "PK-WRAP-OAT",  96,   "EA", 1.5),
    ("FG-OAT-72",  "PK-CTN-96",    1,    "EA", 0.5),

    ("FG-KET-500", "SF-TOM-PASTE", 4.32, "KG", 1.0),
    ("FG-KET-500", "RM-SUGAR",     1.44, "KG", 0.5),
    ("FG-KET-500", "PK-BOTTLE-500", 24,  "EA", 1.0),
    ("FG-KET-500", "PK-LABEL-KET", 24,   "EA", 2.0),
    ("FG-KET-500", "PK-CTN-24",    1,    "EA", 0.5),

    ("SF-NCAKE-62", "RM-MAIDA",    0.050, "KG", 2.0),
    ("SF-NCAKE-62", "RM-PALMOIL",  0.012, "KG", 1.5),
    ("SF-NCAKE-62", "RM-SALT",     0.001, "KG", 1.0),

    ("SF-OCAKE-64", "RM-OATS",     0.030, "KG", 2.0),
    ("SF-OCAKE-64", "RM-MAIDA",    0.025, "KG", 2.0),
    ("SF-OCAKE-64", "RM-PALMOIL",  0.010, "KG", 1.5),
    ("SF-OCAKE-64", "RM-SALT",     0.001, "KG", 1.0),

    ("SF-TM-8",    "SF-SPICE-BLEND", 0.005, "KG", 1.0),
    ("SF-TM-8",    "RM-SALT",        0.002, "KG", 1.0),
    ("SF-TM-8",    "RM-STARCH",      0.001, "KG", 1.0),
    ("SF-TM-8",    "PK-SACHET",      1,     "EA", 2.0),

    ("SF-SPICE-BLEND", "RM-CHILLI",     0.30, "KG", 1.0),
    ("SF-SPICE-BLEND", "RM-CORIANDER",  0.45, "KG", 1.0),
    ("SF-SPICE-BLEND", "RM-TURMERIC",   0.15, "KG", 1.0),
    ("SF-SPICE-BLEND", "RM-GARLIC-PWD", 0.10, "KG", 1.0),

    ("SF-TOM-PASTE", "RM-TOMATO", 5.50, "KG", 3.0),
    ("SF-TOM-PASTE", "RM-SALT",   0.02, "KG", 1.0),
]

# ----------------------------------------------------------------------------
# 3. Routing and work centres
# ----------------------------------------------------------------------------
ROUTING = [
    # item, op seq, operation, work centre, setup min, run min per case
    ("FG-MAG-70", 10, "Dough mixing",            "WC-MIX",   30, 0.74),
    ("FG-MAG-70", 20, "Sheeting, slitting, steaming", "WC-SHEET", 40, 0.78),
    ("FG-MAG-70", 30, "Frying and cooling",      "WC-FRY",   25, 1.20),
    ("FG-MAG-70", 40, "Tastemaker filling",      "WC-TM",    20, 0.50),
    ("FG-MAG-70", 50, "Wrapping and packing",    "WC-PACK",  35, 1.32),
    ("FG-MAG-70", 60, "Cartoning and palletising", "WC-CTN", 15, 0.40),

    ("FG-MAG-280", 10, "Dough mixing",           "WC-MIX",   30, 0.74),
    ("FG-MAG-280", 20, "Sheeting, slitting, steaming", "WC-SHEET", 45, 0.78),
    ("FG-MAG-280", 30, "Frying and cooling",     "WC-FRY",   25, 1.20),
    ("FG-MAG-280", 40, "Tastemaker filling",     "WC-TM",    20, 0.50),
    ("FG-MAG-280", 50, "Wrapping and packing",   "WC-PACK",  45, 1.46),
    ("FG-MAG-280", 60, "Cartoning and palletising", "WC-CTN", 15, 0.40),

    ("FG-OAT-72", 10, "Dough mixing",            "WC-MIX",   45, 0.86),
    ("FG-OAT-72", 20, "Sheeting, slitting, steaming", "WC-SHEET", 50, 0.90),
    ("FG-OAT-72", 30, "Frying and cooling",      "WC-FRY",   30, 1.25),
    ("FG-OAT-72", 40, "Tastemaker filling",      "WC-TM",    25, 0.50),
    ("FG-OAT-72", 50, "Wrapping and packing",    "WC-PACK",  40, 1.37),
    ("FG-OAT-72", 60, "Cartoning and palletising", "WC-CTN", 15, 0.40),

    ("FG-KET-500", 10, "Tomato washing and pulping", "WC-PULP", 45, 1.80),
    ("FG-KET-500", 20, "Cooking and concentration",  "WC-COOK", 60, 2.20),
    ("FG-KET-500", 30, "Bottling and labelling",     "WC-FILL", 30, 2.00),
    ("FG-KET-500", 40, "Cartoning and palletising",  "WC-CTN",  15, 0.50),
]

WORK_CENTRES = [
    # wc, description, machines, hrs per shift, shifts per day, days per week, efficiency pct
    ("WC-MIX",   "Dough mixing line",              1, 8, 3, 6, 90),
    ("WC-SHEET", "Sheeting, slitting and steaming", 1, 8, 3, 6, 88),
    ("WC-FRY",   "Frying and cooling tunnel",       2, 8, 3, 6, 90),
    ("WC-TM",    "Tastemaker blending and filling", 1, 8, 3, 6, 92),
    ("WC-PACK",  "Wrapping and packing line",       2, 8, 3, 6, 85),
    ("WC-CTN",   "Cartoning and palletising",       1, 8, 3, 6, 93),
    ("WC-PULP",  "Tomato washing and pulping",      1, 8, 2, 6, 86),
    ("WC-COOK",  "Cooking and concentration",       1, 8, 2, 6, 88),
    ("WC-FILL",  "Bottling and labelling line",     1, 8, 2, 6, 87),
]

# ----------------------------------------------------------------------------
# 4. Suppliers
# ----------------------------------------------------------------------------
SUPPLIERS = [
    # supplier, name, item, lead time weeks, reliability pct, MOQ, price
    ("SUP-AGRO-01", "Shakti Agro Mills, Indore",     "RM-MAIDA",      2, 96, 20000, 34.5),
    ("SUP-AGRO-01", "Shakti Agro Mills, Indore",     "RM-SALT",       2, 97, 15000, 18.5),
    ("SUP-AGRO-02", "Sahyadri Grain Traders, Nashik", "RM-OATS",      3, 92,  8000, 78.0),
    ("SUP-AGRO-02", "Sahyadri Grain Traders, Nashik", "RM-STARCH",    2, 94, 10000, 46.0),
    ("SUP-OIL-01",  "Konkan Edible Oils, Raigad",    "RM-PALMOIL",    3, 89, 20000, 112.0),
    ("SUP-SPICE-01", "Guntur Spice House",           "RM-CHILLI",     4, 85,  4000, 285.0),
    ("SUP-SPICE-01", "Guntur Spice House",           "RM-CORIANDER",  4, 88,  4000, 196.0),
    ("SUP-SPICE-02", "Erode Turmeric Co-op",         "RM-TURMERIC",   3, 91,  3000, 245.0),
    ("SUP-SPICE-02", "Erode Turmeric Co-op",         "RM-GARLIC-PWD", 4, 83,  2000, 412.0),
    ("SUP-AGRO-03", "Nashik Farm Collective",        "RM-TOMATO",     1, 78, 25000, 16.5),
    ("SUP-AGRO-03", "Nashik Farm Collective",        "RM-SUGAR",      2, 95, 10000, 44.0),
    ("SUP-PACK-01", "Uflex Laminates, Pune",         "PK-WRAP-70",    2, 95, 300000, 0.62),
    ("SUP-PACK-01", "Uflex Laminates, Pune",         "PK-WRAP-280",   2, 95, 100000, 1.35),
    ("SUP-PACK-01", "Uflex Laminates, Pune",         "PK-CTN-96",     1, 96,   8000, 22.5),
    ("SUP-PACK-01", "Uflex Laminates, Pune",         "PK-CTN-24",     1, 96,   6000, 17.8),
    ("SUP-PACK-02", "Paras Flexipack, Vapi",         "PK-WRAP-OAT",   2, 90, 120000, 0.74),
    ("SUP-PACK-02", "Paras Flexipack, Vapi",         "PK-SACHET",     2, 92, 400000, 0.21),
    ("SUP-PACK-03", "Manjushree PET, Bengaluru",     "PK-BOTTLE-500", 3, 94,  60000, 6.40),
    ("SUP-PACK-03", "Manjushree PET, Bengaluru",     "PK-LABEL-KET",  2, 93,  40000, 0.95),
]

# In house lead times in periods for made items
INHOUSE_LT = {
    "FG-MAG-70": 1, "FG-MAG-280": 1, "FG-OAT-72": 1, "FG-KET-500": 1,
    "SF-NCAKE-62": 1, "SF-OCAKE-64": 1, "SF-TM-8": 1,
    "SF-SPICE-BLEND": 1, "SF-TOM-PASTE": 1,
}

CUSTOMERS = [
    ("CUST-01", "Reliance Retail, Western Zone", 1),
    ("CUST-02", "Avenue Supermarts (DMart), Maharashtra", 1),
    ("CUST-03", "Metro Cash and Carry, Bengaluru", 2),
    ("CUST-04", "Spencers Retail, East Zone", 2),
    ("CUST-05", "Shree Ganesh Distributors, Mumbai", 3),
]


# ----------------------------------------------------------------------------
# Demand generation: four different patterns
# ----------------------------------------------------------------------------
def build_demand(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    t = np.arange(1, HIST_PERIODS + PLAN_PERIODS + 1)

    # 1. Trend plus noise
    d1 = 4200 + 18.0 * t + rng.normal(0, 250, t.size)

    # 2. Seasonal (13 week cycle) plus mild trend
    d2 = 1500 + 6.0 * t + 380 * np.sin(2 * np.pi * (t - 3) / 13.0) + rng.normal(0, 110, t.size)

    # 3. Stable level with slight decline
    d3 = 640 - 1.5 * t + rng.normal(0, 52, t.size)

    # 4. Erratic with promotion spikes
    d4 = 900 + rng.normal(0, 130, t.size)
    promo_weeks = rng.choice(t, size=7, replace=False)
    d4 = d4 + np.where(np.isin(t, promo_weeks), 0.45 * 900, 0.0)

    series = {
        "FG-MAG-70": d1,
        "FG-MAG-280": d2,
        "FG-OAT-72": d3,
        "FG-KET-500": d4,
    }

    for item, arr in series.items():
        for i, p in enumerate(t):
            rows.append({
                "Item": item,
                "Period": int(p),
                "Period_Start": period_start(int(p)).isoformat(),
                "Demand": int(max(0, round(arr[i]))),
            })
    full = pd.DataFrame(rows)
    # history file only carries the first HIST_PERIODS; the rest is the hidden
    # "truth" used to seed confirmed customer orders
    return full


def build_customer_orders(full_demand: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Confirmed orders thin out as we look further into the future."""
    coverage = {0: 0.97, 1: 0.86, 2: 0.68, 3: 0.47, 4: 0.30, 5: 0.18, 6: 0.10, 7: 0.06}
    rows = []
    oid = 1000
    for p in range(FIRST_PLAN_PERIOD, LAST_PLAN_PERIOD + 1):
        k = p - FIRST_PLAN_PERIOD
        cov = coverage.get(k, 0.0)
        if cov <= 0:
            continue
        for item in FG_ITEMS:
            true_d = int(full_demand.loc[
                (full_demand["Item"] == item) & (full_demand["Period"] == p), "Demand"].iloc[0])
            booked = int(round(true_d * cov * rng.uniform(0.92, 1.08)))
            if booked <= 0:
                continue
            # split the booked volume across 2 to 4 customers
            n = int(rng.integers(2, 5))
            picks = rng.choice(len(CUSTOMERS), size=n, replace=False)
            weights = rng.dirichlet(np.ones(n) * 2.2)
            for w, ci in zip(weights, picks):
                q = int(round(booked * w))
                if q < 20:
                    continue
                oid += 1
                cust_code, cust_name, prio = CUSTOMERS[ci]
                rows.append({
                    "Order_ID": f"SO-{oid}",
                    "Customer_ID": cust_code,
                    "Customer": cust_name,
                    "Item": item,
                    "Period": p,
                    "Qty": q,
                    "Due_Date": period_end(p).isoformat(),
                    "Priority": prio,
                    "Status": "Confirmed",
                })
    return pd.DataFrame(rows)


def build_inventory_master(rng: np.random.Generator, demand: pd.DataFrame,
                           bom: pd.DataFrame) -> pd.DataFrame:
    """On hand and safety stock sized off roughly one to three weeks of usage."""
    # weekly usage of every item, derived by exploding average FG demand
    avg_fg = (demand[demand["Period"] <= HIST_PERIODS]
              .groupby("Item")["Demand"].mean().to_dict())
    usage = dict(avg_fg)

    # walk the BOM top down (FG -> SF -> RM) to get dependent weekly usage
    order = ["FG-MAG-70", "FG-MAG-280", "FG-OAT-72", "FG-KET-500",
             "SF-NCAKE-62", "SF-OCAKE-64", "SF-TM-8", "SF-SPICE-BLEND", "SF-TOM-PASTE"]
    for parent in order:
        pu = usage.get(parent, 0.0)
        for _, r in bom[bom["Parent"] == parent].iterrows():
            add = pu * float(r["Qty_Per"]) * (1 + float(r["Scrap_Pct"]) / 100.0)
            usage[r["Component"]] = usage.get(r["Component"], 0.0) + add

    sup_lt = {r[2]: r[3] for r in SUPPLIERS}
    sup_of = {r[2]: r[0] for r in SUPPLIERS}

    rows = []
    for code, desc, typ, uom, rule, lot, cost, sup, shelf in ITEMS:
        wk = usage.get(code, 0.0)
        lt = INHOUSE_LT.get(code, sup_lt.get(code, 2))
        supplier = sup_of.get(code, "" if typ in ("FG", "SF") else sup)
        # safety stock: FG about 1.2 weeks, SF 0.7, RM/PK 1.5 weeks of usage
        ss_weeks = {"FG": 1.2, "SF": 0.7, "RM": 1.5, "PK": 1.5}[typ]
        ss = wk * ss_weeks
        # on hand: safety stock plus cover for the replenishment lead time, which
        # is what a plant that has been running steadily would actually hold.
        # a few items are deliberately pushed short further down.
        if typ in ("RM", "PK"):
            oh = ss + wk * lt * rng.uniform(0.7, 1.2)
        else:
            oh = ss + wk * rng.uniform(0.4, 1.6)
        rows.append({
            "Item": code,
            "Description": desc,
            "Item_Type": typ,
            "UoM": uom,
            "On_Hand": round(oh, 2) if uom == "KG" else int(round(oh)),
            "Safety_Stock": round(ss, 2) if uom == "KG" else int(round(ss)),
            "Lead_Time_Periods": int(lt),
            "Lot_Size_Rule": rule,
            "Lot_Size": lot,
            "Unit_Cost": cost,
            "Ordering_Cost": 4500 if typ in ("RM", "PK") else 6500,
            "Holding_Cost_Rate": 0.22,
            "Supplier": supplier,
            "Shelf_Life_Weeks": shelf,
            "Make_or_Buy": "Make" if typ in ("FG", "SF") else "Buy",
        })
    df = pd.DataFrame(rows)

    # force a small number of deliberate exception cases so the control tower
    # has something to detect on day one
    df.loc[df["Item"] == "RM-CHILLI", "On_Hand"] = round(
        float(df.loc[df["Item"] == "RM-CHILLI", "Safety_Stock"].iloc[0]) * 0.42, 2)
    df.loc[df["Item"] == "PK-WRAP-280", "On_Hand"] = int(
        float(df.loc[df["Item"] == "PK-WRAP-280", "Safety_Stock"].iloc[0]) * 0.55)
    df.loc[df["Item"] == "RM-SUGAR", "On_Hand"] = round(
        float(df.loc[df["Item"] == "RM-SUGAR", "Safety_Stock"].iloc[0]) * 4.6, 2)
    return df


def build_scheduled_receipts(inv: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Open purchase and works orders already in the pipeline."""
    rows = []
    n = 1
    for _, r in inv.iterrows():
        if r["Make_or_Buy"] != "Buy":
            continue
        if rng.random() < 0.55:
            p = FIRST_PLAN_PERIOD + int(rng.integers(0, 3))
            qty = float(r["Lot_Size"]) if r["Lot_Size"] else round(float(r["Safety_Stock"]) * 1.4, 2)
            if qty <= 0:
                continue
            rows.append({
                "Receipt_ID": f"PO-{2600 + n}",
                "Item": r["Item"],
                "Period": p,
                "Qty": qty,
                "Due_Date": period_end(p).isoformat(),
                "Supplier": r["Supplier"],
                "Type": "Purchase Order",
            })
            n += 1
    return pd.DataFrame(rows)


def build_production_orders(rng: np.random.Generator) -> pd.DataFrame:
    """A few works orders already released on the shop floor."""
    rows = []
    n = 1
    for item, qty in [("FG-MAG-70", 1200), ("FG-MAG-280", 600), ("FG-KET-500", 480)]:
        rows.append({
            "Order_ID": f"WO-{7100 + n}",
            "Item": item,
            "Qty": qty,
            "Release_Period": FIRST_PLAN_PERIOD,
            "Due_Period": FIRST_PLAN_PERIOD,
            "Due_Date": period_end(FIRST_PLAN_PERIOD).isoformat(),
            "Priority": 1,
            "Status": "Released",
            "Source": "Carried over from previous week",
        })
        n += 1
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    full_demand = build_demand(rng)
    hist = full_demand[full_demand["Period"] <= HIST_PERIODS].copy()

    bom = pd.DataFrame(BOM, columns=["Parent", "Component", "Qty_Per", "UoM", "Scrap_Pct"])
    orders = build_customer_orders(full_demand, rng)
    inv = build_inventory_master(rng, full_demand, bom)
    sr = build_scheduled_receipts(inv, rng)
    po = build_production_orders(rng)

    sup = pd.DataFrame(SUPPLIERS, columns=[
        "Supplier_ID", "Supplier_Name", "Item", "Lead_Time_Periods",
        "Reliability_Pct", "Min_Order_Qty", "Price"])
    sup["Preferred"] = "Yes"

    rout = pd.DataFrame(ROUTING, columns=[
        "Item", "Op_Seq", "Operation", "Work_Centre", "Setup_Time_Min", "Run_Time_Min_Per_Unit"])

    mach = pd.DataFrame(WORK_CENTRES, columns=[
        "Work_Centre", "Description", "Num_Machines", "Hours_Per_Shift",
        "Shifts_Per_Day", "Days_Per_Week", "Efficiency_Pct"])

    # MPS input template: forecast and order columns are filled by the engine
    mps_rows = []
    for item in FG_ITEMS:
        for p in range(FIRST_PLAN_PERIOD, LAST_PLAN_PERIOD + 1):
            mps_rows.append({
                "Item": item, "Period": p,
                "Period_Start": period_start(p).isoformat(),
                "Forecast": "", "Customer_Orders": "", "MPS_Qty": "", "Notes": "",
            })
    mps = pd.DataFrame(mps_rows)

    calendar = pd.DataFrame([
        {"Period": p, "Period_Start": period_start(p).isoformat(),
         "Period_End": period_end(p).isoformat(),
         "Bucket": "Week",
         "Type": "History" if p <= HIST_PERIODS else "Planning"}
        for p in range(1, HIST_PERIODS + PLAN_PERIODS + 1)
    ])

    files = {
        "Demand_History.csv": hist,
        "Customer_Orders.csv": orders,
        "Inventory_Master.csv": inv,
        "BOM.csv": bom,
        "Supplier_LeadTime.csv": sup,
        "MPS.csv": mps,
        "Routing_WorkCentre.csv": rout,
        "Machine_Capacity.csv": mach,
        "Production_Orders.csv": po,
        "Scheduled_Receipts.csv": sr,
        "Period_Calendar.csv": calendar,
    }
    for name, df in files.items():
        df.to_csv(os.path.join(OUT_DIR, name), index=False)
        print(f"{name:28s} {len(df):6d} rows")

    print(f"\nWritten to {OUT_DIR}")


if __name__ == "__main__":
    main()
