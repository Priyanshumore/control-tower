"""
Canonical schema for the control tower.

Every incoming dataset is mapped onto these canonical table and column names.
That is what lets the same dashboard run on the bundled Nestle Maggi data and
on any other dataset the user attaches: only the mapping changes, never the
engine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class Field:
    name: str                     # canonical column name
    synonyms: List[str]           # accepted source header variants
    required: bool = False
    dtype: str = "str"            # str | num | int | date
    default: object = None


@dataclass
class Table:
    name: str                     # canonical table name
    label: str
    file_hints: List[str]         # filename keywords used for auto detection
    fields: List[Field]
    required: bool = False

    def field_map(self) -> Dict[str, Field]:
        return {f.name: f for f in self.fields}


def _n(s: str) -> str:
    """Normalise a header for fuzzy comparison."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


SCHEMA: Dict[str, Table] = {}


def _add(t: Table) -> None:
    SCHEMA[t.name] = t


_add(Table(
    name="demand_history",
    label="Demand history",
    file_hints=["demand", "history", "sales", "shipment"],
    required=True,
    fields=[
        Field("item", ["item", "itemcode", "sku", "product", "productcode", "material",
                       "materialcode", "materialnumber", "matnr", "partno", "partnumber",
                       "fg", "itemid", "itemnumber", "articlecode", "article",
                       "partcode", "partid", "partnbr", "stockcode", "itemno"], True),
        Field("period", ["period", "week", "wk", "month", "bucket", "timeperiod", "periodno",
                         "periodid", "fiscalwk", "fiscalweek", "yearweek", "weekno", "wknum", "t"], True, "int"),
        Field("demand", ["demand", "qty", "quantity", "sales", "actual", "actualdemand", "volume",
                         "units", "shipped", "shippedqty", "shippedquantity", "salesqty",
                         "dispatchqty", "dispatchedunits", "despatchedunits", "dispatched", "despatched",
                         "consumption", "netsales", "offtake", "billedqty"], True, "num"),
        Field("period_start", ["periodstart", "date", "startdate", "weekstart", "weekstartdate",
                               "periodstartdate", "wkstart", "weekcommencing", "commencing",
                               "startdt", "dt", "calendardate"], False, "date"),
    ],
))

_add(Table(
    name="customer_orders",
    label="Customer orders",
    file_hints=["customer", "order", "sales_order", "booked", "so"],
    fields=[
        Field("item", ["item", "itemcode", "sku", "product", "material", "productcode"], True),
        Field("period", ["period", "week", "month", "bucket", "duperiod", "dueperiod"], True, "int"),
        Field("qty", ["qty", "quantity", "orderqty", "demand", "volume", "units", "orderquantity"], True, "num"),
        Field("order_id", ["orderid", "order", "orderno", "ordernumber", "sonumber",
                           "salesordernumber", "salesorder", "docno"]),
        Field("customer", ["customer", "customername", "account", "soldto", "client",
                           "dealer", "dealername", "distributor", "buyer"]),
        Field("due_date", ["duedate", "date", "requesteddate", "deliverydate"], False, "date"),
        Field("priority", ["priority", "prio", "rank", "class"], False, "num", 2),
    ],
))

_add(Table(
    name="inventory_master",
    label="Inventory master",
    file_hints=["inventory", "item_master", "itemmaster", "material_master", "stock"],
    required=True,
    fields=[
        Field("item", ["item", "itemcode", "sku", "product", "material", "partno", "productcode"], True),
        Field("description", ["description", "itemdescription", "name", "itemname", "desc"]),
        Field("item_type", ["itemtype", "type", "category", "level", "materialtype",
                            "mrptype", "class"]),
        Field("uom", ["uom", "unit", "unitofmeasure", "units", "measure"]),
        Field("on_hand", ["onhand", "onhandqty", "stock", "openingstock", "currentstock",
                          "inventory", "qtyonhand", "availableqty", "beginninginventory",
                          "unrestrictedstock", "unrestricted", "stockonhand", "availablestock"], False, "num", 0),
        Field("safety_stock", ["safetystock", "ss", "minstock", "buffer", "safetystockqty",
                               "safetystk", "minimumstock"], False, "num", 0),
        Field("lead_time", ["leadtime", "leadtimeperiods", "leadtimeweeks", "lt",
                            "leadtimedays", "plannedleadtime", "replenishmentleadtime",
                            "planneddeliverytime", "deliverytime", "procurementtime"], False, "num", 1),
        Field("lot_rule", ["lotsizerule", "lotrule", "lotsizingrule", "lotsizekey", "orderpolicy",
                           "policy", "lotsizingpolicy", "sizingrule"], False, "str", "LFL"),
        Field("lot_size", ["lotsize", "fixedorderqty", "foq", "fixedlot", "orderqty", "batchsize",
                           "batchqty", "minlotsize", "standardlot"], False, "num", 0),
        Field("unit_cost", ["unitcost", "cost", "price", "standardcost", "standardprice",
                            "stdprice", "unitprice", "movingprice", "value", "rate"], False, "num", 0),
        Field("ordering_cost", ["orderingcost", "setupcost", "ordercost", "s"], False, "num", 0),
        Field("holding_rate", ["holdingcostrate", "holdingrate", "holdingcostpct", "carryingrate",
                               "carryingcostrate", "carrypct", "carryrate", "carrying",
                               "inventorycarryingrate", "h", "icc"], False, "num", 0.2),
        Field("supplier", ["supplier", "vendor", "supplierid", "vendorcode", "source"]),
        Field("make_buy", ["makeorbuy", "makebuy", "sourcetype", "procurementtype",
                           "proctype", "sourcing", "mb"]),
        Field("shelf_life", ["shelflifeweeks", "shelflife", "expiryweeks"], False, "num", 0),
    ],
))

_add(Table(
    name="bom",
    label="Bill of material",
    file_hints=["bom", "billofmaterial", "recipe", "formulation", "structure"],
    fields=[
        Field("parent", ["parent", "parentitem", "parentcode", "parentmaterial", "headermaterial",
                         "assembly", "topitem", "fg", "item", "product"], True),
        Field("component", ["component", "componentitem", "componentmaterial", "child", "childitem",
                            "material", "ingredient", "componentcode", "subpart", "subassembly",
                            "childpart", "partcode"], True),
        Field("qty_per", ["qtyper", "quantityper", "componentquantity", "qtyperparent", "qty",
                          "quantity", "unitsper", "usage", "usageper", "perunit", "perassembly",
                          "consumption"], True, "num", 1),
        Field("scrap_pct", ["scrappct", "scrap", "scraprate", "wastage", "yieldloss", "scrappercent"], False, "num", 0),
        Field("uom", ["uom", "unit", "unitofmeasure"]),
    ],
))

_add(Table(
    name="supplier_leadtime",
    label="Supplier lead time",
    file_hints=["supplier", "vendor", "leadtime", "sourcing"],
    fields=[
        Field("supplier", ["supplierid", "supplier", "vendor", "vendorcode", "suppliername"], True),
        Field("item", ["item", "itemcode", "material", "sku", "product"], True),
        Field("lead_time", ["leadtimeperiods", "leadtime", "leadtimeweeks", "lt", "leadtimedays"], True, "num", 1),
        Field("reliability", ["reliabilitypct", "reliability", "otd", "otif", "servicelevel",
                              "ontimepct", "performance"], False, "num", 100),
        Field("moq", ["minorderqty", "moq", "minimumorderquantity", "minqty"], False, "num", 0),
        Field("price", ["price", "unitprice", "cost", "rate"], False, "num", 0),
        Field("supplier_name", ["suppliername", "vendorname", "name"]),
    ],
))

_add(Table(
    name="routing",
    label="Routing and work centre",
    file_hints=["routing", "workcentre", "workcenter", "process", "operation"],
    fields=[
        Field("item", ["item", "itemcode", "product", "sku", "job", "material"], True),
        Field("op_seq", ["opseq", "operationseq", "sequence", "seq", "step", "activity",
                         "activitynumber", "operationnumber", "opno", "stepno"], True, "int"),
        Field("operation", ["operation", "operationname", "operationdescription", "task",
                            "process", "description"]),
        Field("work_centre", ["workcentre", "workcenter", "workctr", "wrkctr", "arbpl", "wc",
                              "costcentre", "costcenter", "machine", "resource", "line",
                              "station", "workstation"], True),
        Field("setup_min", ["setuptimemin", "setuptime", "setup", "changeover",
                            "setupminutes", "setuptimeminutes"], False, "num", 0),
        Field("run_min", ["runtimeminperunit", "runtimeperunit", "runtime", "machinetime",
                          "processingtime", "proctime", "cycletime", "minperunit",
                          "processingtimemin", "runtimeminutes", "unitruntime"], True, "num", 0),
    ],
))

_add(Table(
    name="machine_capacity",
    label="Machine capacity",
    file_hints=["machine", "capacity", "resource", "shift"],
    fields=[
        Field("work_centre", ["workcentre", "workcenter", "workctr", "wrkctr", "arbpl", "wc",
                              "costcentre", "costcenter", "machine", "resource", "line",
                              "station"], True),
        Field("description", ["description", "name", "desc"]),
        Field("num_machines", ["nummachines", "machines", "noofmachines", "numberofmachines",
                               "numberofcapacities", "capacities", "capacityunits",
                               "numberofstations", "noofstations", "stations",
                               "parallelmachines", "count"], False, "num", 1),
        Field("hours_per_shift", ["hourspershift", "shifthours", "hoursshift", "hours"], False, "num", 8),
        Field("shifts_per_day", ["shiftsperday", "shifts", "noofshifts", "numbershifts"], False, "num", 1),
        Field("days_per_week", ["daysperweek", "workingdays", "workdays", "days"], False, "num", 6),
        Field("efficiency", ["efficiencypct", "efficiency", "oee", "capacityutilisation",
                             "utilisation", "performance"], False, "num", 100),
    ],
))

_add(Table(
    name="production_orders",
    label="Production orders",
    file_hints=["production", "workorder", "worksorder", "job", "shopfloor"],
    fields=[
        Field("order_id", ["orderid", "order", "workorder", "wo", "jobid", "job", "orderno"], True),
        Field("item", ["item", "itemcode", "product", "sku", "material"], True),
        Field("qty", ["qty", "quantity", "orderqty", "batchsize", "volume"], True, "num"),
        Field("release_period", ["releaseperiod", "startperiod", "arrival", "arrivalperiod",
                                 "releaseweek", "start"], False, "int"),
        Field("due_period", ["dueperiod", "due", "dueweek", "requiredperiod"], False, "int"),
        Field("priority", ["priority", "prio", "rank"], False, "num", 2),
        Field("status", ["status", "state", "orderstatus"]),
    ],
))

_add(Table(
    name="scheduled_receipts",
    label="Scheduled receipts",
    file_hints=["scheduledreceipt", "openorder", "purchaseorder", "receipt", "inbound"],
    fields=[
        Field("item", ["item", "itemcode", "material", "sku", "product"], True),
        Field("period", ["period", "week", "dueperiod", "receiptperiod", "month"], True, "int"),
        Field("qty", ["qty", "quantity", "receiptqty", "openqty", "volume"], True, "num"),
        Field("receipt_id", ["receiptid", "poid", "ponumber", "orderid", "docno"]),
        Field("supplier", ["supplier", "vendor", "supplierid"]),
    ],
))

_add(Table(
    name="mps_input",
    label="MPS template",
    file_hints=["mps", "masterproduction", "masterschedule"],
    fields=[
        Field("item", ["item", "itemcode", "product", "sku"], True),
        Field("period", ["period", "week", "bucket", "month"], True, "int"),
        Field("mps_qty", ["mpsqty", "mps", "plannedqty", "scheduleqty", "masterschedule"], False, "num", 0),
        Field("forecast", ["forecast", "fcst", "demandforecast"], False, "num", 0),
    ],
))

_add(Table(
    name="period_calendar",
    label="Period calendar",
    file_hints=["calendar", "period"],
    fields=[
        Field("period", ["period", "week", "bucket", "month"], True, "int"),
        Field("period_start", ["periodstart", "startdate", "date", "start"], False, "date"),
        Field("period_end", ["periodend", "enddate", "end"], False, "date"),
        Field("type", ["type", "phase", "category"]),
    ],
))


TABLE_ORDER = [
    "demand_history", "customer_orders", "inventory_master", "bom",
    "supplier_leadtime", "mps_input", "routing", "machine_capacity",
    "production_orders", "scheduled_receipts", "period_calendar",
]


# ---------------------------------------------------------------------------
# Auto detection helpers
# ---------------------------------------------------------------------------
# Vocabulary that means the same thing whichever table it turns up in. Applying
# it once here beats repeating it inside every table definition, and it is where
# a new dialect should be taught to the matcher.
COMMON_SYNONYMS = {
    "item": ["partcode", "partid", "partno", "partnumber", "stockcode", "itemno",
             "materialcode", "materialnumber", "matnr", "sku"],
    "period": ["fiscalweek", "fiscalwk", "weekno", "weeknumber", "fiscalperiod",
               "bucket", "wk", "planweek"],
    "qty": ["units", "inboundunits", "orderunits", "buildqty", "receivedqty",
            "despatchedunits", "dispatchedunits"],
    "priority": ["urgency", "prio", "importance"],
    "status": ["orderstatus", "state"],
    "due_date": ["promisedate", "promiseddate", "commitdate"],
    "period_start": ["weekcommencing", "commencing", "weekstartdate"],
    "period_end": ["weekending", "weekend", "enddate"],
    "receipt_id": ["grnref", "grn", "goodsreceipt", "poref", "ponumber"],
    "release_period": ["startweek", "startperiod", "releaseweek", "openweek"],
    "due_period": ["finishweek", "completionweek", "dueweek", "closeweek"],
    "order_id": ["worksorder", "workorder", "wonumber", "ordernumber"],
    "mps_qty": ["buildplan", "buildqty", "planqty", "masterschedule"],
    "type": ["bucket", "phase", "flag"],
    "description": ["partname", "itemname", "costcentrename", "name"],
    "supplier": ["vendorcode", "vendorname", "vendor"],
    "scrap_pct": ["rejectionpct", "rejection", "rejectpct", "wastagepct", "yieldloss"],
    "uom": ["unit", "baseunit", "stockunit"],
}

for _tbl in SCHEMA.values():
    for _f in _tbl.fields:
        _extra = COMMON_SYNONYMS.get(_f.name)
        if _extra:
            _f.synonyms = list(dict.fromkeys(list(_f.synonyms) + _extra))


# Short forms that ERP extracts use constantly. Expanding them before matching
# is what lets WORK_CTR find work_centre and COMP_MATL find component.
ABBREV = {
    "matnr": "material", "matl": "material", "mat": "material", "artnr": "article",
    "nbr": "number", "no": "number", "num": "number", "nr": "number", "id": "id",
    "qty": "quantity", "qnty": "quantity", "amt": "amount", "val": "value",
    "ctr": "centre", "cntr": "centre", "ctre": "centre", "center": "centre",
    "wc": "workcentre", "arbpl": "workcentre", "wrk": "work", "wrks": "works",
    "wk": "week", "wks": "weeks", "dt": "date", "hdr": "header",
    "comp": "component", "cmp": "component", "compt": "component",
    "oper": "operation", "op": "operation", "ops": "operations", "act": "activity",
    "mach": "machine", "machs": "machines", "pct": "percent", "perc": "percent",
    "prod": "production", "deliv": "delivery", "del": "delivery", "req": "required",
    "opn": "operation", "opns": "operations", "ctre": "centre", "cc": "costcentre",
    "wo": "workorder", "grn": "goodsreceipt", "qc": "quality", "asm": "assembly",
    "sa": "subassembly", "pct": "percent", "nos": "number", "mtr": "metre",
    "stk": "stock", "std": "standard", "ord": "order", "cust": "customer",
    "vend": "supplier", "vendor": "supplier", "sup": "supplier", "supp": "supplier",
    "lt": "leadtime", "uom": "unit", "desc": "description", "seq": "sequence",
    "so": "salesorder", "po": "purchaseorder", "wo": "workorder",
    "eff": "efficiency", "util": "utilisation", "capa": "capacity", "cap": "capacity",
    "shft": "shift", "hrs": "hours", "hr": "hour", "mins": "minutes", "min": "minutes",
    "tm": "time", "sched": "scheduled", "rcpt": "receipt", "inv": "inventory",
    "unrestricted": "onhand", "prio": "priority", "fcst": "forecast", "ss": "safetystock",
}


def _split(col: str) -> List[str]:
    """Break a header into word parts, splitting camel case as well as separators."""
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(col))
    return [p for p in re.split(r"[^A-Za-z0-9]+", s) if p]


def _expand(parts: List[str]) -> List[str]:
    return [ABBREV.get(p.lower(), p.lower()) for p in parts]


@dataclass
class _Variants:
    norm: str            # workctr
    toks: set            # {work, ctr, workctr}
    enorm: str           # workcentre
    etoks: set           # {work, centre, workcentre}


def _variants(col: str) -> _Variants:
    parts = _split(col)
    exp = _expand(parts)
    toks = {p.lower() for p in parts}
    toks |= {(a + b).lower() for a, b in zip(parts, parts[1:])}
    etoks = set(exp) | {a + b for a, b in zip(exp, exp[1:])}
    return _Variants(_n(col), toks, "".join(exp), etoks)


def _score_field(f: Field, v: _Variants) -> float:
    """How well one source column answers to one canonical field, 0 to 100ish."""
    best = 0.0
    for cand in [f.name] + f.synonyms:
        k = _n(cand)
        ke = "".join(_expand(_split(cand)))
        if not k:
            continue
        if k == v.norm:
            s = 100.0
        elif ke and ke == v.enorm:
            s = 92.0
        elif k in v.toks:
            s = 74.0
        elif ke and ke in v.etoks:
            s = 68.0
        elif len(k) >= 4 and (k in v.norm or v.norm in k):
            s = 40.0
        elif len(ke) >= 4 and (ke in v.enorm or v.enorm in ke):
            s = 34.0
        else:
            continue
        # a longer synonym that matches is more convincing than a short one
        s += min(len(k), 15) * 0.3
        best = max(best, s)
    return best


MATCH_FLOOR = 33.0


def rank_tables(filename: str, columns: List[str]) -> List[tuple]:
    """Score every canonical table against one file, best first."""
    fn = _n(filename)
    # short hints have to land on a whole word: otherwise "so" matches the middle
    # of "Works_Orders" and a shop order gets read as a sales order
    fn_parts = _split(filename)
    fn_tokens = {p.lower() for p in fn_parts} | set(_expand(fn_parts))
    vars_ = [_variants(c) for c in columns]
    out = []
    for tname, tbl in SCHEMA.items():
        score = 0.0
        for hint in tbl.file_hints:
            h = _n(hint)
            hit = (h in fn_tokens) if len(h) <= 3 else (h and h in fn)
            if hit:
                # longer, more specific hints outrank generic ones such as "order"
                score += 3.0 + 0.2 * len(h)
        bests = []
        for f in tbl.fields:
            bests.append(max((_score_field(f, v) for v in vars_), default=0.0))
        # how much of this table's shape the file actually carries
        shape = sum(bests) / (100.0 * len(tbl.fields)) if tbl.fields else 0.0
        score += 6.0 * shape
        req = [f for f in tbl.fields if f.required]
        if req:
            miss = sum(1 for f, b in zip(tbl.fields, bests) if f.required and b < 55.0)
            score -= 2.5 * miss
        out.append((tname, score))
    return sorted(out, key=lambda x: -x[1])


def guess_table(filename: str, columns: List[str]) -> Optional[str]:
    """Guess which canonical table a file belongs to, by name then by columns."""
    ranked = rank_tables(filename, columns)
    return ranked[0][0] if ranked and ranked[0][1] >= 3.0 else None


def guess_mapping(table: str, columns: List[str]) -> Dict[str, Optional[str]]:
    """Map canonical field -> source column for one table.

    Scores every field against every column and assigns the strongest pairs
    first, so a header like ORDER_COST cannot be claimed by unit_cost while
    ordering_cost, which matches it exactly, goes unmapped.
    """
    tbl = SCHEMA[table]
    vars_ = {c: _variants(c) for c in columns}
    order = {f.name: i for i, f in enumerate(tbl.fields)}

    pairs = []
    for f in tbl.fields:
        for c in columns:
            sc = _score_field(f, vars_[c])
            if sc >= MATCH_FLOOR:
                pairs.append((sc, -1.0 if f.required else 0.0, -order[f.name], f.name, c))
    pairs.sort(key=lambda p: (-p[0], p[1], -p[2]))

    mapping: Dict[str, Optional[str]] = {f.name: None for f in tbl.fields}
    taken: set = set()
    for sc, _req, _ord, fname, col in pairs:
        if mapping[fname] is None and col not in taken:
            mapping[fname] = col
            taken.add(col)
    return mapping


def apply_mapping(df: pd.DataFrame, table: str, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    """Rename to canonical columns, coerce types, apply defaults."""
    tbl = SCHEMA[table]
    out = pd.DataFrame(index=df.index)

    for f in tbl.fields:
        src = mapping.get(f.name)
        if src is not None and src in df.columns:
            col = df[src]
        else:
            col = pd.Series([f.default] * len(df), index=df.index)

        if f.dtype in ("num", "int"):
            col = pd.to_numeric(col, errors="coerce")
            if f.default is not None:
                col = col.fillna(f.default)
            if f.dtype == "int":
                col = col.fillna(0).astype("int64")
        elif f.dtype == "date":
            col = pd.to_datetime(col, errors="coerce")
        else:
            fill = f.default if f.default is not None else ""
            col = col.astype("object").where(col.notna(), fill)
            col = col.map(lambda v: str(v).strip())
            col = col.replace({"nan": "", "None": "", "<NA>": "", "NaN": ""})
            if f.default is not None:
                col = col.where(col != "", f.default)

        out[f.name] = col

    return out


def validate(table: str, df: pd.DataFrame) -> List[str]:
    """Return a list of human readable problems for a mapped table."""
    issues: List[str] = []
    tbl = SCHEMA[table]
    for f in tbl.fields:
        if not f.required:
            continue
        if f.name not in df.columns:
            issues.append(f"{tbl.label}: required field '{f.name}' is not mapped")
            continue
        col = df[f.name]
        if f.dtype in ("num", "int"):
            bad = col.isna().sum()
        else:
            bad = (col.astype(str).str.strip() == "").sum()
        if bad:
            issues.append(f"{tbl.label}: '{f.name}' is empty on {bad} row(s)")
    if len(df) == 0:
        issues.append(f"{tbl.label}: no rows")
    return issues
