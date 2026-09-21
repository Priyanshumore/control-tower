"""
Load a dataset from a folder, a zip archive or a set of uploaded files, detect
which canonical table each file is, map the columns and normalise periods.
"""

from __future__ import annotations

import io
import os
import re
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from . import schema as sch


@dataclass
class Dataset:
    raw: Dict[str, pd.DataFrame] = field(default_factory=dict)        # table -> raw frame
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)     # table -> mapped frame
    mappings: Dict[str, Dict[str, Optional[str]]] = field(default_factory=dict)
    sources: Dict[str, str] = field(default_factory=dict)             # table -> filename
    issues: List[str] = field(default_factory=list)
    unmatched: List[str] = field(default_factory=list)
    name: str = "dataset"

    # period bookkeeping filled by normalise_periods
    hist_periods: List[int] = field(default_factory=list)
    plan_start: int = 0
    period_dates: Dict[int, pd.Timestamp] = field(default_factory=dict)

    def has(self, table: str) -> bool:
        return table in self.tables and len(self.tables[table]) > 0

    def get(self, table: str) -> pd.DataFrame:
        if self.has(table):
            return self.tables[table].copy()
        cols = [f.name for f in sch.SCHEMA[table].fields]
        return pd.DataFrame(columns=cols)


def _read_any(name: str, data: bytes) -> Optional[pd.DataFrame]:
    ext = os.path.splitext(name)[1].lower()
    try:
        if ext in (".csv", ".txt"):
            return pd.read_csv(io.BytesIO(data), sep=None, engine="python")
        if ext == ".tsv":
            return pd.read_csv(io.BytesIO(data), sep="\t")
        if ext in (".xlsx", ".xlsm", ".xls"):
            return pd.read_excel(io.BytesIO(data))
        if ext == ".json":
            return pd.read_json(io.BytesIO(data))
    except Exception:
        return None
    return None


def _collect(files: List[Tuple[str, bytes]]) -> List[Tuple[str, pd.DataFrame]]:
    """Expand zips and read every readable tabular file."""
    out: List[Tuple[str, pd.DataFrame]] = []
    for name, data in files:
        if name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    for zi in z.namelist():
                        if zi.endswith("/") or "__MACOSX" in zi:
                            continue
                        df = _read_any(zi, z.read(zi))
                        if df is not None and len(df.columns) > 0:
                            out.append((os.path.basename(zi), df))
            except Exception:
                continue
        elif name.lower().endswith((".xlsx", ".xlsm", ".xls")):
            # a workbook may hold one table per sheet
            try:
                book = pd.read_excel(io.BytesIO(data), sheet_name=None)
                for sheet, df in book.items():
                    if len(df.columns):
                        out.append((f"{os.path.splitext(name)[0]}_{sheet}", df))
            except Exception:
                continue
        else:
            df = _read_any(name, data)
            if df is not None and len(df.columns) > 0:
                out.append((name, df))
    return out


def load_files(files: List[Tuple[str, bytes]], name: str = "uploaded",
               overrides: Optional[Dict[str, str]] = None) -> Dataset:
    """Build a Dataset from (filename, bytes) pairs.

    overrides maps filename -> canonical table name, for manual correction.
    """
    ds = Dataset(name=name)
    overrides = overrides or {}

    # Score every file against every canonical table first, then hand out the
    # tables best pair first. Deciding file by file lets an early file claim a
    # table that a later file fits far better.
    frames: Dict[str, pd.DataFrame] = {}
    candidates = []
    for fname, df in _collect(files):
        df = df.dropna(axis=1, how="all")
        df.columns = [str(c).strip() for c in df.columns]
        frames[fname] = df
        if fname in overrides:
            candidates.append((1e6, fname, overrides[fname]))
            continue
        for tname, score in sch.rank_tables(fname, list(df.columns)):
            if score >= 3.0:
                candidates.append((score, fname, tname))

    candidates.sort(key=lambda c: (-c[0], c[1]))
    placed: Dict[str, str] = {}
    for score, fname, tname in candidates:
        if fname in placed or tname in ds.raw:
            continue
        placed[fname] = tname
        ds.raw[tname] = frames[fname]
        ds.sources[tname] = fname

    for fname in frames:
        if fname not in placed:
            ds.unmatched.append(fname)

    for tname, df in ds.raw.items():
        mapping = sch.guess_mapping(tname, list(df.columns))
        ds.mappings[tname] = mapping
        ds.tables[tname] = sch.apply_mapping(df, tname, mapping)
        for f in sch.SCHEMA[tname].fields:
            if f.required and mapping.get(f.name) is None:
                ds.issues.append(
                    f"{sch.SCHEMA[tname].label}: could not detect a column for "
                    f"'{f.name}', set it manually in Data and mapping")
        ds.issues.extend(sch.validate(tname, ds.tables[tname]))

    for req in ("demand_history", "inventory_master"):
        if req not in ds.tables:
            ds.issues.append(f"Missing required table: {sch.SCHEMA[req].label}")

    normalise_periods(ds)
    return ds


def load_folder(path: str, name: Optional[str] = None) -> Dataset:
    files: List[Tuple[str, bytes]] = []
    for fn in sorted(os.listdir(path)):
        fp = os.path.join(path, fn)
        if os.path.isfile(fp) and fn.lower().endswith(
                (".csv", ".tsv", ".txt", ".xlsx", ".xlsm", ".xls", ".json", ".zip")):
            with open(fp, "rb") as fh:
                files.append((fn, fh.read()))
    return load_files(files, name or os.path.basename(path.rstrip("/")))


def remap(ds: Dataset, table: str, mapping: Dict[str, Optional[str]]) -> Dataset:
    """Re apply a user corrected column mapping for one table."""
    if table not in ds.raw:
        return ds
    ds.mappings[table] = mapping
    ds.tables[table] = sch.apply_mapping(ds.raw[table], table, mapping)
    ds.issues = [i for i in ds.issues if not i.startswith(sch.SCHEMA[table].label)]
    ds.issues.extend(sch.validate(table, ds.tables[table]))
    normalise_periods(ds)
    return ds


# ---------------------------------------------------------------------------
# Period normalisation
# ---------------------------------------------------------------------------
def _period_to_int(v) -> Optional[int]:
    if pd.isna(v):
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return int(v)
    s = str(v).strip()
    m = re.findall(r"\d+", s)
    if m:
        # 2026-W37 style: take the last number
        return int(m[-1])
    return None


def normalise_periods(ds: Dataset) -> None:
    """Make every period column a clean 1..N integer index and record the split
    between history and planning horizon."""
    for tname in ("demand_history", "customer_orders", "scheduled_receipts",
                  "mps_input", "period_calendar", "production_orders"):
        if not ds.has(tname):
            continue
        df = ds.tables[tname]
        for col in ("period", "release_period", "due_period"):
            if col in df.columns:
                vals = df[col].map(_period_to_int)
                if vals.notna().any():
                    df[col] = vals.fillna(0).astype("int64")
        ds.tables[tname] = df

    if not ds.has("demand_history"):
        return

    dh = ds.tables["demand_history"]
    # if periods are dates or unordered labels, rank them
    if dh["period"].nunique() > 0 and dh["period"].min() > 1000:
        # looks like a year-week code such as 202637: rank instead
        uniq = sorted(dh["period"].unique())
        remap_p = {v: i + 1 for i, v in enumerate(uniq)}
        dh["period"] = dh["period"].map(remap_p)
        ds.tables["demand_history"] = dh

    ds.hist_periods = sorted(dh["period"].unique().tolist())
    last_hist = max(ds.hist_periods) if ds.hist_periods else 0

    plan_candidates = []
    for tname in ("customer_orders", "mps_input", "scheduled_receipts"):
        if ds.has(tname) and "period" in ds.tables[tname].columns:
            vals = ds.tables[tname]["period"]
            vals = vals[vals > last_hist]
            if len(vals):
                plan_candidates.append(int(vals.min()))
    ds.plan_start = min(plan_candidates) if plan_candidates else last_hist + 1

    # period -> date lookup
    dates: Dict[int, pd.Timestamp] = {}
    if ds.has("period_calendar"):
        cal = ds.tables["period_calendar"]
        for _, r in cal.iterrows():
            if pd.notna(r.get("period_start")):
                dates[int(r["period"])] = pd.Timestamp(r["period_start"])
    elif "period_start" in dh.columns and dh["period_start"].notna().any():
        g = dh.dropna(subset=["period_start"]).groupby("period")["period_start"].min()
        dates = {int(k): pd.Timestamp(v) for k, v in g.items()}
    ds.period_dates = dates


def period_label(ds: Dataset, p: int) -> str:
    d = ds.period_dates.get(int(p))
    if d is not None:
        return f"P{int(p)} ({d.strftime('%d %b')})"
    return f"P{int(p)}"
