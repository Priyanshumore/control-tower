"""
Package the Vayu Mobility dataset as one Excel workbook.

This is the file to use when testing whether the control tower really does work
on a dataset it has never seen: it is a single upload, the sheet names and column
headers are plant engineering vocabulary rather than the vocabulary of the
bundled Maggi data, and the fiscal weeks are numbered 201 to 248.

Run:  python scripts/build_vayu_workbook.py
Output: exports/Vayu_Mobility_Plant_Data.xlsx
"""

from __future__ import annotations

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "data", "vayu_mobility")
OUT = os.path.join(ROOT, "exports", "Vayu_Mobility_Plant_Data.xlsx")

# file -> sheet name shown in Excel
SHEETS = [
    ("Despatch_History.csv", "Despatch History"),
    ("Dealer_Orders.csv", "Dealer Orders"),
    ("Part_Master.csv", "Part Master"),
    ("Bill_Of_Materials.csv", "Bill Of Materials"),
    ("Vendor_Master.csv", "Vendor Master"),
    ("Process_Routing.csv", "Process Routing"),
    ("Cost_Centre_Capacity.csv", "Cost Centre Capacity"),
    ("Works_Orders.csv", "Works Orders"),
    ("Inbound_Receipts.csv", "Inbound Receipts"),
    ("Build_Plan.csv", "Build Plan"),
    ("Fiscal_Calendar.csv", "Fiscal Calendar"),
]

HEADER_BG = "#1C242F"
HEADER_FG = "#F5A524"


def main() -> None:
    if not os.path.isdir(SRC):
        raise SystemExit("Run scripts/generate_vayu_dataset.py first.")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    with pd.ExcelWriter(OUT, engine="xlsxwriter") as xl:
        book = xl.book
        head = book.add_format({"bold": True, "bg_color": HEADER_BG,
                                "font_color": HEADER_FG, "border": 1,
                                "border_color": "#26313D", "valign": "vcenter"})
        for fname, sheet in SHEETS:
            df = pd.read_csv(os.path.join(SRC, fname))
            df.to_excel(xl, sheet_name=sheet, index=False, startrow=1, header=False)
            ws = xl.sheets[sheet]
            for col, name in enumerate(df.columns):
                ws.write(0, col, name, head)
                # the build plan column is deliberately blank, so guard the max
                longest = df[name].astype(str).str.len().max()
                longest = 8 if pd.isna(longest) else int(longest)
                width = max(len(str(name)) + 3, longest + 2)
                ws.set_column(col, col, min(width, 34))
            ws.freeze_panes(1, 0)
            ws.autofilter(0, 0, len(df), len(df.columns) - 1)
            print(f"  {sheet:<24} {len(df):>4} rows x {len(df.columns)} cols")

    size = os.path.getsize(OUT) / 1024
    print(f"\nWritten {OUT} ({size:,.0f} KB)")


if __name__ == "__main__":
    main()
