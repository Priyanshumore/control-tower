"""A calm, precise visual system for manufacturing planning."""

from __future__ import annotations

import html
from typing import Dict, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

class Theme:
    INK = "#F4F6F8"
    PANEL = "#FFFFFF"
    PANEL_2 = "#EDF3F2"
    LINE = "#DCE5E7"
    TEXT = "#172D35"
    MUTED = "#637880"
    ACCENT = "#087F79"
    AMBER = "#A16B18"
    GREEN = "#21865B"
    RED = "#C74E48"
    BLUE = "#527CB6"
    VIOLET = "#8B75AD"

    STATUS_COLOUR = {"OK": GREEN, "Tight": AMBER, "Overloaded": RED,
                     "High": RED, "Medium": AMBER, "Low": BLUE,
                     "Released": GREEN, "Held": RED,
                     "Stockout": RED, "Below safety stock": ACCENT,
                     "Negative": RED, "Shortage": RED,
                     "Past due release": RED, "Release immediately": ACCENT}

    SERIES = [ACCENT, BLUE, GREEN, VIOLET, "#E07A5F", "#5FC7C7", "#D48FD4", "#A8C256"]


    def __init__(self, mode="light"):
        self.STATUS_COLOUR = {"OK": self.GREEN, "Tight": self.AMBER, "Overloaded": self.RED,
            "High": self.RED, "Medium": self.AMBER, "Low": self.BLUE, "Released": self.GREEN,
            "Held": self.RED, "Stockout": self.RED, "Below safety stock": self.AMBER,
            "Negative": self.RED, "Shortage": self.RED, "Past due release": self.RED,
            "Release immediately": self.AMBER}
        self.SERIES = [self.ACCENT, self.BLUE, self.GREEN, self.VIOLET, "#E07A5F", "#5FC7C7", "#D48FD4", "#A8C256"]
        self.CSS = f"""
        <style>
        /* Scope discipline: this stylesheet styles the cards this app draws itself and
           nothing else. It deliberately does not set a font, a font size, a line height
           or a white-space rule on any Streamlit component. Those components are laid
           out for the metrics of the theme font, and overriding them from here is what
           makes labels sit on top of each other in a dozen unrelated places at once.
           Colours and the base font live in .streamlit/config.toml, where Streamlit
           applies them properly. */

        /* Colours only. None of these affect how anything is measured or wrapped, and
           keeping them here means the app still looks right if .streamlit/config.toml
           did not make it into the deployment. */
        .stApp {{ background: {self.INK} !important; color: {self.TEXT}; color-scheme: light; }}
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] > .main,
        [data-testid="stMainBlockContainer"] {{ background: {self.INK} !important; color: {self.TEXT}; }}
        [data-testid="stHeader"] {{ background: {self.INK} !important; }}
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
        .stApp p, .stApp label {{ color: {self.TEXT}; }}
        .stApp input, .stApp textarea {{ color: {self.TEXT} !important; background: {self.PANEL}; border-color: {self.LINE}; }}
        .stApp input::placeholder, .stApp textarea::placeholder {{ color: {self.MUTED} !important; opacity: 1; }}
        [data-testid="stSidebar"] {{ background: {self.PANEL}; border-right: 1px solid {self.LINE}; }}
        [data-testid="stDataFrame"] {{ border: 1px solid {self.LINE}; border-radius: 12px; }}
        div[data-testid="stExpander"] {{ border: 1px solid {self.LINE}; border-radius: 12px; }}

        .stTabs [data-baseweb="tab-list"] {{ border-bottom: 1px solid {self.LINE}; }}
        .stTabs [aria-selected="true"] {{ border-bottom: 2px solid {self.ACCENT}; }}

        /* ---- the plan status ribbon: the one deliberately bold element ---- */
        .ribbon {{
            display: flex; align-items: stretch; flex-wrap: wrap;
            border: 1px solid {self.LINE}; border-left: 4px solid {self.ACCENT};
            background: {self.PANEL}; border-radius: 12px;
            margin: 0.2rem 0 1.1rem 0; overflow: hidden;
        }}
        .ribbon .verdict {{
            padding: 0.9rem 1.3rem; flex: 1 1 260px; min-width: 220px;
            border-right: 1px solid {self.LINE}; background: {self.PANEL_2};
        }}
        .ribbon .verdict .line1 {{ font-size: 1.35rem; font-weight: 700; line-height: 1.25; }}
        .ribbon .verdict .line2 {{ font-size: 0.8rem; color: {self.MUTED}; margin-top: 0.3rem;
                                  line-height: 1.4; }}
        .ribbon .stat {{
            padding: 0.9rem 1.15rem; border-right: 1px solid {self.LINE};
            flex: 1 1 145px; min-width: 130px;
        }}
        .ribbon .stat:last-child {{ border-right: none; }}
        .ribbon .stat .v {{ font-size: 1.25rem; font-weight: 600; line-height: 1.3;
                           font-variant-numeric: tabular-nums; }}
        .ribbon .stat .l {{ font-size: 0.74rem; color: {self.MUTED}; margin-top: 0.25rem;
                           line-height: 1.35; }}

        /* ---- cards and inline notes ---- */
        .panel {{
            border: 1px solid {self.LINE}; background: {self.PANEL}; border-radius: 12px;
            padding: 0.85rem 1rem; margin-bottom: 0.7rem; line-height: 1.5;
        }}
        .note {{ color: {self.MUTED}; font-size: 0.85rem; line-height: 1.55; margin-bottom: 0.4rem; }}
        .step {{ color: {self.MUTED}; font-size: 0.78rem; font-weight: 500; margin-bottom: 0.35rem; }}
        .formula {{
            color: {self.TEXT}; background: {self.PANEL_2}; border-left: 2px solid {self.ACCENT};
            padding: 0.55rem 0.8rem; font-size: 0.84rem; line-height: 1.6; border-radius: 2px;
            font-variant-numeric: tabular-nums; margin: 0.35rem 0 0.7rem 0;
        }}
        .pill {{
            display: inline-block; padding: 0.1rem 0.5rem; border-radius: 999px;
            font-size: 0.74rem; font-weight: 500; line-height: 1.5; border: 1px solid;
        }}

        /* ---- the agent console ---- */
        .roster {{ display: flex; flex-wrap: wrap; gap: 0.45rem; margin-bottom: 0.9rem; }}
        .agentcard {{
            border: 1px solid {self.LINE}; border-top: 2px solid {self.ACCENT}; background: {self.PANEL};
            border-radius: 12px; padding: 0.55rem 0.75rem; flex: 1 1 155px; min-width: 145px;
        }}
        .agentcard .an {{ font-weight: 600; font-size: 0.88rem; line-height: 1.35; }}
        .agentcard .ao {{ color: {self.MUTED}; font-size: 0.72rem; line-height: 1.35; margin-top: 0.1rem; }}
        .agentcard .ac {{ color: {self.ACCENT}; font-size: 0.74rem; line-height: 1.35; margin-top: 0.35rem;
                         font-variant-numeric: tabular-nums; }}
        .msg {{
            border-left: 2px solid {self.LINE}; padding: 0.35rem 0 0.35rem 0.7rem;
            margin: 0.18rem 0; font-size: 0.83rem; line-height: 1.5;
        }}
        .msg .who {{ font-weight: 600; }}
        .msg .tp {{ color: {self.MUTED}; font-size: 0.74rem; }}
        .msg .bd {{ color: {self.TEXT}; }}
        .reply {{
            border: 1px solid {self.LINE}; border-left: 3px solid {self.ACCENT}; background: {self.PANEL_2};
            border-radius: 12px; padding: 0.9rem 1.1rem; margin: 0.5rem 0 0.8rem 0;
        }}
        .reply .hl {{ font-size: 1.02rem; font-weight: 600; line-height: 1.55; }}
        .reply .sec {{ font-size: 0.87rem; line-height: 1.55; margin-top: 0.5rem; }}
        .reply .sec b {{ color: {self.ACCENT}; font-weight: 600; }}
        .askrow {{ color: {self.MUTED}; font-size: 0.8rem; margin-bottom: 0.25rem; }}

        /* long values from a dataset wrap inside these cards rather than run past them */
        .panel, .reply, .msg, .agentcard, .ribbon, .note, .formula {{ overflow-wrap: break-word; }}

        [data-testid="stMainBlockContainer"] {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1680px; }}
        [data-testid="stHeader"] {{ background: {self.INK}; }}
        [data-testid="stMetric"] {{ background: {self.PANEL}; border: 1px solid {self.LINE}; border-radius: 12px; padding: 14px 18px; }}
        [data-testid="stPlotlyChart"] {{ border: 1px solid {self.LINE}; border-radius: 14px; overflow: hidden; }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 18px; padding-bottom: 8px; }}
        .stButton button {{ border-radius: 9px; border: 1px solid {self.LINE} !important;
                            background: {self.PANEL} !important; color: {self.TEXT} !important; }}
        .stButton button:hover {{ border-color: {self.ACCENT} !important; background: {self.PANEL_2} !important;
                                  color: {self.TEXT} !important; }}
        [data-testid="stBaseButton-secondary"] {{ background: {self.PANEL} !important;
                                                   color: {self.TEXT} !important;
                                                   border-color: {self.LINE} !important; }}
        [data-testid="stFormSubmitButton"] button,
        div.stFormSubmitButton > button,
        html body div.stFormSubmitButton > button,
        button[kind="primary"],
        [data-testid="stBaseButton-primary"],
        [data-testid="baseButton-primary"] {{ background: {self.ACCENT} !important; color: #FFFFFF !important;
                                   border-color: {self.ACCENT} !important; font-weight: 650; }}
        [data-testid="stFormSubmitButton"] button *,
        div.stFormSubmitButton > button *,
        html body div.stFormSubmitButton > button,
        html body div.stFormSubmitButton > button *,
        button[kind="primary"] *,
        [data-testid="stBaseButton-primary"] *,
        [data-testid="baseButton-primary"] * {{ color: #FFFFFF !important; }}
        [data-testid="stFormSubmitButton"],
        [data-testid="stFormSubmitButton"] *,
        div.stFormSubmitButton,
        div.stFormSubmitButton *,
        [data-testid="stForm"] button,
        [data-testid="stForm"] button *,
        .stForm button,
        .stForm button *,
        form button,
        form button * {{ color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important; }}
        [data-testid="stFormSubmitButton"] button:hover,
        div.stFormSubmitButton > button:hover,
        button[kind="primary"]:hover,
        [data-testid="stBaseButton-primary"]:hover,
        [data-testid="baseButton-primary"]:hover {{ background: #066B66 !important; color: #FFFFFF !important; }}
        [data-testid="stSidebar"] .stButton button {{ width: 100%; }}
        .ribbon {{ box-shadow: 0 4px 20px #172d3506; margin-top: 1rem; margin-bottom: 1.5rem; }}
        .step {{ text-transform: uppercase; letter-spacing: 0.12em; color: {self.ACCENT}; margin-top: 1rem; }}

        .ribbon {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); }}
        .ribbon .verdict {{ grid-column: 1 / -1; border-right: 0; border-bottom: 1px solid {self.LINE}; }}
        .ribbon .stat {{ min-width: 0; padding: 1rem; }}
        @media (max-width: 700px) {{
            .ribbon {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
        }}
        </style>
        """

    def inject(self) -> None:
        st.markdown(self.CSS, unsafe_allow_html=True)


    def fit_height(self, n_categories: int, per: int = 26, base: int = 120,
                   lo: int = 240, hi: int = 900) -> int:
        """Height for a chart with one label per category.

        A fixed height is fine until a dataset arrives with three times as many
        items, at which point the tick labels sit on top of each other. Sizing from
        the data instead means the chart cannot outgrow its own axis.
        """
        return int(min(max(base + per * max(int(n_categories), 1), lo), hi))


    def _legend_plan(self, fig: go.Figure, assumed_width: int) -> dict:
        """Decide where the legend goes and how much room to reserve for it.

        A horizontal legend sits above the plot. If its entries need more than one
        row, the extra rows are drawn down over the chart unless the top margin
        already allows for them. Rather than guess a fixed margin, measure the text:
        a Gantt coloured by job can carry twenty entries, and long series names such
        as "Gross requirement" fill a row on their own.
        """
        names = [str(t.name) for t in fig.data
                 if getattr(t, "name", None) and getattr(t, "showlegend", None) is not False]
        n = len(names)
        if n <= 1:
            return {"show": False, "top": 34, "right": None}
        # roughly 6.6px per character at 11px, plus the swatch and the gap after it
        width_px = sum(len(name) * 6.6 + 34 for name in names)
        if n > 14:
            # too many to label usefully; the colours still separate them and the
            # detail is on hover
            return {"show": False, "top": 34, "right": None}
        if width_px > assumed_width * 2:
            return {"show": True, "orient": "v", "top": 34, "right": 170}
        rows = max(1, int(width_px / assumed_width) + (1 if width_px % assumed_width else 0))
        return {"show": True, "orient": "h", "top": 30 + rows * 19, "right": None}


    def theme(self, fig: go.Figure, height: int = 340, legend: bool = True,
              right_pad: int = 10, assumed_width: int = 430) -> go.Figure:
        plan = self._legend_plan(fig, assumed_width) if legend else {"show": False, "top": 34,
                                                                "right": None}
        show = bool(legend and plan["show"])
        if show and plan.get("orient") == "v":
            legend_cfg = dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.01,
                              bgcolor="rgba(0,0,0,0)", font=dict(size=11))
        else:
            legend_cfg = dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                              bgcolor="rgba(0,0,0,0)", font=dict(size=11))
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor=self.PANEL, plot_bgcolor=self.PANEL,
            font=dict(family="IBM Plex Sans, sans-serif", size=12, color=self.TEXT),
            margin=dict(l=10, r=plan.get("right") or right_pad, t=plan["top"], b=10),
            height=height,
            colorway=self.SERIES,
            legend=legend_cfg if show else None,
            showlegend=show,
            hoverlabel=dict(bgcolor=self.PANEL_2, bordercolor=self.LINE,
                            font=dict(family="IBM Plex Sans, sans-serif", size=12)),
        )
        fig.update_xaxes(gridcolor=self.LINE, zerolinecolor=self.LINE, linecolor=self.LINE,
                         title_font=dict(size=11), automargin=True)
        fig.update_yaxes(gridcolor=self.LINE, zerolinecolor=self.LINE, linecolor=self.LINE,
                         title_font=dict(size=11), automargin=True)
        return fig


    def esc(self, value) -> str:
        """Anything coming from a dataset is escaped before it reaches the page.

        An item description containing an angle bracket would otherwise be read as
        markup and pull the rest of the card apart.
        """
        return html.escape(str(value), quote=False)


    def step(self, text: str) -> None:
        st.markdown(f"<div class='step'>{text}</div>", unsafe_allow_html=True)


    def note(self, text: str) -> None:
        st.markdown(f"<div class='note'>{text}</div>", unsafe_allow_html=True)


    def formula(self, text: str) -> None:
        st.markdown(f"<div class='formula'>{text}</div>", unsafe_allow_html=True)


    def pill(self, text: str, colour: str) -> str:
        return (f"<span class='pill' style='color:{colour};border-color:{colour};"
                f"background:{colour}1a'>{self.esc(text)}</span>")


    def ribbon(self, verdict: str, sub: str, colour: str, stats: list) -> None:
        cells = "".join(
            f"<div class='stat'><div class='v' style='color:{c}'>{self.esc(v)}</div>"
            f"<div class='l'>{self.esc(l)}</div></div>" for v, l, c in stats)
        st.markdown(
            f"<div class='ribbon' style='border-left-color:{colour}'>"
            f"<div class='verdict'><div class='line1' style='color:{colour}'>{self.esc(verdict)}</div>"
            f"<div class='line2'>{self.esc(sub)}</div></div>{cells}</div>",
            unsafe_allow_html=True)


    def status_frame(self, df: pd.DataFrame, status_col: str = "status",
                     colours: Optional[Dict[str, str]] = None):
        """Colour a status column without turning the table into a rainbow."""
        colours = colours or self.STATUS_COLOUR
        if status_col not in df.columns:
            return df

        def _style(v):
            c = colours.get(str(v))
            return f"color: {c}; font-weight: 500" if c else ""

        return df.style.map(_style, subset=[status_col])


    def download_frames(self, frames: Dict[str, pd.DataFrame], filename: str, label: str) -> None:
        """Offer a multi sheet workbook of the frames given."""
        import io
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as xl:
            for name, df in frames.items():
                if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
                    continue
                df.to_excel(xl, sheet_name=name[:31], index=False)
        st.download_button(label, buf.getvalue(), file_name=filename,
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
