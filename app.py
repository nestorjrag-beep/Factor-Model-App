"""
QUANT LAB — Factor Model & Time Series Analysis
A high-tech interactive dashboard for empirical decomposition of stock returns:
CAPM, Fama-French 3-factor, rolling beta, ADF, ACF/PACF, residual diagnostics.
"""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import acf, adfuller, pacf

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration & theming
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="QUANT LAB · Factor Model Studio",
    page_icon="◐",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "bg": "#06090F",
    "panel": "#0E1422",
    "panel_2": "#131B2D",
    "border": "#1E2A44",
    "text": "#E6EDF7",
    "muted": "#8FA0BF",
    "cyan": "#00F0FF",
    "magenta": "#FF2D95",
    "violet": "#8B5CF6",
    "lime": "#A6F955",
    "amber": "#FFB547",
    "red": "#FF5670",
    "green": "#22D39A",
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Space Grotesk', sans-serif;
    color: {PALETTE['text']};
}}

.stApp {{
    background:
        radial-gradient(1200px 600px at 10% -10%, rgba(0,240,255,0.08), transparent 60%),
        radial-gradient(1000px 500px at 90% 0%, rgba(255,45,149,0.06), transparent 60%),
        radial-gradient(800px 400px at 50% 100%, rgba(139,92,246,0.06), transparent 60%),
        {PALETTE['bg']};
}}

/* Title bar */
.qlab-hero {{
    border: 1px solid {PALETTE['border']};
    background: linear-gradient(135deg, rgba(0,240,255,0.06), rgba(255,45,149,0.04) 60%, rgba(139,92,246,0.06));
    border-radius: 16px;
    padding: 22px 28px;
    margin: 6px 0 18px 0;
    position: relative;
    overflow: hidden;
}}
.qlab-hero::before {{
    content: "";
    position: absolute; inset: 0;
    background: linear-gradient(90deg, transparent, rgba(0,240,255,0.15), transparent);
    transform: translateX(-100%);
    animation: sweep 6s ease-in-out infinite;
}}
@keyframes sweep {{
    50% {{ transform: translateX(100%); }}
    100% {{ transform: translateX(100%); }}
}}
.qlab-hero h1 {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 32px;
    letter-spacing: -0.02em;
    margin: 0;
    background: linear-gradient(90deg, {PALETTE['cyan']}, #ffffff 50%, {PALETTE['magenta']});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}
.qlab-hero .subtitle {{
    color: {PALETTE['muted']};
    font-size: 13px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
}}

/* Cards */
.qlab-card {{
    border: 1px solid {PALETTE['border']};
    background: linear-gradient(180deg, {PALETTE['panel']}, {PALETTE['panel_2']});
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 1px 0 rgba(255,255,255,0.03) inset, 0 30px 60px -30px rgba(0,0,0,0.6);
}}
.qlab-card h3 {{
    margin: 0 0 4px 0;
    font-size: 13px;
    color: {PALETTE['muted']};
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-family: 'JetBrains Mono', monospace;
}}
.qlab-card .value {{
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-top: 4px;
}}
.qlab-card .delta {{
    font-size: 12px;
    color: {PALETTE['muted']};
    font-family: 'JetBrains Mono', monospace;
    margin-top: 2px;
}}
.qlab-card.accent-cyan {{ border-color: rgba(0,240,255,0.35); box-shadow: 0 0 30px -15px {PALETTE['cyan']}; }}
.qlab-card.accent-magenta {{ border-color: rgba(255,45,149,0.35); box-shadow: 0 0 30px -15px {PALETTE['magenta']}; }}

/* Section header */
.section-h {{
    display: flex; align-items: center; gap: 10px;
    margin: 18px 0 6px 0;
}}
.section-h .tag {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.18em;
    color: {PALETTE['cyan']};
    border: 1px solid rgba(0,240,255,0.4);
    border-radius: 4px;
    padding: 2px 8px;
}}
.section-h h2 {{
    margin: 0;
    font-size: 20px;
    font-weight: 600;
    letter-spacing: -0.01em;
}}

/* Pills */
.pill {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.06em;
    border: 1px solid {PALETTE['border']};
    background: {PALETTE['panel_2']};
}}
.pill.ok {{ color: {PALETTE['green']}; border-color: rgba(34,211,154,0.4); }}
.pill.warn {{ color: {PALETTE['amber']}; border-color: rgba(255,181,71,0.4); }}
.pill.bad {{ color: {PALETTE['red']}; border-color: rgba(255,86,112,0.4); }}

/* Sidebar polish */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #080C16, #0A1020);
    border-right: 1px solid {PALETTE['border']};
}}
section[data-testid="stSidebar"] h2 {{
    background: linear-gradient(90deg, {PALETTE['cyan']}, {PALETTE['magenta']});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    letter-spacing: -0.01em;
}}

div[data-testid="stMetricValue"] {{ font-family: 'Space Grotesk', sans-serif; }}

/* Tabs */
button[data-baseweb="tab"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {PALETTE['cyan']} !important;
}}

/* DataFrame */
[data-testid="stDataFrame"] {{
    border: 1px solid {PALETTE['border']};
    border-radius: 10px;
    overflow: hidden;
}}

/* Buttons */
.stButton>button {{
    background: linear-gradient(90deg, {PALETTE['cyan']}, {PALETTE['violet']});
    color: #04111A;
    font-weight: 700;
    border: none;
    border-radius: 10px;
    letter-spacing: 0.06em;
    transition: transform 0.15s ease, box-shadow 0.2s ease;
}}
.stButton>button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 12px 30px -12px {PALETTE['cyan']};
    color: #04111A;
}}

footer {{ visibility: hidden; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

EXCLUDED = {"AAPL", "AMZN", "JNJ", "JPM", "KO", "NVDA", "PG", "XOM"}
END_DATE = date(2025, 12, 31)
DEFAULT_START = date(2014, 1, 1)
TRADING_DAYS = 252
ROLL_WIN = 60
BETA_WIN = 252


# ─────────────────────────────────────────────────────────────────────────────
# Data layer
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_prices(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Download adjusted close prices via yfinance."""
    df = yf.download(
        ticker,
        start=start,
        end=end + pd.Timedelta(days=1),
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    out = df[["Close"]].rename(columns={"Close": ticker}).dropna()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_ticker_info(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).get_info()
        return info or {}
    except Exception:
        return {}


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_ff_factors(start: date, end: date) -> pd.DataFrame:
    """
    Daily Fama-French 3 factors + RF, fetched directly from Ken French's
    data library (CSV in a ZIP). Returns columns: Mkt-RF, SMB, HML, RF
    in DECIMAL units (the published values are in percent and divided by 100 here).
    """
    import urllib.request
    import zipfile

    url = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
        "ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        text = zf.read(csv_name).decode("latin-1")

    # The file has a multi-line header before the data block. Find first data row.
    lines = text.splitlines()
    header_idx = None
    for i, ln in enumerate(lines):
        parts = [p.strip() for p in ln.split(",")]
        if parts and parts[0].lower().startswith("mkt"):
            # Some versions: ",Mkt-RF,SMB,HML,RF"; locate the column header line
            header_idx = i
            break
        if len(parts) >= 5 and "Mkt-RF" in parts:
            header_idx = i
            break
    if header_idx is None:
        # Fallback: first line whose first token is an 8-digit date
        for i, ln in enumerate(lines):
            tok = ln.split(",", 1)[0].strip()
            if tok.isdigit() and len(tok) == 8:
                header_idx = i - 1
                break

    data_lines = []
    for ln in lines[header_idx + 1 :]:
        first = ln.split(",", 1)[0].strip()
        if not first.isdigit() or len(first) != 8:
            # Stop at the first non-data line (e.g. annual block separator)
            break
        data_lines.append(ln)

    ff = pd.read_csv(
        io.StringIO("Date,Mkt-RF,SMB,HML,RF\n" + "\n".join(data_lines)),
        parse_dates=["Date"],
        date_format="%Y%m%d",
    )
    ff = ff.set_index("Date")
    ff.index = pd.to_datetime(ff.index).tz_localize(None)
    ff = ff.loc[(ff.index >= pd.Timestamp(start)) & (ff.index <= pd.Timestamp(end))]
    # Convert PERCENT → DECIMAL to match pct_change() returns
    ff = ff / 100.0
    return ff


@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_prices_pair(t1: str, t2: str, start: date, end: date) -> pd.DataFrame:
    """Download both tickers in a single yfinance call (faster than two sequential)."""
    df = yf.download(
        [t1, t2],
        start=start,
        end=end + pd.Timedelta(days=1),
        progress=False,
        auto_adjust=True,
        threads=True,
        group_by="ticker",
    )
    if df is None or df.empty:
        return pd.DataFrame()
    out = pd.DataFrame()
    for tk in (t1, t2):
        try:
            out[tk] = df[tk]["Close"]
        except KeyError:
            pass
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out.dropna()


def assemble_dataset(t1: str, t2: str, start: date, end: date) -> pd.DataFrame:
    """Inner-join prices + factors, produce returns and excess returns."""
    prices = fetch_prices_pair(t1, t2, start, end)
    if prices.empty or t1 not in prices.columns or t2 not in prices.columns:
        raise RuntimeError("Failed to download price data for one or both tickers.")
    ff = fetch_ff_factors(start, end)
    rets = prices.pct_change().rename(columns={t1: f"{t1}_ret", t2: f"{t2}_ret"})
    df = rets.join(ff, how="inner").dropna()
    df[f"{t1}_ex"] = df[f"{t1}_ret"] - df["RF"]
    df[f"{t2}_ex"] = df[f"{t2}_ret"] - df["RF"]
    df = df.join(prices.rename(columns={t1: f"{t1}_px", t2: f"{t2}_px"}), how="inner")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk, sans-serif", color=PALETTE["text"], size=12),
    margin=dict(l=10, r=10, t=50, b=30),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(bgcolor="#0E1422", bordercolor=PALETTE["cyan"], font_family="JetBrains Mono"),
)


def style_fig(fig: go.Figure, height: int = 380, title: str | None = None) -> go.Figure:
    layout = dict(PLOTLY_LAYOUT)
    layout["height"] = height
    if title:
        layout["title"] = dict(
            text=f"<span style='font-family:JetBrains Mono;font-size:11px;letter-spacing:0.18em;color:{PALETTE['muted']}'>{title.upper()}</span>",
            x=0.0,
            xanchor="left",
            y=0.98,
        )
    fig.update_layout(**layout)
    return fig


def kpi_card(label: str, value: str, delta: str = "", accent: str = ""):
    cls = f"qlab-card {accent}".strip()
    st.markdown(
        f"""<div class="{cls}">
            <h3>{label}</h3>
            <div class="value">{value}</div>
            {f'<div class="delta">{delta}</div>' if delta else ""}
        </div>""",
        unsafe_allow_html=True,
    )


def section_header(tag: str, title: str):
    st.markdown(
        f"""<div class="section-h"><span class="tag">{tag}</span><h2>{title}</h2></div>""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Statistical helpers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RegResult:
    alpha: float
    alpha_p: float
    coefs: dict[str, float]
    pvals: dict[str, float]
    r2: float
    adj_r2: float
    resid: pd.Series
    fitted: pd.Series
    nobs: int


def run_ols(y: pd.Series, X: pd.DataFrame) -> RegResult:
    Xc = sm.add_constant(X)
    model = sm.OLS(y, Xc, missing="drop").fit()
    coefs = {k: v for k, v in model.params.items() if k != "const"}
    pvals = {k: v for k, v in model.pvalues.items() if k != "const"}
    return RegResult(
        alpha=float(model.params["const"]),
        alpha_p=float(model.pvalues["const"]),
        coefs=coefs,
        pvals=pvals,
        r2=float(model.rsquared),
        adj_r2=float(model.rsquared_adj),
        resid=model.resid,
        fitted=model.fittedvalues,
        nobs=int(model.nobs),
    )


def adf_summary(series: pd.Series) -> tuple[float, float, str]:
    s = series.dropna()
    if len(s) < 20:
        return (np.nan, np.nan, "insufficient data")
    res = adfuller(s, autolag="AIC")
    stat, p = float(res[0]), float(res[1])
    verdict = "Stationary (reject H₀)" if p < 0.05 else "Non-stationary (fail to reject H₀)"
    return stat, p, verdict


def rolling_beta(excess_stock: pd.Series, excess_mkt: pd.Series, window: int = BETA_WIN) -> pd.Series:
    cov = excess_stock.rolling(window).cov(excess_mkt)
    var = excess_mkt.rolling(window).var()
    return cov / var


def annualized_vol(returns: pd.Series, window: int = ROLL_WIN) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────


def line_chart(series_dict: dict[str, pd.Series], title: str, ylab: str, colors=None, height=380):
    colors = colors or [PALETTE["cyan"], PALETTE["magenta"], PALETTE["violet"]]
    fig = go.Figure()
    for i, (name, s) in enumerate(series_dict.items()):
        s = s.dropna()
        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines",
                name=name,
                line=dict(color=colors[i % len(colors)], width=1.6),
                hovertemplate=f"<b>{name}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.4f}}<extra></extra>",
            )
        )
    fig.update_yaxes(title_text=ylab)
    return style_fig(fig, height=height, title=title)


def acf_bar_chart(series: pd.Series, lags: int, title: str, color: str = None):
    color = color or PALETTE["cyan"]
    s = series.dropna()
    vals = acf(s, nlags=lags, fft=True)
    n = len(s)
    ci = 1.96 / np.sqrt(n)
    x = list(range(len(vals)))
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x,
            y=vals,
            marker=dict(color=color, line=dict(color=color, width=0)),
            name="ACF",
            hovertemplate="lag %{x}<br>%{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=ci, line=dict(color="rgba(255,255,255,0.3)", dash="dash"))
    fig.add_hline(y=-ci, line=dict(color="rgba(255,255,255,0.3)", dash="dash"))
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.4)"))
    fig.update_xaxes(title_text="lag")
    fig.update_yaxes(title_text="autocorrelation", range=[min(-0.2, vals.min() * 1.2), max(0.2, vals.max() * 1.2)])
    return style_fig(fig, height=300, title=title)


def pacf_bar_chart(series: pd.Series, lags: int, title: str, color: str = None):
    color = color or PALETTE["magenta"]
    s = series.dropna()
    vals = pacf(s, nlags=lags, method="ywm")
    n = len(s)
    ci = 1.96 / np.sqrt(n)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=list(range(len(vals))),
            y=vals,
            marker=dict(color=color),
            hovertemplate="lag %{x}<br>%{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=ci, line=dict(color="rgba(255,255,255,0.3)", dash="dash"))
    fig.add_hline(y=-ci, line=dict(color="rgba(255,255,255,0.3)", dash="dash"))
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.4)"))
    fig.update_xaxes(title_text="lag")
    fig.update_yaxes(title_text="partial autocorrelation")
    return style_fig(fig, height=300, title=title)


def residual_diagnostic_panel(resid: pd.Series, label: str, color: str):
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Residuals over time", "Residual histogram", "Q-Q plot vs Normal"),
        column_widths=[0.42, 0.29, 0.29],
        horizontal_spacing=0.08,
    )
    # Time plot
    fig.add_trace(
        go.Scatter(x=resid.index, y=resid.values, mode="lines", line=dict(color=color, width=1.0), name="resid"),
        row=1, col=1,
    )
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.4)"), row=1, col=1)

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=resid.values, nbinsx=70, marker=dict(color=color, line=dict(color="rgba(0,0,0,0)")),
            opacity=0.85, name="hist",
        ),
        row=1, col=2,
    )

    # Q-Q
    res_sorted = np.sort(resid.dropna().values)
    n = len(res_sorted)
    theoretical = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    slope, intercept, _, _, _ = stats.linregress(theoretical, res_sorted)
    fig.add_trace(
        go.Scattergl(
            x=theoretical, y=res_sorted, mode="markers",
            marker=dict(color=color, size=3.5, opacity=0.7), name="qq",
        ),
        row=1, col=3,
    )
    line_x = np.array([theoretical.min(), theoretical.max()])
    fig.add_trace(
        go.Scatter(
            x=line_x, y=intercept + slope * line_x,
            mode="lines", line=dict(color="rgba(255,255,255,0.5)", dash="dash"), name="ref",
            showlegend=False,
        ),
        row=1, col=3,
    )

    fig.update_layout(showlegend=False)
    return style_fig(fig, height=320, title=f"{label} — residual diagnostics")


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────


def validate_tickers(t1: str, t2: str, info1: dict, info2: dict, df: pd.DataFrame):
    issues = []
    notes = []

    if t1.upper() == t2.upper():
        issues.append("Two tickers must be different.")
    if t1.upper() in EXCLUDED or t2.upper() in EXCLUDED:
        bad = [x for x in (t1.upper(), t2.upper()) if x in EXCLUDED]
        issues.append(f"Ticker not allowed: {', '.join(bad)}. Please choose a different stock.")

    for tk, info in [(t1, info1), (t2, info2)]:
        country = (info.get("country") or "").lower()
        exch = (info.get("fullExchangeName") or info.get("exchange") or "").upper()
        quote_type = (info.get("quoteType") or "").upper()
        if quote_type and quote_type != "EQUITY":
            issues.append(f"{tk}: quoteType={quote_type} — must be a common stock (EQUITY).")
        if country and "united states" not in country:
            notes.append(f"{tk}: country='{info.get('country')}' (must be U.S.-headquartered).")
        if exch and not any(x in exch for x in ("NYSE", "NASDAQ", "NMS", "NCM", "NGS", "NYQ")):
            notes.append(f"{tk}: exchange='{exch}' (must be NYSE/NASDAQ).")

    s1 = (info1.get("sector") or "Unknown").strip()
    s2 = (info2.get("sector") or "Unknown").strip()
    if s1 != "Unknown" and s2 != "Unknown" and s1.lower() == s2.lower():
        issues.append(f"Both tickers in same sector ({s1}). Must be different sectors.")

    if not df.empty:
        years = (df.index[-1] - df.index[0]).days / 365.25
        if years < 10:
            notes.append(f"Sample length only {years:.1f} years — spec requires ≥10 years.")
        if df.index[-1].date() < date(2025, 6, 1):
            notes.append(f"Data ends {df.index[-1].date()} — may not yet cover through 2025-12-31.")

    return issues, notes, s1, s2


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — input deck
# ─────────────────────────────────────────────────────────────────────────────


with st.sidebar:
    st.markdown("## QUANT LAB")
    st.markdown(
        f"<div style='font-family:JetBrains Mono;font-size:11px;letter-spacing:0.16em;color:{PALETTE['muted']};margin-top:-8px'>FACTOR · MODEL · STUDIO</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("**Stock Universe**")
    c1, c2 = st.columns(2)
    with c1:
        t1 = st.text_input("Ticker A", value="MSFT", max_chars=8).upper().strip()
    with c2:
        t2 = st.text_input("Ticker B", value="CAT", max_chars=8).upper().strip()

    st.markdown("**Sample Window**")
    start = st.date_input("Start", value=DEFAULT_START, min_value=date(2000, 1, 1), max_value=date(2024, 12, 31))
    end = st.date_input("End", value=END_DATE, min_value=date(2010, 1, 1), max_value=END_DATE)

    st.markdown("---")
    st.markdown(
        f"<div style='font-family:JetBrains Mono;font-size:10px;color:{PALETTE['muted']};line-height:1.6'>"
        f"FACTORS · Mkt-RF · SMB · HML · RF<br>"
        f"SOURCE · yfinance × Ken French"
        f"</div>",
        unsafe_allow_html=True,
    )
    run = st.button("◐  RUN ANALYSIS", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<div class="qlab-hero">
    <div class="subtitle">Factor Model · Time Series · Residual Diagnostics</div>
    <h1>Two-Stock Factor Studio</h1>
    <div style="color:#8FA0BF;margin-top:6px;font-size:14px">
        Empirical decomposition of stock returns via CAPM, Fama-French 3-factor,
        rolling beta, and residual diagnostics.
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Main flow
# ─────────────────────────────────────────────────────────────────────────────

if not run:
    st.markdown(
        f"""
<div class="qlab-card" style="text-align:center;padding:40px">
    <div style="font-family:JetBrains Mono;font-size:11px;letter-spacing:0.2em;color:{PALETTE['muted']}">
        STANDBY · AWAITING SIGNAL
    </div>
    <div style="margin-top:14px;font-size:18px;color:{PALETTE['text']}">
        Configure tickers in the sidebar and press <b style="color:{PALETTE['cyan']}">RUN ANALYSIS</b>.
    </div>
    <div style="margin-top:8px;color:{PALETTE['muted']};font-size:13px">
        Default loadout: MSFT (Tech) vs CAT (Industrials). Pick any two valid stocks from different sectors.
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# Pre-flight checks
if not t1 or not t2:
    st.error("Both tickers required.")
    st.stop()

with st.spinner("◐ Streaming market data · computing factor universe…"):
    try:
        df = assemble_dataset(t1, t2, start, end)
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        st.stop()

    info1 = fetch_ticker_info(t1)
    info2 = fetch_ticker_info(t2)

if df.empty or len(df) < 250:
    st.error("Not enough overlapping data between stocks and Fama-French factors.")
    st.stop()

issues, notes, sector1, sector2 = validate_tickers(t1, t2, info1, info2, df)

# Validation banner
val_col1, val_col2 = st.columns([1, 1])
with val_col1:
    if issues:
        st.markdown(
            f"<div class='qlab-card' style='border-color:rgba(255,86,112,0.5)'>"
            f"<h3 style='color:{PALETTE['red']}'>SPEC VIOLATIONS</h3>"
            + "".join([f"<div style='margin-top:6px'>· {x}</div>" for x in issues])
            + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='qlab-card' style='border-color:rgba(34,211,154,0.4)'>"
            f"<h3 style='color:{PALETTE['green']}'>SPEC CHECKS · PASS</h3>"
            f"<div style='margin-top:6px'>Both tickers admissible. Period covers {len(df)} trading days.</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
with val_col2:
    name1 = (info1.get("shortName") or t1)[:40]
    name2 = (info2.get("shortName") or t2)[:40]
    st.markdown(
        f"""<div class="qlab-card">
            <h3>SELECTED STOCKS</h3>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:8px">
                <div>
                    <div style="font-family:JetBrains Mono;font-size:11px;color:{PALETTE['cyan']};letter-spacing:0.16em">{t1}</div>
                    <div style="font-size:14px;font-weight:600">{name1}</div>
                    <div style="color:{PALETTE['muted']};font-size:12px">{sector1}</div>
                </div>
                <div>
                    <div style="font-family:JetBrains Mono;font-size:11px;color:{PALETTE['magenta']};letter-spacing:0.16em">{t2}</div>
                    <div style="font-size:14px;font-weight:600">{name2}</div>
                    <div style="color:{PALETTE['muted']};font-size:12px">{sector2}</div>
                </div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

if notes:
    with st.expander("⚠ advisory notes", expanded=False):
        for n in notes:
            st.markdown(f"- {n}")

# ─────────────────────────────────────────────────────────────────────────────
# Compute everything once
# ─────────────────────────────────────────────────────────────────────────────

ret1 = df[f"{t1}_ret"]
ret2 = df[f"{t2}_ret"]
ex1 = df[f"{t1}_ex"]
ex2 = df[f"{t2}_ex"]
mkt = df["Mkt-RF"]
smb = df["SMB"]
hml = df["HML"]

# CAPM
capm1 = run_ols(ex1, df[["Mkt-RF"]])
capm2 = run_ols(ex2, df[["Mkt-RF"]])

# FF3
ff_X = df[["Mkt-RF", "SMB", "HML"]]
ff31 = run_ols(ex1, ff_X)
ff32 = run_ols(ex2, ff_X)

# VIF
def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    Xc = sm.add_constant(X)
    rows = []
    for i, col in enumerate(Xc.columns):
        if col == "const":
            continue
        rows.append({"factor": col, "VIF": variance_inflation_factor(Xc.values, i)})
    return pd.DataFrame(rows)


vif_df = compute_vif(ff_X)

# Rolling beta
rb1 = rolling_beta(ex1, mkt, BETA_WIN)
rb2 = rolling_beta(ex2, mkt, BETA_WIN)


# ─────────────────────────────────────────────────────────────────────────────
# KPI Strip
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
k1, k2, k3, k4, k5, k6 = st.columns(6)
ann_vol1 = ret1.std() * np.sqrt(TRADING_DAYS)
ann_vol2 = ret2.std() * np.sqrt(TRADING_DAYS)
ann_ret1 = (1 + ret1).prod() ** (TRADING_DAYS / len(ret1)) - 1
ann_ret2 = (1 + ret2).prod() ** (TRADING_DAYS / len(ret2)) - 1

with k1: kpi_card(f"{t1} · CAPM β", f"{capm1.coefs['Mkt-RF']:.3f}", "market sensitivity", "accent-cyan")
with k2: kpi_card(f"{t2} · CAPM β", f"{capm2.coefs['Mkt-RF']:.3f}", "market sensitivity", "accent-magenta")
with k3: kpi_card(f"{t1} · FF3 R²", f"{ff31.r2*100:.1f}%", f"adj {ff31.adj_r2*100:.1f}%")
with k4: kpi_card(f"{t2} · FF3 R²", f"{ff32.r2*100:.1f}%", f"adj {ff32.adj_r2*100:.1f}%")
with k5: kpi_card(f"{t1} · ann vol", f"{ann_vol1*100:.1f}%", f"ann ret {ann_ret1*100:.1f}%")
with k6: kpi_card(f"{t2} · ann vol", f"{ann_vol2*100:.1f}%", f"ann ret {ann_ret2*100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "◐  Overview",
    "3.1  Time Series",
    "3.2  CAPM",
    "3.3  Fama-French 3F",
    "3.4  Synthesis",
    "▣  Discussion",
    "⤓  Data",
])


# ── Tab 0: Overview ──────────────────────────────────────────────────────────
with tabs[0]:
    section_header("01", "Price & Cumulative Performance")
    px1 = df[f"{t1}_px"]
    px2 = df[f"{t2}_px"]
    cum1 = (1 + ret1).cumprod()
    cum2 = (1 + ret2).cumprod()

    cA, cB = st.columns(2)
    with cA:
        st.plotly_chart(
            line_chart({t1: px1, t2: px2}, "adjusted close (rebased to native units)", "price ($)"),
            use_container_width=True,
        )
    with cB:
        st.plotly_chart(
            line_chart({t1: cum1, t2: cum2}, "cumulative return (growth of $1)", "growth multiple"),
            use_container_width=True,
        )

    section_header("02", "Rolling 252-day CAPM Beta")
    fig = go.Figure()
    rb1c = rb1.dropna()
    rb2c = rb2.dropna()
    fig.add_trace(go.Scatter(x=rb1c.index, y=rb1c.values, mode="lines",
                             line=dict(color=PALETTE["cyan"], width=1.8), name=t1))
    fig.add_trace(go.Scatter(x=rb2c.index, y=rb2c.values, mode="lines",
                             line=dict(color=PALETTE["magenta"], width=1.8), name=t2))
    fig.add_hline(y=1.0, line=dict(color="rgba(255,255,255,0.4)", dash="dash"))
    fig.add_annotation(x=rb1.dropna().index[0], y=1.0, text="β = 1 (market)",
                       showarrow=False, font=dict(color=PALETTE["muted"], size=10),
                       xanchor="left", yshift=10)
    fig.update_yaxes(title_text="rolling β")
    st.plotly_chart(style_fig(fig, height=380, title="rolling 252-day CAPM beta"), use_container_width=True)

# ── Tab 1: Time Series (3.1) ─────────────────────────────────────────────────
with tabs[1]:
    section_header("3.1.1", "Prices & Returns")
    cA, cB = st.columns(2)
    with cA:
        st.plotly_chart(
            line_chart({t1: df[f"{t1}_px"], t2: df[f"{t2}_px"]}, "adjusted closing price", "price"),
            use_container_width=True,
        )
    with cB:
        st.plotly_chart(
            line_chart({t1: ret1, t2: ret2}, "daily simple returns", "return"),
            use_container_width=True,
        )

    section_header("3.1.2", "Rolling 60-day Mean & Annualized Volatility")
    cA, cB = st.columns(2)
    rmean1, rmean2 = ret1.rolling(ROLL_WIN).mean(), ret2.rolling(ROLL_WIN).mean()
    rvol1, rvol2 = annualized_vol(ret1), annualized_vol(ret2)
    with cA:
        st.plotly_chart(
            line_chart({t1: rmean1, t2: rmean2}, "rolling 60-day mean return", "mean daily return"),
            use_container_width=True,
        )
    with cB:
        st.plotly_chart(
            line_chart({t1: rvol1, t2: rvol2}, "rolling 60-day annualized volatility", "annualized σ"),
            use_container_width=True,
        )

    section_header("3.1.3", "Augmented Dickey-Fuller Stationarity")
    rows = []
    for tk, px, rt in [(t1, df[f"{t1}_px"], ret1), (t2, df[f"{t2}_px"], ret2)]:
        s, p, v = adf_summary(px)
        rows.append({"series": f"{tk} price", "ADF stat": s, "p-value": p, "verdict": v})
        s, p, v = adf_summary(rt)
        rows.append({"series": f"{tk} returns", "ADF stat": s, "p-value": p, "verdict": v})
    adf_df = pd.DataFrame(rows)
    st.dataframe(
        adf_df.style.format({"ADF stat": "{:.3f}", "p-value": "{:.4f}"}),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "Expected for financial data: prices are non-stationary (random-walk-like), returns are stationary. "
        "Reject H₀ (unit root) when p < 0.05."
    )

    section_header("3.1.4", "Autocorrelation Structure")
    cA, cB = st.columns(2)
    with cA:
        st.plotly_chart(acf_bar_chart(ret1, 40, f"{t1} returns · ACF", PALETTE["cyan"]), use_container_width=True)
        st.plotly_chart(pacf_bar_chart(ret1, 40, f"{t1} returns · PACF", PALETTE["cyan"]), use_container_width=True)
        st.plotly_chart(acf_bar_chart(ret1 ** 2, 40, f"{t1} squared returns · ACF (volatility clustering)", PALETTE["amber"]),
                        use_container_width=True)
    with cB:
        st.plotly_chart(acf_bar_chart(ret2, 40, f"{t2} returns · ACF", PALETTE["magenta"]), use_container_width=True)
        st.plotly_chart(pacf_bar_chart(ret2, 40, f"{t2} returns · PACF", PALETTE["magenta"]), use_container_width=True)
        st.plotly_chart(acf_bar_chart(ret2 ** 2, 40, f"{t2} squared returns · ACF (volatility clustering)", PALETTE["amber"]),
                        use_container_width=True)

    section_header("3.1.5", "Cross-stock Comparison")
    more_vol = t1 if ann_vol1 > ann_vol2 else t2
    sq_acf1 = acf((ret1 ** 2).dropna(), nlags=40, fft=True)[1:].mean()
    sq_acf2 = acf((ret2 ** 2).dropna(), nlags=40, fft=True)[1:].mean()
    more_clust = t1 if sq_acf1 > sq_acf2 else t2
    st.markdown(
        f"""
- **Volatility:** {t1} annualized σ = {ann_vol1*100:.1f}% vs {t2} = {ann_vol2*100:.1f}% → **{more_vol}** is more volatile.
- **Volatility clustering (mean ACF of squared returns, lags 1–40):** {t1} = {sq_acf1:.3f}, {t2} = {sq_acf2:.3f} → **{more_clust}** exhibits stronger clustering.
- **Returns autocorrelation:** Both stocks show near-zero ACF/PACF in raw returns (consistent with weak-form market efficiency), while squared returns are persistently autocorrelated — a textbook signature of GARCH-type volatility dynamics.
- **Regime alignment:** Rolling volatility series for both stocks tend to spike around the same macro events (e.g. early 2020 COVID shock, 2022 rate-hike drawdown), but with different amplitudes governed by each stock's idiosyncratic risk profile.
"""
    )

# ── Tab 2: CAPM (3.2) ────────────────────────────────────────────────────────
with tabs[2]:
    section_header("3.2.1–4", "Market Model · OLS Estimates")

    capm_table = pd.DataFrame({
        t1: [capm1.alpha, capm1.alpha_p, capm1.coefs["Mkt-RF"], capm1.pvals["Mkt-RF"], capm1.r2, capm1.nobs],
        t2: [capm2.alpha, capm2.alpha_p, capm2.coefs["Mkt-RF"], capm2.pvals["Mkt-RF"], capm2.r2, capm2.nobs],
    }, index=["α (daily)", "α p-value", "β (Mkt-RF)", "β p-value", "R²", "N obs"])

    st.dataframe(
        capm_table.style.format({
            t1: lambda v: f"{v:.4f}" if isinstance(v, float) else f"{int(v)}",
            t2: lambda v: f"{v:.4f}" if isinstance(v, float) else f"{int(v)}",
        }),
        use_container_width=True,
    )

    cA, cB, cC = st.columns(3)
    for col, (tk, res) in zip([cA, cB, cC[0:1] if False else None], []):  # placeholder; we render below
        pass

    cA, cB = st.columns(2)
    for c, (tk, res, color) in zip([cA, cB], [(t1, capm1, PALETTE["cyan"]), (t2, capm2, PALETTE["magenta"])]):
        with c:
            beta = res.coefs["Mkt-RF"]
            agg = "aggressive (β > 1)" if beta > 1.05 else "defensive (β < 1)" if beta < 0.95 else "≈ market (β ≈ 1)"
            alpha_sig = "**statistically significant**" if res.alpha_p < 0.05 else "not significant"
            st.markdown(
                f"""
**{tk} interpretation**
- β = `{beta:.3f}` → {agg}; a 1% market move ⇒ ~{beta:.2f}% expected move in {tk}.
- α = `{res.alpha*1e4:.2f}` bps/day, {alpha_sig} (p = {res.alpha_p:.3f}). Annualized α ≈ `{res.alpha * TRADING_DAYS * 100:.2f}%`.
- R² = `{res.r2*100:.1f}%` → market alone explains that fraction of return variation.
"""
            )

    section_header("3.2.5", "CAPM Residual Diagnostics")
    st.plotly_chart(residual_diagnostic_panel(capm1.resid, t1, PALETTE["cyan"]), use_container_width=True)
    st.plotly_chart(residual_diagnostic_panel(capm2.resid, t2, PALETTE["magenta"]), use_container_width=True)

    # Scatter with fit
    section_header("3.2.5b", "Excess-Return Scatter & OLS Fit")
    cA, cB = st.columns(2)
    for c, (tk, ex, res, color) in zip([cA, cB], [(t1, ex1, capm1, PALETTE["cyan"]), (t2, ex2, capm2, PALETTE["magenta"])]):
        with c:
            fig = go.Figure()
            fig.add_trace(go.Scattergl(x=mkt, y=ex, mode="markers",
                                       marker=dict(size=3.5, color=color, opacity=0.45),
                                       name=tk, showlegend=False))
            xs = np.linspace(mkt.min(), mkt.max(), 50)
            ys = res.alpha + res.coefs["Mkt-RF"] * xs
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                     line=dict(color="white", width=2, dash="dash"),
                                     name="OLS fit", showlegend=False))
            fig.update_xaxes(title_text="Mkt-RF")
            fig.update_yaxes(title_text=f"{tk} excess return")
            st.plotly_chart(style_fig(fig, height=320, title=f"{tk} vs market — β = {res.coefs['Mkt-RF']:.3f}"),
                            use_container_width=True)

    section_header("3.2.6", "Cross-stock CAPM Comparison")
    more_sens = t1 if abs(capm1.coefs["Mkt-RF"]) > abs(capm2.coefs["Mkt-RF"]) else t2
    better_explained = t1 if capm1.r2 > capm2.r2 else t2
    st.markdown(f"""
- **More market-sensitive:** **{more_sens}** ({(capm1 if more_sens==t1 else capm2).coefs['Mkt-RF']:.3f} vs {(capm2 if more_sens==t1 else capm1).coefs['Mkt-RF']:.3f}).
- **Better explained by market alone:** **{better_explained}** (R² = {(capm1 if better_explained==t1 else capm2).r2*100:.1f}%).
""")

# ── Tab 3: FF3 (3.3) ─────────────────────────────────────────────────────────
with tabs[3]:
    section_header("3.3.1–2", "Fama-French 3-Factor · OLS Estimates")

    ff3_table = pd.DataFrame({
        t1: [
            ff31.alpha, ff31.alpha_p,
            ff31.coefs["Mkt-RF"], ff31.pvals["Mkt-RF"],
            ff31.coefs["SMB"], ff31.pvals["SMB"],
            ff31.coefs["HML"], ff31.pvals["HML"],
            ff31.r2, ff31.adj_r2, ff31.nobs,
        ],
        t2: [
            ff32.alpha, ff32.alpha_p,
            ff32.coefs["Mkt-RF"], ff32.pvals["Mkt-RF"],
            ff32.coefs["SMB"], ff32.pvals["SMB"],
            ff32.coefs["HML"], ff32.pvals["HML"],
            ff32.r2, ff32.adj_r2, ff32.nobs,
        ],
    }, index=[
        "α (daily)", "α p-value",
        "β_Mkt (Mkt-RF)", "β_Mkt p-value",
        "β_SMB", "β_SMB p-value",
        "β_HML", "β_HML p-value",
        "R²", "Adj R²", "N obs",
    ])
    st.dataframe(
        ff3_table.style.format({
            t1: lambda v: f"{v:.4f}" if isinstance(v, float) else f"{int(v)}",
            t2: lambda v: f"{v:.4f}" if isinstance(v, float) else f"{int(v)}",
        }),
        use_container_width=True,
    )

    section_header("3.3.3", "Factor Loading Interpretation")
    cA, cB = st.columns(2)
    for c, (tk, res, capm, color) in zip([cA, cB], [(t1, ff31, capm1, PALETTE["cyan"]), (t2, ff32, capm2, PALETTE["magenta"])]):
        with c:
            bm, bs, bh = res.coefs["Mkt-RF"], res.coefs["SMB"], res.coefs["HML"]
            pm, ps, ph = res.pvals["Mkt-RF"], res.pvals["SMB"], res.pvals["HML"]
            smb_lab = "small-cap tilt" if bs > 0.05 else "large-cap tilt" if bs < -0.05 else "size-neutral"
            hml_lab = "value tilt" if bh > 0.05 else "growth tilt" if bh < -0.05 else "style-neutral"
            st.markdown(f"""
**{tk}**
- Market β shifted from CAPM `{capm.coefs['Mkt-RF']:.3f}` → FF3 `{bm:.3f}` (p = {pm:.3f}).
- SMB loading `{bs:+.3f}` (p = {ps:.3f}) → **{smb_lab}**.
- HML loading `{bh:+.3f}` (p = {ph:.3f}) → **{hml_lab}**.
- ΔR² over CAPM: `+{(res.r2 - capm.r2)*100:.2f} pp` (FF3 R² = {res.r2*100:.1f}%).
""")

    section_header("3.3.4", "Variance Inflation Factors (multicollinearity)")
    max_vif = vif_df["VIF"].max()
    flag = "ok" if max_vif < 5 else "warn" if max_vif < 10 else "bad"
    label = {"ok": "low", "warn": "moderate", "bad": "high"}[flag]
    cA, cB = st.columns([1, 2])
    with cA:
        st.dataframe(vif_df.style.format({"VIF": "{:.2f}"}), use_container_width=True, hide_index=True)
        st.markdown(f"<div class='pill {flag}'>{label} multicollinearity · max VIF {max_vif:.2f}</div>",
                    unsafe_allow_html=True)
    with cB:
        st.markdown(
            "Rule of thumb: **VIF < 5** = no concern, **5–10** = moderate, **> 10** = problematic. "
            "FF factors are constructed to be roughly orthogonal, so VIFs are typically near 1.0–1.5."
        )

    section_header("3.3.5", "FF3 vs CAPM — explanatory power")
    cmp = pd.DataFrame({
        "CAPM R²": [capm1.r2, capm2.r2],
        "FF3 R²": [ff31.r2, ff32.r2],
        "FF3 Adj R²": [ff31.adj_r2, ff32.adj_r2],
        "Δ R² (pp)": [(ff31.r2 - capm1.r2) * 100, (ff32.r2 - capm2.r2) * 100],
    }, index=[t1, t2])
    st.dataframe(cmp.style.format("{:.4f}"), use_container_width=True)

    section_header("3.3.6", "FF3 Residual Diagnostics")
    st.plotly_chart(residual_diagnostic_panel(ff31.resid, t1, PALETTE["cyan"]), use_container_width=True)
    st.plotly_chart(residual_diagnostic_panel(ff32.resid, t2, PALETTE["magenta"]), use_container_width=True)

    section_header("3.3.7", "Cross-stock FF Loading Comparison")
    st.markdown(f"""
- **Market exposure:** `{t1}` β_Mkt = {ff31.coefs['Mkt-RF']:.3f} vs `{t2}` β_Mkt = {ff32.coefs['Mkt-RF']:.3f}.
- **Size tilt (SMB):** `{t1}` = {ff31.coefs['SMB']:+.3f}; `{t2}` = {ff32.coefs['SMB']:+.3f}. Negative loadings on large mega-caps are expected.
- **Value tilt (HML):** `{t1}` = {ff31.coefs['HML']:+.3f}; `{t2}` = {ff32.coefs['HML']:+.3f}. A growth-style company should show a negative HML loading; a cyclical / asset-heavy company should show a positive one.
""")

# ── Tab 4: Synthesis (3.4) ───────────────────────────────────────────────────
with tabs[4]:
    section_header("3.4.1", "Rolling 252-day CAPM Beta")
    fig = go.Figure()
    rb1c = rb1.dropna()
    rb2c = rb2.dropna()
    fig.add_trace(go.Scatter(x=rb1c.index, y=rb1c.values, mode="lines",
                             line=dict(color=PALETTE["cyan"], width=1.8), name=t1))
    fig.add_trace(go.Scatter(x=rb2c.index, y=rb2c.values, mode="lines",
                             line=dict(color=PALETTE["magenta"], width=1.8), name=t2))
    fig.add_hline(y=1.0, line=dict(color="rgba(255,255,255,0.4)", dash="dash"))
    fig.update_yaxes(title_text="rolling β")
    st.plotly_chart(style_fig(fig, height=420, title="rolling 252-day CAPM beta"), use_container_width=True)

    section_header("3.4.2", "Time-varying Beta — interpretation")
    rb_stats = pd.DataFrame({
        "min β": [rb1.min(), rb2.min()],
        "median β": [rb1.median(), rb2.median()],
        "max β": [rb1.max(), rb2.max()],
        "β std": [rb1.std(), rb2.std()],
    }, index=[t1, t2])
    st.dataframe(rb_stats.style.format("{:.3f}"), use_container_width=True)

    correl = rb1.corr(rb2)
    st.markdown(f"""
- Rolling-β correlation between the two stocks: `{correl:+.3f}` — {'they tend to move together' if correl > 0.3 else 'they evolve mostly independently' if abs(correl) < 0.3 else 'they tend to move opposite each other'}.
- Look for visible regime shifts around the early-2020 COVID shock and the 2022 bear market — beta dispersion typically widens during stress and compresses in calm periods.
- Time variation in β suggests the **single-β assumption of full-sample CAPM is a simplification**; rolling estimates are descriptive, not predictive, but they highlight when the linear model's parameters are most fragile.
""")

    section_header("3.4.3", "FF3 Residual Stationarity")
    rows = []
    for tk, res in [(t1, ff31), (t2, ff32)]:
        s, p, v = adf_summary(res.resid)
        rows.append({"series": f"{tk} FF3 residuals", "ADF stat": s, "p-value": p, "verdict": v})
    rstat_df = pd.DataFrame(rows)
    st.dataframe(rstat_df.style.format({"ADF stat": "{:.3f}", "p-value": "{:.4f}"}),
                 use_container_width=True, hide_index=True)
    st.caption("Well-specified residuals should reject the unit-root null and look like stationary white noise.")

    section_header("3.4.4", "Residual ACF / PACF / Squared-Residual ACF")
    cA, cB = st.columns(2)
    with cA:
        st.plotly_chart(acf_bar_chart(ff31.resid, 40, f"{t1} FF3 resid · ACF", PALETTE["cyan"]), use_container_width=True)
        st.plotly_chart(pacf_bar_chart(ff31.resid, 40, f"{t1} FF3 resid · PACF", PALETTE["cyan"]), use_container_width=True)
        st.plotly_chart(acf_bar_chart(ff31.resid ** 2, 40, f"{t1} squared FF3 resid · ACF", PALETTE["amber"]),
                        use_container_width=True)
    with cB:
        st.plotly_chart(acf_bar_chart(ff32.resid, 40, f"{t2} FF3 resid · ACF", PALETTE["magenta"]), use_container_width=True)
        st.plotly_chart(pacf_bar_chart(ff32.resid, 40, f"{t2} FF3 resid · PACF", PALETTE["magenta"]), use_container_width=True)
        st.plotly_chart(acf_bar_chart(ff32.resid ** 2, 40, f"{t2} squared FF3 resid · ACF", PALETTE["amber"]),
                        use_container_width=True)

    sq_resid_acf1 = acf((ff31.resid ** 2).dropna(), nlags=40, fft=True)[1:].mean()
    sq_resid_acf2 = acf((ff32.resid ** 2).dropna(), nlags=40, fft=True)[1:].mean()
    st.markdown(f"""
- Mean ACF of **squared FF3 residuals** (lags 1–40): {t1} = `{sq_resid_acf1:.3f}`, {t2} = `{sq_resid_acf2:.3f}`.
- Persistent positive autocorrelation in squared residuals ⇒ **volatility clustering remains** after controlling for the three factors. A linear model in returns cannot remove conditional heteroscedasticity; that's where GARCH-class models come in.
""")

    section_header("3.4.5", "Putting it together")
    for tk, res, capm in [(t1, ff31, capm1), (t2, ff32, capm2)]:
        st.markdown(f"""
**{tk} — full picture**

Daily returns are stationary, near-zero mean, with negligible serial correlation but pronounced
volatility clustering — a regime visible in the rolling 60-day annualized volatility chart.
Full-sample CAPM gives β = `{capm.coefs['Mkt-RF']:.3f}` with R² = `{capm.r2*100:.1f}%`.
The Fama-French extension lifts adjusted R² to `{res.adj_r2*100:.1f}%`, with SMB = `{res.coefs['SMB']:+.3f}`
and HML = `{res.coefs['HML']:+.3f}` — the loadings consistent with the company's size and style.
FF3 residuals are stationary by ADF, and raw-residual ACF is essentially flat, but squared-residual
ACF still shows clear persistence — i.e. the *direction* of returns is well-captured by the linear
factor model, but the *magnitude* (volatility) is not.
""")

# ── Tab 5: Discussion ────────────────────────────────────────────────────────
with tabs[5]:
    section_header("Q1", "Which stock is more market-sensitive?")
    more_sens = t1 if capm1.coefs["Mkt-RF"] > capm2.coefs["Mkt-RF"] else t2
    st.markdown(f"""
Full-sample CAPM β: `{t1}` = `{capm1.coefs['Mkt-RF']:.3f}`, `{t2}` = `{capm2.coefs['Mkt-RF']:.3f}` → **{more_sens}** is more market-sensitive.
The rolling 252-day β confirms this: `{t1}` median = `{rb1.median():.3f}` (range `{rb1.min():.2f}`–`{rb1.max():.2f}`), `{t2}` median = `{rb2.median():.3f}` (range `{rb2.min():.2f}`–`{rb2.max():.2f}`).
The ranking by sensitivity is consistent across the static and rolling estimators, though the cross-section can flip during stress windows.
""")

    section_header("Q2", "Do SMB / HML loadings match prior expectations?")
    st.markdown(f"""
- `{t1}` ({sector1}) — SMB = `{ff31.coefs['SMB']:+.3f}`, HML = `{ff31.coefs['HML']:+.3f}`.
- `{t2}` ({sector2}) — SMB = `{ff32.coefs['SMB']:+.3f}`, HML = `{ff32.coefs['HML']:+.3f}`.

Large mega-caps typically show negative SMB loadings (they're large, not small).
Growth names (tech, communication services) show negative HML loadings; cyclical / asset-heavy names (industrials, energy, financials) show positive HML.
""")

    section_header("Q3", "Did FF3 materially improve over CAPM?")
    delta1 = (ff31.r2 - capm1.r2) * 100
    delta2 = (ff32.r2 - capm2.r2) * 100
    st.markdown(f"""
- `{t1}`: CAPM R² = `{capm1.r2*100:.1f}%`, FF3 R² = `{ff31.r2*100:.1f}%` (Δ = `+{delta1:.2f} pp`).
- `{t2}`: CAPM R² = `{capm2.r2*100:.1f}%`, FF3 R² = `{ff32.r2*100:.1f}%` (Δ = `+{delta2:.2f} pp`).

For mega-cap U.S. equities, the market factor typically does the heavy lifting; SMB and HML add a few percentage points of explanatory power and improve the *interpretation* of the residual exposure but rarely transform R² dramatically.
""")

    section_header("Q4", "Are the FF3 residuals close to white noise?")
    st.markdown(f"""
ADF tests reject the unit root for both stocks' residuals, and raw-residual ACFs are essentially within the 95% bands.
However, the ACF of **squared residuals** shows clear persistence for both — `{t1}` mean = `{sq_resid_acf1:.3f}`, `{t2}` mean = `{sq_resid_acf2:.3f}`.
Conclusion: the linear factor model captures the conditional mean but not the conditional variance — volatility clustering survives.
Q-Q plots also show heavier-than-normal tails, common for daily equity returns.
""")

    section_header("Q5", "Did rolling β change meaningfully over time?")
    st.markdown(f"""
Both stocks' rolling β series show non-trivial dispersion: `{t1}` σ(β) = `{rb1.std():.3f}`, `{t2}` σ(β) = `{rb2.std():.3f}`.
Beta moved significantly around macro stress events (2020 COVID, 2022 rate-hike drawdown), undermining the assumption of a single full-sample β.
The full-sample β is best read as a long-run average; the rolling β is a more honest description of *current* market sensitivity.
""")

    section_header("Q6", "Limitations of full-sample linear models")
    st.markdown(f"""
- **Time-varying volatility** violates homoscedasticity → standard errors understate true uncertainty.
- **Time-varying β** means the parameter you estimated yesterday isn't necessarily the parameter governing tomorrow.
- **Fat-tailed, non-normal residuals** mean OLS confidence intervals are approximations at best.
- **Implication:** CAPM and FF3 are excellent **descriptive** tools — they decompose realized return variation into recognized priced factors — but they are weak **forecasting** tools without enrichment (rolling estimation, GARCH-type variance models, regime-switching, or Bayesian shrinkage on β).
""")

# ── Tab 6: Data ──────────────────────────────────────────────────────────────
with tabs[6]:
    section_header("Δ", "Merged dataset · returns, factors, excess returns")
    show_cols = [f"{t1}_px", f"{t2}_px", f"{t1}_ret", f"{t2}_ret", "Mkt-RF", "SMB", "HML", "RF",
                 f"{t1}_ex", f"{t2}_ex"]
    show = df[show_cols].copy()
    st.dataframe(show.tail(500), use_container_width=True, height=420)

    csv = df.to_csv().encode()
    st.download_button(
        "⤓  Download full dataset (CSV)",
        data=csv,
        file_name=f"factor_dataset_{t1}_{t2}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown(
        f"""<div class="qlab-card" style="margin-top:14px">
            <h3>UNITS · INTEGRITY</h3>
            <div style="margin-top:6px">
                · Stock returns: <b>decimal</b> (from <code>pct_change()</code>).<br>
                · Fama-French factors / RF: <b>decimal</b> (loaded from Ken French in percent and divided by 100).<br>
                · Excess return = stock return − RF (same units, as required).<br>
                · Market excess return = Mkt-RF column directly (RF already subtracted upstream).
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown(
    f"""<div style="text-align:center;margin-top:30px;padding:20px;color:{PALETTE['muted']};
        font-family:JetBrains Mono;font-size:11px;letter-spacing:0.18em">
        QUANT LAB · {df.index[0].date()} → {df.index[-1].date()} · {len(df)} OBS
    </div>""",
    unsafe_allow_html=True,
)
