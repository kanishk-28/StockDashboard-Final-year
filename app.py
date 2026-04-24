# ================= IMPORTS =================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import requests

from ta.trend import MACD, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

# ================= CONFIG =================
st.set_page_config(page_title="Stock Dashboard", layout="wide")
st_autorefresh(interval=60000, key="refresh")

TWELVE_API_KEY = "aa26e5be64824cf2827531879044b6c6"

# ================= SESSION STATE =================
if "favorites" not in st.session_state:
    st.session_state.favorites = []

if "selected_fav" not in st.session_state:
    st.session_state.selected_fav = None

# ================= STOCK DATABASE =================
STOCK_MAP = {
    "Reliance": "RELIANCE",
    "TCS": "TCS",
    "Infosys": "INFY",
    "HDFC Bank": "HDFCBANK",
    "ICICI Bank": "ICICIBANK",
    "SBI": "SBIN",
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Microsoft": "MSFT",
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD"
}

NAME_TO_SYMBOL = {k.upper(): v for k, v in STOCK_MAP.items()}
US_TICKERS = {"AAPL","TSLA","MSFT","GOOGL","AMZN","META","NVDA"}

# ================= HELPERS =================
def smart_ticker(symbol):
    try:
        s = symbol.upper().strip()

        if s in NAME_TO_SYMBOL:
            return NAME_TO_SYMBOL[s]

        s_clean = s.replace(" ", "")

        if s_clean in ["NIFTY", "NIFTY50"]:
            return "^NSEI"

        if s_clean in ["BTC", "ETH"]:
            return s_clean + "-USD"

        return s_clean
    except:
        return symbol


def get_yf_symbol(symbol):
    try:
        s = symbol.upper()

        if s.startswith("^") or "-" in s:
            return s

        if s in US_TICKERS:
            return s

        return s + ".NS"
    except:
        return symbol


# ================= REAL-TIME PRICE =================
@st.cache_data(ttl=30)
def fetch_live_price(symbol):
    try:
        s = symbol
        if "-" in s:
            s = s.replace("-", "/")

        url = f"https://api.twelvedata.com/price?symbol={s}&apikey={TWELVE_API_KEY}"
        res = requests.get(url, timeout=5).json()

        if "price" in res:
            return float(res["price"]), None

        return None, "Live price not available"
    except Exception as e:
        return None, str(e)


# ================= DATA FETCH =================
@st.cache_data(ttl=120)
def fetch_data(symbol):
    try:
        yf_symbol = get_yf_symbol(symbol)

        df = yf.download(
            yf_symbol,
            period="3mo",
            interval="1d",
            progress=False,
            threads=False
        )

        if df is None or df.empty:
            df = yf.download(symbol, period="1mo")

        if df is None or df.empty:
            return None, "No historical data found"

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        return df, None

    except Exception as e:
        return None, str(e)


# ================= LOAD MODEL =================
try:
    model = joblib.load("ml_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# ================= UI =================
st.title("📊 Stock Prediction Dashboard")

search = st.sidebar.text_input("🔍 Search Stock")
selected = st.sidebar.selectbox("Select Stock", list(STOCK_MAP.keys()))

# Default ticker
ticker = STOCK_MAP[selected]

# Search override
if search:
    ticker = smart_ticker(search)

# ================= FAVORITES =================
st.sidebar.markdown("### ⭐ Favorites")

# Add to favorites
if st.sidebar.button("➕ Add Current"):
    if ticker not in st.session_state.favorites:
        st.session_state.favorites.append(ticker)
        st.sidebar.success(f"{ticker} added")
    else:
        st.sidebar.warning("Already exists")

# Show favorites
if st.session_state.favorites:
    fav_choice = st.sidebar.selectbox(
        "Your Favorites",
        st.session_state.favorites,
        key="fav_select"
    )

    col1, col2 = st.sidebar.columns(2)

    if col1.button("📌 Use"):
        st.session_state.selected_fav = fav_choice

    if col2.button("❌ Remove"):
        st.session_state.favorites.remove(fav_choice)
        st.session_state.selected_fav = None
        st.rerun()

# Apply favorite override (highest priority)
if st.session_state.selected_fav:
    ticker = st.session_state.selected_fav

st.sidebar.success(f"Using: {ticker}")

# ================= FETCH =================
df, error = fetch_data(ticker)

if error:
    st.warning(f"⚠️ {error}")

if df is None or df.empty or 'Close' not in df.columns:
    st.error(f"❌ Unable to fetch data for {ticker}")
    st.stop()

# ================= PRICE =================
close = df['Close'].dropna()

if close.empty:
    st.error("❌ No valid price data available")
    st.stop()

# ================= LIVE PRICE =================
live_price, live_error = fetch_live_price(ticker)

if live_price:
    price = live_price
    source = "Real-Time (TwelveData)"
else:
    price = float(close.iloc[-1])
    source = "Yahoo Finance"

st.info(f"Data Source: {source}")

# ================= INDICATORS =================
try:
    df['RSI'] = RSIIndicator(close, window=14).rsi()
    df['EMA20'] = EMAIndicator(close).ema_indicator()

    macd = MACD(close)
    df['MACD'] = macd.macd()
    df['MACD_SIGNAL'] = macd.macd_signal()

    bb = BollingerBands(close)
    df['BB_HIGH'] = bb.bollinger_hband()
    df['BB_LOW'] = bb.bollinger_lband()
except Exception as e:
    st.warning(f"⚠️ Indicator issue: {e}")

# ================= ML =================
required_cols = ['RSI','EMA20','MACD','MACD_SIGNAL','BB_HIGH','BB_LOW']
df_ml = df[required_cols].dropna()

if df_ml.empty or len(df_ml) < 5:
    df['Prediction'] = 0
    signal = "NO SIGNAL ⚪"
else:
    try:
        scaled = scaler.transform(df_ml)
        preds = model.predict(scaled)
        df.loc[df_ml.index, 'Prediction'] = preds
        signal = "BUY 🟢" if preds[-1] == 1 else "SELL 🔴"
    except Exception as e:
        df['Prediction'] = 0
        signal = "NO SIGNAL ⚪"

# ================= SAFE VALUES =================
rsi_series = df['RSI'].dropna()
rsi_value = f"{rsi_series.iloc[-1]:.2f}" if not rsi_series.empty else "N/A"

ema_series = df['EMA20'].dropna()
trend = "Uptrend 📈" if not ema_series.empty and price > ema_series.iloc[-1] else "Downtrend 📉"

# ================= TRADE =================
entry = price
stop_loss = price * 0.995
target = price * 1.005

# ================= DASHBOARD =================
st.subheader("📊 Trading Dashboard")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Price", f"{price:.2f}")
c2.metric("Signal", signal)
c3.metric("Model", "Active")
c4.metric("RSI", rsi_value)

st.markdown(f"### Trend: {trend}")

# ================= TRADE SETUP =================
st.subheader("📌 Trade Setup")

t1, t2, t3 = st.columns(3)
t1.metric("Entry", f"{entry:.2f}")
t2.metric("Stop Loss", f"{stop_loss:.2f}")
t3.metric("Target", f"{target:.2f}")

# ================= CHART =================
try:
    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['EMA20'],
        name="EMA20"
    ))

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.warning(f"⚠️ Chart error: {e}")
