import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from datetime import datetime, timedelta, timezone
from hyperliquid.info import Info
from hyperliquid.utils import constants

# Streamlit App Configuration
st.set_page_config(page_title="Statistical Reliability Index (SRI)", layout="wide")

# === Sidebar for User Inputs ===
st.sidebar.title("SRI Configuration")
ASSET = st.sidebar.text_input("Asset", value="BTC")
TF = st.sidebar.selectbox("Time Frame", ["1d", "4h", "1h", "15m"], index=0)
Bars_to_Fetch = st.sidebar.number_input("Bars to Fetch", min_value=50, max_value=1000, value=252)

length = st.sidebar.number_input("SMA Length", min_value=1, max_value=200, value=50)
w1 = st.sidebar.slider("Weight 1 (CDF)", 0.0, 1.0, 0.4)
w2 = st.sidebar.slider("Weight 2 (Bias)", 0.0, 1.0, 0.3)
w3 = st.sidebar.slider("Weight 3 (CV)", 0.0, 1.0, 0.3)
cdf = st.sidebar.slider("CDF", 0.0, 1.0, 0.95)
BF = st.sidebar.number_input("Bias Factor", min_value=0.1, max_value=5.0, value=1.5)
CV = st.sidebar.number_input("Coefficient of Variation", min_value=0.1, max_value=5.0, value=0.5)
stability_length = st.sidebar.number_input("Stability Length", min_value=1, max_value=100, value=14)
chop_threshold = st.sidebar.number_input("Chop Threshold", min_value=1, max_value=100, value=20)

# === Initialize Hyperliquid API ===
info = Info(constants.MAINNET_API_URL)

# === Fetch BTC Data from 1D Chart ===
@st.cache_data
def fetch_btc_data(asset, interval, days):
    end = int(datetime.now(timezone.utc).timestamp() * 1000)
    start = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    try:
        candles = info.candles_snapshot(asset, interval=interval, startTime=start, endTime=end)
        if not candles:
            st.error(f"No candle data [{interval}]")
            return pd.DataFrame()

        df = pd.DataFrame(candles)
        df.rename(columns={"c": "close", "o": "open", "h": "high", "l": "low", "t": "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df[["open", "high", "low", "close"]].astype(float).dropna()
    except Exception as e:
        st.error(f"Error fetching BTC data: {str(e)}")
        return pd.DataFrame()

# === Calculate SRI ===
def calculate_sri(df):
    returns = df['close'].pct_change().dropna()
    cdf_value = returns.rolling(window=length).apply(lambda x: np.percentile(x, cdf * 100), raw=True)
    mean_price = df['close'].rolling(window=length).mean()
    bias_factor = df['close'] / mean_price
    std_dev = returns.rolling(window=length).std()
    mean_return = returns.rolling(window=length).mean()
    cv = std_dev / mean_return
    normalized_bias = bias_factor / BF
    normalized_cv = cv / CV
    sri = -((w1 * cdf_value) + (w2 * normalized_bias) + (w3 * (1 - normalized_cv)))
    sri_volatility = sri.rolling(window=stability_length).std()
    is_chop = sri_volatility > chop_threshold
    return pd.DataFrame({"SRI": sri, "SRI_Volatility": sri_volatility, "Chop": is_chop}, index=df.index)

# === Plot SRI and Price ===
def plot_sri(btc_df, sri_df):
    fig, ax = plt.subplots(3, 1, figsize=(12, 15))

    # Plot SRI
    sri_colors = np.where(sri_df['SRI'] >= 1, 'green', 'red')
    for i in range(len(sri_df) - 1):
        ax[0].plot(sri_df.index[i:i+2], sri_df['SRI'].iloc[i:i+2], color=sri_colors[i], linewidth=2)
    ax[0].axhline(1, color='gray', linestyle='--')
    ax[0].set_title('Statistical Reliability Index (SRI)')
    ax[0].set_ylabel('SRI')

    # SRI Volatility Plot
    ax[1].plot(sri_df.index, sri_df['SRI_Volatility'], label='SRI Volatility', color='purple')
    ax[1].axhline(chop_threshold, color='red', linestyle='--', label='Chop Threshold')
    ax[1].fill_between(sri_df.index, sri_df['SRI_Volatility'], where=sri_df['Chop'], color='orange', alpha=0.3)
    ax[1].set_title('SRI Volatility')
    ax[1].legend()

    # Plot BTC Price
    ax[2].plot(btc_df.index, btc_df['close'], label='Asset Price', color='black')
    ax[2].set_title('Asset Price (1D)')
    ax[2].set_ylabel('Price (USD)')
    ax[2].legend()

    st.pyplot(fig)

# === Main Logic ===
st.title("Statistical Reliability Index (SRI) Dashboard")
if st.sidebar.button("Update Chart"):
    st.write("Fetching and calculating SRI data...")
    btc_df = fetch_btc_data(ASSET, TF, Bars_to_Fetch)
    if not btc_df.empty:
        sri_df = calculate_sri(btc_df)
        st.write("### SRI Data (Last 5 rows)")
        st.dataframe(sri_df.tail())
        plot_sri(btc_df, sri_df)
    else:
        st.warning("No data available to display. Please check your settings.")
