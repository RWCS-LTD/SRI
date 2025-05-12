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
def fetch_btc_data(interval=TF, days=Bars_to_Fetch):
    end = int(datetime.now(timezone.utc).timestamp() * 1000)
    start = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    try:
        candles = info.candles_snapshot(ASSET, interval=interval, startTime=start, endTime=end)
        if not candles:
            st.error(f"No candle data [{interval}]")
            return pd.DataFrame()

        df = pd.DataFrame(candles)
        df.rename(columns={"c": "close", "o": "open", "h": "high", "l": "low", "t": "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df = df[["open", "high", "low", "close"]].astype(float).dropna()
        return df
    except Exception as e:
        st.error(f"Error fetching BTC data: {str(e)}")
        return pd.DataFrame()

# === Statistical Calculations ===
def calculate_percentile(series, length, percentile):
    return series.rolling(window=length).apply(lambda x: np.percentile(x, percentile * 100), raw=True)

def calculate_sma(series, length):
    return series.rolling(window=length).mean()

def calculate_std(series, length):
    return series.rolling(window=length).std()

# === Calculate SRI ===
def calculate_sri(df):
    returns = df['close'].pct_change().dropna()

    # CDF Extrapolation
    cdf_value = calculate_percentile(returns, length, cdf)

    # Bias Factor
    mean_price = calculate_sma(df['close'], length)
    bias_factor = df['close'] / mean_price

    # Coefficient of Variation (CV)
    std_dev = calculate_std(returns, length)
    mean_return = calculate_sma(returns, length)
    cv = std_dev / mean_return

    # Normalization
    normalized_bias = bias_factor / BF
    normalized_cv = cv / CV

    # SRI Calculation
    sri = -((w1 * cdf_value) + (w2 * normalized_bias) + (w3 * (1 - normalized_cv)))

    # SRI Volatility Calculation
    sri_volatility = calculate_std(sri, stability_length)

    # Chop Mode Identification
    is_chop = sri_volatility > chop_threshold
    sri_data = pd.DataFrame({"SRI": sri, "SRI_Volatility": sri_volatility, "Chop": is_chop}, index=df.index)

    return sri_data

# Plot SRI, Volatility, and BTC Price
def plot_sri(btc_df, sri_df):
    plt.figure(figsize=(12, 15))

    # Plot SRI with Dynamic Coloring
    plt.subplot(3, 1, 1)
    for i in range(len(sri_df) - 1):
        color = 'green' if sri_df['SRI'].iloc[i] >= 1 else 'red'
        plt.plot(sri_df.index[i:i+2], sri_df['SRI'].iloc[i:i+2], color=color, linewidth=2)
    plt.axhline(1, color='gray', linestyle='--')
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label='SRI (Above Benchmark)'),
        Line2D([0], [0], color='red', lw=2, label='SRI (Below Benchmark)'),
        Line2D([0], [0], color='gray', linestyle='--', lw=2, label='Benchmark (1)')
    ]
    plt.legend(handles=legend_elements)
    plt.title('Statistical Reliability Index (SRI)')
    plt.ylabel('SRI')

    # Plot SRI Volatility
    plt.subplot(3, 1, 2)
    plt.plot(sri_df.index, sri_df['SRI_Volatility'], label='SRI Volatility', color='purple', linewidth=2)
    plt.axhline(chop_threshold, color='red', linestyle='--', label='Chop Threshold')
    plt.fill_between(
        sri_df.index, sri_df['SRI_Volatility'], 
        where=sri_df['Chop'], color='orange', alpha=0.3, label='Chop Mode'
    )
    plt.title('SRI Volatility')
    plt.ylabel('Volatility')
    plt.legend()

    # Plot BTC Price as a standard line plot (black)
    plt.subplot(3, 1, 3)
    plt.plot(btc_df.index, btc_df['close'], label='Asset Price', color='black', linestyle='--')
    plt.title('Asset Price (1D)')
    plt.ylabel('Price (USD)')
    plt.legend(['Asset Price'])

    # Display the plots in Streamlit
    st.pyplot(plt)

# === Update Chart Button ===
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

# === Main Logic with Update Button ===
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
