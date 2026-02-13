"""
Mean Reversion Strategy for QRT Academy Competition
=====================================================
Identifies stocks that have deviated significantly from recent norms,
betting they revert to the mean. Goes long oversold, short overbought.

Since QRT auto-hedges beta, this captures purely idiosyncratic reversion.

Signals:
  1. Price z-score vs rolling mean (primary)
  2. RSI extremes (confirmation)
  3. Volume-weighted deviation (conviction scaling)

Usage:
    from mean_reversion import generate_mean_reversion_targets
    targets = generate_mean_reversion_targets(region="AMER")
"""

import pandas as pd
import numpy as np
import eikon as ek

# ── Configuration ──────────────────────────────────────────────────────────────

# LSEG Eikon app key — set yours here or via ek.set_app_key() before calling
# ek.set_app_key("YOUR_APP_KEY")

# Universe indices
UNIVERSE = {
    "AMER": "0#.RUA",   # Russell 3000
    "EMEA": "0#.STOXX",  # STOXX Europe 600
}

CURRENCY = {
    "AMER": "USD",
    "EMEA": "EUR",
}

# Strategy parameters
LOOKBACK_SHORT = 20    # Short-term MA window (days)
LOOKBACK_LONG = 60     # Long-term MA window (days)
ZSCORE_ENTRY = 1.5     # Z-score threshold to enter a position
ZSCORE_MAX = 4.0       # Cap z-score to avoid distressed/broken stocks
RSI_PERIOD = 14        # RSI lookback
RSI_OVERSOLD = 30      # RSI confirmation threshold (long)
RSI_OVERBOUGHT = 70    # RSI confirmation threshold (short)
MIN_ADV_USD = 5_000_000  # Minimum average daily volume in USD

# Position sizing
MAX_POSITION_USD = 2_000_000       # QRT hard limit per stock
MAX_PCT_ADV = 0.025                # 2.5% of 60-day ADV
TARGET_GROSS_EXPOSURE = 20_000_000 # Target GMV (adjust based on risk budget)
MAX_SINGLE_POSITION_PCT = 0.05     # Max 5% of gross in one name


# ── Data Fetching ─────────────────────────────────────────────────────────────

def fetch_universe(region: str) -> list[str]:
    """Fetch constituent RICs for the given region's index."""
    chain_ric = UNIVERSE[region]
    constituents = ek.get_data(chain_ric, ["TR.RIC"])[0]
    rics = constituents["Instrument"].dropna().unique().tolist()
    print(f"[{region}] Fetched {len(rics)} constituents")
    return rics


def fetch_price_data(rics: list[str], lookback_days: int = 90) -> pd.DataFrame:
    """
    Fetch daily OHLCV data for a list of RICs.
    Returns a DataFrame with MultiIndex (date, ric).
    """
    # Eikon has a limit on how many RICs per call, so we batch
    batch_size = 50
    all_data = []

    for i in range(0, len(rics), batch_size):
        batch = rics[i : i + batch_size]
        try:
            df = ek.get_timeseries(
                batch,
                fields=["CLOSE", "HIGH", "LOW", "VOLUME"],
                start_date=pd.Timestamp.now() - pd.Timedelta(days=lookback_days + 30),
                end_date=pd.Timestamp.now(),
                interval="daily",
            )
            if df is not None and not df.empty:
                all_data.append(df)
        except Exception as e:
            print(f"  Warning: batch {i}-{i+len(batch)} failed: {e}")
            continue

    if not all_data:
        raise ValueError("No price data fetched — check your Eikon connection/app key")

    prices = pd.concat(all_data)
    print(f"  Fetched price data: {prices.shape[0]} rows across {prices.columns.get_level_values(0).nunique() if isinstance(prices.columns, pd.MultiIndex) else 'N/A'} instruments")
    return prices


def fetch_adv(rics: list[str]) -> pd.Series:
    """Fetch 60-day average daily volume in USD for filtering and sizing."""
    batch_size = 100
    results = []

    for i in range(0, len(rics), batch_size):
        batch = rics[i : i + batch_size]
        try:
            df, _ = ek.get_data(
                batch,
                [
                    "TR.Volume.Average(Days=60)",
                    "TR.PriceClose",
                    "CF_CURR",
                ]
            )
            if df is not None:
                results.append(df)
        except Exception as e:
            print(f"  Warning: ADV batch {i} failed: {e}")
            continue

    if not results:
        return pd.Series(dtype=float)

    adv_df = pd.concat(results, ignore_index=True)
    adv_df.columns = ["ric", "avg_volume", "price", "currency"]
    adv_df["avg_volume"] = pd.to_numeric(adv_df["avg_volume"], errors="coerce")
    adv_df["price"] = pd.to_numeric(adv_df["price"], errors="coerce")
    adv_df["adv_usd"] = adv_df["avg_volume"] * adv_df["price"]
    adv_df = adv_df.set_index("ric")

    return adv_df["adv_usd"]


# ── Signal Computation ────────────────────────────────────────────────────────

def compute_zscore(close_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute z-score of current price vs rolling mean.
    close_prices: DataFrame with columns = RICs, index = dates.
    Returns: DataFrame of z-scores (same shape).
    """
    ma_short = close_prices.rolling(window=LOOKBACK_SHORT, min_periods=15).mean()
    std_short = close_prices.rolling(window=LOOKBACK_SHORT, min_periods=15).std()

    # Z-score: how many std devs is current price from the short-term mean
    zscore = (close_prices - ma_short) / std_short

    # Also compute vs long-term mean for trend filter
    ma_long = close_prices.rolling(window=LOOKBACK_LONG, min_periods=40).mean()

    # Only take mean reversion trades where price is still "near" long-term trend
    # (avoids catching falling knives in structural declines)
    long_term_deviation = (close_prices - ma_long) / ma_long
    trend_filter = long_term_deviation.abs() < 0.20  # Within 20% of 60-day MA

    zscore = zscore.where(trend_filter, 0)

    return zscore


def compute_rsi(close_prices: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    """Compute RSI for each stock."""
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()

    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_volume_ratio(volume: pd.DataFrame) -> pd.DataFrame:
    """
    Ratio of recent volume to average — high volume on a deviation
    means more conviction in the reversion signal.
    """
    avg_vol = volume.rolling(window=20, min_periods=15).mean()
    vol_ratio = volume / avg_vol
    return vol_ratio


def generate_signals(close_prices: pd.DataFrame, volume: pd.DataFrame) -> pd.Series:
    """
    Combine z-score, RSI, and volume into a composite mean reversion signal.

    Returns a Series indexed by RIC with signal values:
      - Negative signal = oversold → go LONG
      - Positive signal = overbought → go SHORT
    We flip the sign at the end so positive = long, negative = short.
    """
    zscore = compute_zscore(close_prices)
    rsi = compute_rsi(close_prices)
    vol_ratio = compute_volume_ratio(volume)

    # Get latest values
    z_latest = zscore.iloc[-1]
    rsi_latest = rsi.iloc[-1]
    vol_latest = vol_ratio.iloc[-1].clip(0.5, 3.0)  # Cap volume multiplier

    # ── Primary signal: z-score ──
    # Only trade when z-score exceeds threshold
    signal = z_latest.copy()
    signal[signal.abs() < ZSCORE_ENTRY] = 0
    signal = signal.clip(-ZSCORE_MAX, ZSCORE_MAX)

    # ── RSI confirmation: boost signal when RSI agrees ──
    rsi_boost = pd.Series(1.0, index=signal.index)
    rsi_boost[rsi_latest < RSI_OVERSOLD] = 1.3    # RSI confirms oversold
    rsi_boost[rsi_latest > RSI_OVERBOUGHT] = 1.3  # RSI confirms overbought

    # Penalise if RSI disagrees (z-score says oversold but RSI says overbought)
    rsi_disagree_long = (z_latest < -ZSCORE_ENTRY) & (rsi_latest > RSI_OVERBOUGHT)
    rsi_disagree_short = (z_latest > ZSCORE_ENTRY) & (rsi_latest < RSI_OVERSOLD)
    rsi_boost[rsi_disagree_long | rsi_disagree_short] = 0.5

    # ── Volume conviction: scale by volume ratio ──
    # Higher volume on deviation = more conviction
    vol_multiplier = vol_latest ** 0.3  # Damped, so 2x volume ≈ 1.23x signal

    # ── Composite signal ──
    composite = signal * rsi_boost * vol_multiplier

    # Flip sign: negative z-score (oversold) → positive signal (long)
    composite = -composite

    # Drop zeros and NaN
    composite = composite.dropna()
    composite = composite[composite != 0]

    print(f"  Signals generated: {len(composite)} active "
          f"({(composite > 0).sum()} long, {(composite < 0).sum()} short)")

    return composite


# ── Position Sizing ───────────────────────────────────────────────────────────

def size_positions(
    signals: pd.Series,
    adv_usd: pd.Series,
    target_gross: float = TARGET_GROSS_EXPOSURE,
) -> pd.Series:
    """
    Convert raw signals into dollar notional positions.
    Enforces all QRT competition constraints.
    """
    # Align signals with ADV data (drop stocks we don't have ADV for)
    common = signals.index.intersection(adv_usd.index)
    signals = signals.loc[common]
    adv = adv_usd.loc[common]

    # Filter out low-liquidity stocks
    liquid = adv >= MIN_ADV_USD
    signals = signals[liquid]
    adv = adv[liquid]

    if signals.empty:
        return pd.Series(dtype=float)

    # Normalize signals to get portfolio weights
    total_abs_signal = signals.abs().sum()
    weights = signals / total_abs_signal

    # Scale to target gross exposure
    positions = weights * target_gross

    # ── Enforce constraints ──

    # 1. Max position size: $2M
    positions = positions.clip(-MAX_POSITION_USD, MAX_POSITION_USD)

    # 2. Max 2.5% of 60-day ADV
    adv_limit = adv * MAX_PCT_ADV
    positions = positions.clip(-adv_limit, adv_limit)

    # 3. Max single position as % of gross
    gross = positions.abs().sum()
    if gross > 0:
        max_single = gross * MAX_SINGLE_POSITION_PCT
        positions = positions.clip(-max_single, max_single)

    # 4. Re-balance to maintain approximate dollar neutrality
    net = positions.sum()
    if positions.abs().sum() > 0:
        # Distribute net imbalance proportionally
        adjustment = net * positions.abs() / positions.abs().sum()
        positions = positions - adjustment

    # Round to nearest $1000
    positions = (positions / 1000).round() * 1000

    # Drop tiny positions
    positions = positions[positions.abs() >= 10_000]

    print(f"  Sized positions: {len(positions)} stocks, "
          f"gross=${positions.abs().sum():,.0f}, net=${positions.sum():,.0f}")

    return positions


# ── Main Entry Point ──────────────────────────────────────────────────────────

def generate_mean_reversion_targets(
    region: str = "AMER",
    app_key: str = None,
) -> pd.DataFrame:
    """
    Full pipeline: fetch data → compute signals → size positions → return targets.

    Parameters:
        region: "AMER" or "EMEA"
        app_key: LSEG Eikon app key (if not already set)

    Returns:
        DataFrame with columns: internal_code, currency, target_notional
    """
    if app_key:
        ek.set_app_key(app_key)

    print(f"\n{'='*60}")
    print(f"  MEAN REVERSION STRATEGY — {region}")
    print(f"{'='*60}")

    # 1. Get universe
    print("\n[1/4] Fetching universe...")
    rics = fetch_universe(region)

    # 2. Fetch price data
    print("\n[2/4] Fetching price data...")
    price_data = fetch_price_data(rics, lookback_days=LOOKBACK_LONG + 30)

    # Reshape — eikon returns different formats depending on 1 vs many RICs
    if isinstance(price_data.columns, pd.MultiIndex):
        close_prices = price_data.xs("CLOSE", axis=1, level=1)
        volume = price_data.xs("VOLUME", axis=1, level=1)
    else:
        close_prices = price_data[["CLOSE"]]
        volume = price_data[["VOLUME"]]

    # 3. Fetch ADV for sizing constraints
    print("\n[3/4] Fetching ADV data...")
    adv = fetch_adv(rics)

    # 4. Generate signals and size
    print("\n[4/4] Computing signals and sizing positions...")
    signals = generate_signals(close_prices, volume)
    positions = size_positions(signals, adv)

    if positions.empty:
        print("  ⚠ No positions generated — check data or loosen thresholds")
        return pd.DataFrame(columns=["internal_code", "currency", "target_notional"])

    # Format as QRT targets
    targets = pd.DataFrame({
        "internal_code": positions.index,
        "currency": CURRENCY[region],
        "target_notional": positions.values,
    })

    print(f"\n  ✓ Generated {len(targets)} targets for {region}")
    print(f"    Long:  {(targets['target_notional'] > 0).sum()} positions, "
          f"${targets.loc[targets['target_notional'] > 0, 'target_notional'].sum():,.0f}")
    print(f"    Short: {(targets['target_notional'] < 0).sum()} positions, "
          f"${targets.loc[targets['target_notional'] < 0, 'target_notional'].sum():,.0f}")

    return targets


# ── Standalone execution for testing ──────────────────────────────────────────

if __name__ == "__main__":
    # Quick test — run for AMER only
    # Make sure Eikon is running and app key is set
    # ek.set_app_key("YOUR_APP_KEY_HERE")

    targets = generate_mean_reversion_targets("AMER")
    print("\nSample targets:")
    print(targets.head(10))
