"""
QRT Academy Backtester
=======================
Simulates the QRT competition execution engine locally, replicating the
detailed report format (QSec_Detailed_*.xlsx).

Simulates:
  - Daily target generation from strategy
  - Trade execution with 2bps costs
  - Daily mark-to-market P&L
  - Financing costs (0.5% annualized on GMV)
  - Dividend P&L (70% on longs, pay 100% on shorts)
  - Automatic beta hedging against SPX (AMER) / EuroStoxx50 (EMEA)
  - Position size constraints (max $2M, 2.5% ADV)

Output matches QRT's QSec_Detailed report format.

Usage:
    from backtester import Backtester
    bt = Backtester(region="AMER")
    bt.run(start_date="2025-11-01", end_date="2026-02-07")
    bt.export_report("backtest_results.xlsx")
"""

import pandas as pd
import numpy as np
from typing import Callable
from pathlib import Path

# ── Cost & Competition Parameters ─────────────────────────────────────────────

EXEC_COST_BPS = 2          # 2bps execution cost
FINANCING_RATE = 0.005     # 0.5% annualized on GMV
DIV_TAX_LONG = 0.70        # Receive 70% of dividends on longs
DIV_TAX_SHORT = 1.00       # Pay 100% of dividends on shorts
MAX_POSITION_USD = 2_000_000
MAX_PCT_ADV = 0.025
TRADING_DAYS_PER_YEAR = 252

HEDGE_INDEX = {
    "AMER": ".SPX",
    "EMEA": ".STOXX50E",
}


class Backtester:
    """
    Simulates QRT competition execution and produces detailed P&L reports.

    Parameters:
        region: "AMER" or "EMEA"
        signal_func: Callable that takes (date, price_history, volume_history)
                     and returns a pd.Series of target notional positions
                     indexed by RIC.
        prices: DataFrame of daily close prices, columns = RICs, index = dates
        volumes: DataFrame of daily volumes, same shape as prices
        betas: Series of stock betas indexed by RIC (or DataFrame by date)
        adv: Series of 60-day ADV in USD indexed by RIC
        dividends: Optional DataFrame of dividend amounts (date x RIC)
    """

    def __init__(
        self,
        region: str,
        signal_func: Callable,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        betas: pd.Series,
        adv: pd.Series,
        dividends: pd.DataFrame = None,
        forex: pd.Series = None,
    ):
        self.region = region
        self.signal_func = signal_func
        self.prices = prices.sort_index()
        self.volumes = volumes.sort_index()
        self.betas = betas
        self.adv = adv
        self.dividends = dividends
        self.forex = forex  # FX rates to USD if EMEA
        self.hedge_index = HEDGE_INDEX[region]

        # State
        self.positions = pd.Series(dtype=float)  # Current positions in USD notional
        self.daily_records = []      # List of daily detail rows
        self.summary_records = []    # Daily summary rows

    def run(self, start_date: str, end_date: str):
        """Run the backtest over the specified date range."""
        dates = self.prices.loc[start_date:end_date].index
        print(f"\n{'='*60}")
        print(f"  BACKTESTING {self.region}: {dates[0].date()} → {dates[-1].date()}")
        print(f"  {len(dates)} trading days, {self.prices.shape[1]} instruments")
        print(f"{'='*60}\n")

        for i, date in enumerate(dates):
            # Need at least some history for signals
            if i < 20:
                continue

            # Get price history up to today (strategy can only see this)
            hist_end = i + 1
            price_hist = self.prices.iloc[:hist_end]
            vol_hist = self.volumes.iloc[:hist_end]

            # ── 1. Generate target positions from strategy ────────────────
            try:
                targets = self.signal_func(date, price_hist, vol_hist)
            except Exception as e:
                print(f"  {date.date()}: Signal error: {e}")
                targets = pd.Series(dtype=float)

            # ── 2. Simulate one day ───────────────────────────────────────
            self._simulate_day(date, targets, price_hist)

            if (i + 1) % 20 == 0:
                cum_pnl = sum(r.get("Net PnL USD", 0) for r in self.daily_records)
                print(f"  {date.date()}: Day {i+1}/{len(dates)}, "
                      f"positions={len(self.positions[self.positions != 0])}, "
                      f"cumulative PnL=${cum_pnl:,.0f}")

        print(f"\n  ✓ Backtest complete: {len(self.daily_records)} position-day records")

    def _simulate_day(self, date, targets: pd.Series, price_hist: pd.DataFrame):
        """Simulate a single trading day."""
        today_prices = self.prices.loc[date]
        prev_date_idx = self.prices.index.get_loc(date) - 1
        if prev_date_idx < 0:
            return
        prev_prices = self.prices.iloc[prev_date_idx]

        # Get all relevant RICs (union of current positions and targets)
        all_rics = set(self.positions.index) | set(targets.index)
        all_rics = [r for r in all_rics if r in today_prices.index and r in prev_prices.index]

        if not all_rics:
            return

        # Align everything
        targets = targets.reindex(all_rics, fill_value=0)
        current = self.positions.reindex(all_rics, fill_value=0)

        # ── Calculate trades ──────────────────────────────────────────────
        trades = targets - current  # Notional to trade

        # Enforce daily trading limit: 2.5% of ADV
        for ric in all_rics:
            if ric in self.adv.index:
                max_trade = self.adv[ric] * MAX_PCT_ADV
                trades[ric] = np.clip(trades[ric], -max_trade, max_trade)

        # Actual new positions after constrained trading
        new_positions = current + trades

        # ── Calculate P&L components for each position ────────────────────
        day_records = []
        total_beta_exposure = 0.0

        for ric in all_rics:
            if current[ric] == 0 and trades[ric] == 0:
                continue

            sod_price = prev_prices.get(ric, np.nan)
            eod_price = today_prices.get(ric, np.nan)

            if pd.isna(sod_price) or pd.isna(eod_price) or sod_price == 0:
                continue

            pos_sod = current[ric]
            pos_eod = new_positions[ric]
            traded_usd = abs(trades[ric])

            # Day P&L: mark-to-market on position held through the day
            # Approximate: position * (eod_price / sod_price - 1)
            price_return = (eod_price - sod_price) / sod_price
            day_pnl = pos_sod * price_return

            # Exec costs: 2bps on traded notional
            exec_costs = -traded_usd * EXEC_COST_BPS / 10_000

            # Financing costs: 0.5% annualized on abs(position SOD)
            financing = -abs(pos_sod) * FINANCING_RATE / TRADING_DAYS_PER_YEAR

            # Dividend P&L (if dividend data available)
            div_pnl = 0.0
            if self.dividends is not None and ric in self.dividends.columns:
                div_amount = self.dividends.loc[date, ric] if date in self.dividends.index else 0
                if div_amount > 0:
                    if pos_sod > 0:
                        div_pnl = div_amount * (pos_sod / sod_price) * DIV_TAX_LONG
                    elif pos_sod < 0:
                        div_pnl = -div_amount * abs(pos_sod / sod_price) * DIV_TAX_SHORT

            net_pnl = day_pnl + div_pnl + exec_costs + financing

            # Beta exposure for hedge calculation
            beta = self.betas.get(ric, 1.0) if isinstance(self.betas, pd.Series) else 1.0
            total_beta_exposure += pos_sod * beta

            # FX rate
            fx = 1.0
            if self.forex is not None and ric in self.forex.index:
                fx = self.forex[ric]

            # Determine buy/sell amounts
            if trades[ric] > 0:
                amt_buy = trades[ric]
                amt_sell = 0.0
                avg_buy = eod_price  # Approximate: trade at EOD
                avg_sell = np.nan
            elif trades[ric] < 0:
                amt_buy = 0.0
                amt_sell = abs(trades[ric])
                avg_buy = np.nan
                avg_sell = eod_price
            else:
                amt_buy = amt_sell = 0.0
                avg_buy = avg_sell = np.nan

            shares_eod = pos_eod / eod_price if eod_price != 0 else 0

            record = {
                "Instrument": ric,
                "Asset Type": "Equity",
                "TradeDate": date,
                "Traded USD": traded_usd,
                "Avg Buy Price": avg_buy,
                "Avg Sell Price": avg_sell,
                "Amount Buy USD": amt_buy,
                "Amount Sell USD": amt_sell,
                "Position SOD USD": pos_sod,
                "Position EOD USD": pos_eod,
                "Position EOD Shares": shares_eod,
                "SOD Price": sod_price,
                "EOD Price": eod_price,
                "ADV USD": self.adv.get(ric, np.nan),
                "Beta": beta,
                "Forex": fx,
                "Currency": "USD" if self.region == "AMER" else self._get_currency(ric),
                "Target USD": targets.get(ric, 0),
                "In Universe": True,
                "Yest PnL USD": 0.0,  # Simplified — would need prior day data
                "Day PnL USD": day_pnl,
                "Div PnL USD": div_pnl,
                "Exec costs USD": exec_costs,
                "Financing costs USD": financing,
                "Net PnL USD": net_pnl,
                "Book ID": f"CAM18_{self.region}",
            }
            day_records.append(record)

        # ── Beta hedge (SPX / EuroStoxx50) ────────────────────────────────
        if total_beta_exposure != 0 and self.hedge_index in today_prices.index:
            hedge_notional = -total_beta_exposure  # Opposite sign to neutralise
            hedge_sod = prev_prices.get(self.hedge_index, np.nan)
            hedge_eod = today_prices.get(self.hedge_index, np.nan)

            if not pd.isna(hedge_sod) and not pd.isna(hedge_eod) and hedge_sod != 0:
                hedge_return = (hedge_eod - hedge_sod) / hedge_sod
                hedge_pnl = hedge_notional * hedge_return

                day_records.append({
                    "Instrument": self.hedge_index,
                    "Asset Type": "Equity",
                    "TradeDate": date,
                    "Traded USD": abs(hedge_notional),
                    "Avg Buy Price": hedge_eod if hedge_notional > 0 else np.nan,
                    "Avg Sell Price": hedge_eod if hedge_notional < 0 else np.nan,
                    "Amount Buy USD": max(hedge_notional, 0),
                    "Amount Sell USD": max(-hedge_notional, 0),
                    "Position SOD USD": 0,  # Hedge is recalculated daily
                    "Position EOD USD": 0,
                    "Position EOD Shares": np.nan,
                    "SOD Price": hedge_sod,
                    "EOD Price": hedge_eod,
                    "ADV USD": np.nan,
                    "Beta": np.nan,
                    "Forex": 1.0,
                    "Currency": "USD",
                    "Target USD": 0,
                    "In Universe": True,
                    "Yest PnL USD": 0.0,
                    "Day PnL USD": hedge_pnl,
                    "Div PnL USD": 0.0,
                    "Exec costs USD": 0.0,  # Hedge trades are free
                    "Financing costs USD": 0.0,
                    "Net PnL USD": hedge_pnl,
                    "Book ID": f"CAM18_{self.region}",
                })

        # ── Update state ──────────────────────────────────────────────────
        self.positions = new_positions[new_positions != 0]
        self.daily_records.extend(day_records)

        # Summary record
        day_df = pd.DataFrame(day_records)
        if not day_df.empty:
            self.summary_records.append({
                "TradeDate": date,
                "Book ID": f"CAM18_{self.region}",
                "PnL (k$)": day_df["Net PnL USD"].sum() / 1000,
                "GMV (M$)": day_df["Position EOD USD"].abs().sum() / 1_000_000,
                "Risk (k$)": np.nan,  # Would need proper risk model
            })

    def _get_currency(self, ric: str) -> str:
        """Infer currency from RIC suffix."""
        suffix_map = {
            ".L": "GBp", ".PA": "EUR", ".DE": "EUR", ".AS": "EUR",
            ".MI": "EUR", ".MC": "EUR", ".OL": "NOK", ".ST": "SEK",
            ".HE": "EUR", ".BR": "EUR", ".LS": "EUR", ".VI": "EUR",
            ".S": "CHF", ".CO": "DKK",
        }
        for suffix, ccy in suffix_map.items():
            if ric.endswith(suffix):
                return ccy
        return "EUR"

    # ── Reporting ─────────────────────────────────────────────────────────────

    def get_daily_pnl(self) -> pd.DataFrame:
        """Get daily P&L summary as a DataFrame."""
        if not self.summary_records:
            return pd.DataFrame()
        df = pd.DataFrame(self.summary_records)
        df = df.set_index("TradeDate")
        return df

    def get_detailed_records(self) -> pd.DataFrame:
        """Get all position-level records (Sheet1 equivalent)."""
        if not self.daily_records:
            return pd.DataFrame()
        return pd.DataFrame(self.daily_records)

    def get_performance_stats(self) -> dict:
        """Calculate strategy performance statistics."""
        daily = self.get_daily_pnl()
        if daily.empty:
            return {}

        daily_pnl = daily["PnL (k$)"] * 1000  # Back to USD
        total_pnl = daily_pnl.sum()
        avg_daily = daily_pnl.mean()
        std_daily = daily_pnl.std()
        sharpe = (avg_daily / std_daily * np.sqrt(TRADING_DAYS_PER_YEAR)) if std_daily > 0 else 0
        max_dd = (daily_pnl.cumsum() - daily_pnl.cumsum().cummax()).min()
        win_rate = (daily_pnl > 0).mean()

        return {
            "Total PnL": f"${total_pnl:,.0f}",
            "Avg Daily PnL": f"${avg_daily:,.0f}",
            "Daily Std": f"${std_daily:,.0f}",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"${max_dd:,.0f}",
            "Win Rate": f"{win_rate:.1%}",
            "Trading Days": len(daily_pnl),
            "Avg GMV": f"${daily['GMV (M$)'].mean():.2f}M",
        }

    def export_report(self, filepath: str):
        """
        Export backtest results in QRT's detailed report format.
        Creates an Excel file with Sheet1 (position detail) and Sheet2 (summary).
        """
        filepath = Path(filepath)

        detail_df = self.get_detailed_records()
        summary_df = self.get_daily_pnl().reset_index()

        # Match QRT column order for Sheet1
        sheet1_cols = [
            "Instrument", "Asset Type", "TradeDate", "Traded USD",
            "Avg Buy Price", "Avg Sell Price", "Amount Buy USD", "Amount Sell USD",
            "Position SOD USD", "Position EOD USD", "Position EOD Shares",
            "SOD Price", "EOD Price", "ADV USD", "Beta", "Forex", "Currency",
            "Target USD", "In Universe",
            "Yest PnL USD", "Day PnL USD", "Div PnL USD",
            "Exec costs USD", "Financing costs USD", "Net PnL USD", "Book ID",
        ]

        # Only include columns that exist
        sheet1_cols = [c for c in sheet1_cols if c in detail_df.columns]

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            detail_df[sheet1_cols].to_excel(writer, sheet_name="Sheet1", index=True)
            summary_df.to_excel(writer, sheet_name="Sheet2", index=True)

        print(f"\n  ✓ Report exported to {filepath}")

    def print_summary(self):
        """Print performance summary to console."""
        stats = self.get_performance_stats()
        print(f"\n{'='*50}")
        print(f"  BACKTEST RESULTS — {self.region}")
        print(f"{'='*50}")
        for k, v in stats.items():
            print(f"  {k:>20s}: {v}")
        print(f"{'='*50}")


# ── Helper: Quick backtest runner ─────────────────────────────────────────────

def run_backtest_with_eikon(
    region: str = "AMER",
    start_date: str = "2025-11-01",
    end_date: str = "2026-02-07",
    signal_func: Callable = None,
    app_key: str = None,
    export_path: str = None,
) -> Backtester:
    """
    Convenience function: fetches data from Eikon and runs the backtest.

    Parameters:
        region: "AMER" or "EMEA"
        start_date: Backtest start date
        end_date: Backtest end date
        signal_func: Your strategy function — takes (date, prices, volumes)
                     and returns pd.Series of target notionals indexed by RIC
        app_key: Eikon app key
        export_path: Where to save the Excel report (optional)

    Returns:
        Backtester instance with results
    """
    import eikon as ek

    if app_key:
        ek.set_app_key(app_key)

    print(f"Fetching data for {region} backtest...")

    # ── Fetch universe ────────────────────────────────────────────────────
    from mean_reversion import fetch_universe, fetch_adv, fetch_price_data

    rics = fetch_universe(region)

    # ── Fetch historical prices ───────────────────────────────────────────
    # Need extra history before start_date for signal computation
    fetch_start = pd.Timestamp(start_date) - pd.Timedelta(days=120)
    all_data = []
    batch_size = 50

    for i in range(0, len(rics), batch_size):
        batch = rics[i : i + batch_size]
        try:
            df = ek.get_timeseries(
                batch,
                fields=["CLOSE", "VOLUME"],
                start_date=fetch_start.strftime("%Y-%m-%d"),
                end_date=end_date,
                interval="daily",
            )
            if df is not None and not df.empty:
                all_data.append(df)
        except Exception as e:
            print(f"  Batch {i} failed: {e}")

    raw = pd.concat(all_data)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw.xs("CLOSE", axis=1, level=1)
        volumes = raw.xs("VOLUME", axis=1, level=1)
    else:
        prices = raw[["CLOSE"]]
        volumes = raw[["VOLUME"]]

    # ── Fetch betas and ADV ───────────────────────────────────────────────
    adv = fetch_adv(rics)

    # Simple beta estimate from prices (or fetch from Eikon)
    if ".SPX" in prices.columns or ".STOXX50E" in prices.columns:
        idx_col = ".SPX" if region == "AMER" else ".STOXX50E"
        idx_ret = prices[idx_col].pct_change()
        betas = pd.Series(dtype=float)
        for ric in prices.columns:
            if ric == idx_col:
                continue
            stock_ret = prices[ric].pct_change()
            cov = stock_ret.cov(idx_ret)
            var = idx_ret.var()
            betas[ric] = cov / var if var > 0 else 1.0
    else:
        # Default to 1.0 if no index data
        betas = pd.Series(1.0, index=prices.columns)

    # ── Run backtest ──────────────────────────────────────────────────────
    bt = Backtester(
        region=region,
        signal_func=signal_func,
        prices=prices,
        volumes=volumes,
        betas=betas,
        adv=adv,
    )
    bt.run(start_date, end_date)
    bt.print_summary()

    if export_path:
        bt.export_report(export_path)

    return bt


# ── Example usage ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Example: backtest the mean reversion strategy.

    Before running:
      1. Set your Eikon app key
      2. Make sure Eikon Workspace is running
    """
    from mean_reversion import (
        generate_signals, size_positions, compute_zscore, compute_rsi,
        MIN_ADV_USD
    )

    # ek.set_app_key("YOUR_KEY")

    def mean_reversion_signal(date, price_hist, volume_hist):
        """
        Adapter: wraps mean_reversion.py signals into the format
        the backtester expects (pd.Series of target notionals).
        """
        close = price_hist
        vol = volume_hist

        signals = generate_signals(close, vol)
        # For sizing, we need ADV — use a simple proxy from volume data
        recent_vol = vol.iloc[-60:].mean() * close.iloc[-1]
        adv_proxy = recent_vol.dropna()
        adv_proxy = adv_proxy[adv_proxy >= MIN_ADV_USD]

        positions = size_positions(signals, adv_proxy)
        return positions

    # Run it
    bt = run_backtest_with_eikon(
        region="AMER",
        start_date="2025-11-01",
        end_date="2026-02-07",
        signal_func=mean_reversion_signal,
        export_path="backtest_results.xlsx",
    )
