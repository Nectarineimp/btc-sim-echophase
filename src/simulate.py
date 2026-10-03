#!/usr/bin/env python3
"""
EchoPhase: Forward Monte Carlo Price Simulator
Architecture: Harmonic-Weighted Empirical Block Bootstrap (Era 4)
Integrates a 240-day institutional rebalancing phase filter into stationary block selection.
"""

import argparse
from datetime import timedelta
import os
import numpy as np
import pandas as pd


def load_and_calibrate(data_path: str):
    df = pd.read_csv(data_path)

    time_col = "date" if "date" in df.columns else "time"
    price_col = "close" if "close" in df.columns else "price"

    df["time"] = pd.to_datetime(df[time_col])
    df["price"] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["time", "price"]).sort_values("time").reset_index(drop=True)

    genesis = pd.to_datetime("2009-01-03 18:15:05", utc=True)
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")

    df["tau"] = (df["time"] - genesis).dt.total_seconds() / 86400.0
    df = df[df["tau"] > 0].copy()
    df["log_tau"] = np.log(df["tau"])
    df["log_price"] = np.log(df["price"])

    # Longitudinal Secular Baseline
    slope, intercept = np.polyfit(df["log_tau"], df["log_price"], 1)
    df["residual"] = df["log_price"] - (intercept + slope * df["log_tau"])

    # Isolate Era 4 (Spot ETF Regime: 2024-01-11 onwards)
    era4_start = pd.to_datetime("2024-01-11", utc=True)
    df_era4 = df[df["time"] >= era4_start].copy().reset_index(drop=True)

    df_era4["residual_diff"] = df_era4["residual"].diff()
    empirical_innovations = df_era4["residual_diff"].dropna().values

    current_price = df["price"].iloc[-1]
    current_tau = df["tau"].iloc[-1]
    current_eps = df["residual"].iloc[-1]
    last_date = df["time"].iloc[-1]

    print("==================================================")
    print(" EchoPhase Model Calibration (Harmonic Era 4 MBB)")
    print("==================================================")
    print(f"Data Cutoff Date:            {last_date.strftime('%Y-%m-%d')}")
    print(f"Latest Reference Price:      ${current_price:,.2f}")
    print(f"Power-Law Elasticity (beta): {slope:.4f}")
    print(f"Current Residual:            {current_eps:.4f}")
    print(f"Era 4 Empirical Shocks:      {len(empirical_innovations)}")
    print("==================================================\n")

    return {
        "alpha": intercept,
        "beta": slope,
        "current_tau": current_tau,
        "current_eps": current_eps,
        "innovations": empirical_innovations,
        "last_date": last_date,
    }


def simulate_echophase(params: dict, n_paths: int = 10000, horizon_days: int = 365, block_size: int = 7, cycle_days: float = 240.0):
    alpha = params["alpha"]
    beta = params["beta"]
    tau_0 = params["current_tau"]
    eps_0 = params["current_eps"]
    innovations = params["innovations"]

    n_innovations = len(innovations)
    max_start_idx = n_innovations - block_size
    n_blocks = int(np.ceil(horizon_days / block_size))

    # Pre-calculate block drifts to characterize blocks as expansion vs consolidation
    block_drifts = np.array([
        np.sum(innovations[i : i + block_size]) for i in range(max_start_idx + 1)
    ])

    np.random.seed(42)
    shocks = np.zeros((horizon_days, n_paths))

    for b in range(n_blocks):
        t_mid = (b + 0.5) * block_size
        # Harmonic phase factor in [-1.0, 1.0]
        # Positive phase favors positive blocks; negative phase favors consolidation blocks
        phase_intensity = np.sin(2.0 * np.pi * t_mid / cycle_days)

        # Soft exponential weighting based on institutional cycle phase
        logits = 0.40 * phase_intensity * (block_drifts / (np.std(block_drifts) + 1e-8))
        probs = np.exp(logits - np.max(logits))
        probs /= np.sum(probs)

        # Sample blocks according to harmonic regime probability
        chosen_indices = np.random.choice(max_start_idx + 1, size=n_paths, p=probs)

        start_day = b * block_size
        end_day = min((b + 1) * block_size, horizon_days)
        actual_len = end_day - start_day

        for p in range(n_paths):
            idx = chosen_indices[p]
            shocks[start_day:end_day, p] = innovations[idx : idx + actual_len]

    # Cumulative residual drift
    simulated_residuals = eps_0 + np.cumsum(shocks, axis=0)

    # Power-law secular backbone
    tau_series = tau_0 + np.arange(1, horizon_days + 1)
    base_log_price = alpha + beta * np.log(tau_series)

    log_prices = base_log_price[:, np.newaxis] + simulated_residuals
    return np.exp(log_prices)


def aggregate_monthly(price_paths: np.ndarray, last_date: pd.Timestamp):
    horizon_days, n_paths = price_paths.shape
    date_range = [last_date + timedelta(days=i + 1) for i in range(horizon_days)]

    df_dates = pd.DataFrame({"date": date_range})
    df_dates["month_label"] = df_dates["date"].dt.strftime("%Y-%m")
    unique_months = df_dates["month_label"].unique()[:12]

    results = []
    for m in unique_months:
        idx = df_dates[df_dates["month_label"] == m].index
        month_paths = price_paths[idx, :]

        month_highs = np.max(month_paths, axis=0)
        month_lows = np.min(month_paths, axis=0)

        results.append({
            "Month": m,
            "Low_p05": np.percentile(month_lows, 5),
            "Low_p16": np.percentile(month_lows, 16),
            "Low_p50": np.percentile(month_lows, 50),
            "Low_p84": np.percentile(month_lows, 84),
            "Low_p95": np.percentile(month_lows, 95),
            "High_p05": np.percentile(month_highs, 5),
            "High_p16": np.percentile(month_highs, 16),
            "High_p50": np.percentile(month_highs, 50),
            "High_p84": np.percentile(month_highs, 84),
            "High_p95": np.percentile(month_highs, 95),
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Run EchoPhase Harmonic Bootstrap Simulator")
    parser.add_argument("--data", type=str, default="data/btc_daily_price.csv", help="Path to reference CSV")
    parser.add_argument("--iterations", type=int, default=10000, help="Number of Monte Carlo paths")
    parser.add_argument("--block-size", type=int, default=7, help="Block size in days for Moving Block Bootstrap")
    parser.add_argument("--output", type=str, default="output/monthly_forecast.csv", help="Output CSV path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    params = load_and_calibrate(args.data)
    price_paths = simulate_echophase(params, n_paths=args.iterations, horizon_days=365, block_size=args.block_size)
    df_forecast = aggregate_monthly(price_paths, params["last_date"])

    df_forecast.to_csv(args.output, index=False)
    print(f"EchoPhase forecast written to: {args.output}\n")

    display_df = df_forecast.copy()
    for col in display_df.columns:
        if col != "Month":
            display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}")

    print(display_df[["Month", "Low_p05", "Low_p50", "High_p50", "High_p95"]].to_string(index=False))


if __name__ == "__main__":
    main()