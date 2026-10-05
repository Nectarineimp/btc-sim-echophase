#!/usr/bin/env python3
"""
EchoPhase: Forward 12-Month Stochastic Forecast Simulator
Architecture: Harmonic-Weighted Empirical Block Bootstrap (Era 4)
Calibrated to 285-day Institutional Rebalancing Modes & Phase-Aligned Initial Conditions.
"""

import argparse
from datetime import timedelta
import os
import sys
import numpy as np
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run EchoPhase Harmonic Bootstrap Simulator"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="btc_daily_price.csv",
        help="Path to reference historical CSV.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="echophase_forecast.csv",
        help="Output CSV path for monthly quantiles.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10000,
        help="Number of Monte Carlo paths (default: 10000).",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=7,
        help="Block size in days for Moving Block Bootstrap (default: 7).",
    )
    parser.add_argument(
        "--cycle-days",
        type=float,
        default=285.0,
        help="Institutional harmonic mode in days (default: 285.0).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Forward forecast horizon in days (default: 365).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


def load_and_calibrate(data_path: str):
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error loading {data_path}: {e}", file=sys.stderr)
        sys.exit(1)

    time_col = "date" if "date" in df.columns else "time"
    price_col = "close" if "close" in df.columns else "price"

    if time_col not in df.columns or price_col not in df.columns:
        print(
            f"Missing required columns. Found: {list(df.columns)}", file=sys.stderr
        )
        sys.exit(1)

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

    # Secular Longitudinal Power-Law Baseline: ln(P) = alpha + beta * ln(tau)
    slope, intercept = np.polyfit(df["log_tau"], df["log_price"], 1)
    df["residual"] = df["log_price"] - (intercept + slope * df["log_tau"])

    # Era 4 Isolation (Spot ETF Regime: 2024-01-11 onwards)
    era4_start = pd.to_datetime("2024-01-11", utc=True)
    df_era4 = df[df["time"] >= era4_start].copy().reset_index(drop=True)

    # Empirical residual increments: delta_eps = eps_t - eps_{t-1}
    df_era4["residual_diff"] = df_era4["residual"].diff()
    empirical_innovations = df_era4["residual_diff"].dropna().values

    current_price = df["price"].iloc[-1]
    current_tau = df["tau"].iloc[-1]
    current_eps = df["residual"].iloc[-1]
    last_date = df["time"].iloc[-1]

    print("==================================================")
    print(" EchoPhase Calibration (Harmonic Era 4 Resampler)")
    print("==================================================")
    print(f"Data Cutoff Date:            {last_date.strftime('%Y-%m-%d')}")
    print(f"Latest Reference Price:      ${current_price:,.2f}")
    print(f"Power-Law Elasticity (beta): {slope:.4f}")
    print(f"Current Residual (eps_0):    {current_eps:.4f}")
    print(f"Era 4 Empirical Shocks:      {len(empirical_innovations)}")
    print(f"Daily Innovation Mean:       {np.mean(empirical_innovations):.6f}")
    print(f"Daily Innovation Std:        {np.std(empirical_innovations):.6f}")
    print("==================================================\n")

    return {
        "alpha": intercept,
        "beta": slope,
        "current_tau": current_tau,
        "current_eps": current_eps,
        "innovations": empirical_innovations,
        "last_date": last_date,
    }


def simulate_echophase(
    params: dict,
    n_paths: int = 10000,
    horizon_days: int = 365,
    block_size: int = 7,
    cycle_days: float = 285.0,
    seed: int = 42,
):
    alpha = params["alpha"]
    beta = params["beta"]
    tau_0 = params["current_tau"]
    eps_0 = params["current_eps"]
    innovations = params["innovations"]

    n_innovations = len(innovations)
    max_start_idx = n_innovations - block_size
    n_blocks = int(np.ceil(horizon_days / block_size))

    # Precompute aggregate drift for all valid candidate blocks in Era 4
    block_drifts = np.array(
        [np.sum(innovations[i : i + block_size]) for i in range(max_start_idx + 1)]
    )
    drift_std = np.std(block_drifts) + 1e-8
    normalized_drifts = block_drifts / drift_std

    np.random.seed(seed)
    shocks = np.zeros((horizon_days, n_paths))

    # Phase offset: -pi/2 anchors t=0 (late May 2026) at cycle trough/absorption
    phase_offset = -np.pi / 2.0

    for b in range(n_blocks):
        t_mid = (b + 0.5) * block_size

        # Institutional harmonic cycle: smooth cosine oscillation [-1, 1]
        cycle_phase = np.cos(2.0 * np.pi * t_mid / cycle_days + phase_offset)

        # Tempered logit sensitivity (0.18): biases probabilities without starving diversity
        logits = 0.18 * cycle_phase * normalized_drifts
        probs = np.exp(logits - np.max(logits))
        probs /= np.sum(probs)

        # Draw empirical blocks conditioned on phase state
        chosen_indices = np.random.choice(max_start_idx + 1, size=n_paths, p=probs)

        start_day = b * block_size
        end_day = min((b + 1) * block_size, horizon_days)
        actual_len = end_day - start_day

        for p in range(n_paths):
            idx = chosen_indices[p]
            shocks[start_day:end_day, p] = innovations[idx : idx + actual_len]

    # Cumulative residual drift
    simulated_residuals = eps_0 + np.cumsum(shocks, axis=0)

    # Reconstruct paths against the deterministic secular power-law backbone
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
    args = parse_arguments()
    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    params = load_and_calibrate(args.data)
    price_paths = simulate_echophase(
        params,
        n_paths=args.iterations,
        horizon_days=args.days,
        block_size=args.block_size,
        cycle_days=args.cycle_days,
        seed=args.seed,
    )
    df_forecast = aggregate_monthly(price_paths, params["last_date"])
    df_forecast.to_csv(args.output, index=False)
    print(f"EchoPhase forecast written to: {args.output}\n")

    display_df = df_forecast.copy()
    for col in display_df.columns:
        if col != "Month":
            display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}")

    print(
        display_df[
            ["Month", "Low_p05", "Low_p50", "High_p50", "High_p95"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
    