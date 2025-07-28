"""
fit_exponential_decay.py

This script reads a CSV file containing timestamped measurements (e.g., pressure over time),
fits the data to an exponential decay function:

    y(t) = y0 * exp(-t / t0) + C

and generates a semilogarithmic plot of the data along with the fitted curve.
The plot is saved as a PNG file, and the fitted parameters are printed to the console.

Usage:
    python fit_exponential_decay.py <csv_file> [--output <output_png>]

Arguments:
    csv_file         Path to the input CSV file. Must contain:
                     - A time column (parseable as datetime)
                     - A measurement column (e.g., pressure)
    --output         (Optional) Path to save the PNG plot. Defaults to 'fit_result.png'.

Input:
    CSV file with two columns:
        1. 'Time' - timestamps
        2. Measured values (e.g., pressure). Column name may contain metadata.

Output:
    - A semilog-y plot of the data and fitted model saved as PNG.
    - Fitted parameters printed to the console.

Dependencies:
    - numpy
    - pandas
    - matplotlib
    - scipy

Example:
    python fit_exponential_decay.py "Trap-data.csv" --output "decay_fit.png"
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def exp_decay(t, y0, t0, C):
    return y0 * np.exp(-t / t0) + C

def main(csv_file, output_png):
    # Load the CSV
    df = pd.read_csv(csv_file)

    # Rename pressure column for convenience
    pressure_col = df.columns[1]
    df.rename(columns={pressure_col: "Pressure"}, inplace=True)

    # Parse time and compute elapsed time in seconds
    df["Time"] = pd.to_datetime(df["Time"])
    t0 = df["Time"].iloc[0]
    df["Elapsed_s"] = (df["Time"] - t0).dt.total_seconds()

    # Convert pressure to numeric
    df["Pressure"] = pd.to_numeric(df["Pressure"], errors="coerce")
    df.dropna(subset=["Pressure", "Elapsed_s"], inplace=True)

    # Initial guess
    initial_guess = [df["Pressure"].max(), 100.0, df["Pressure"].min()]

    # Fit the model
    popt, _ = curve_fit(exp_decay, df["Elapsed_s"], df["Pressure"], p0=initial_guess)
    y0_fit, t0_fit, C_fit = popt

    # Generate fitted curve
    t_fit = np.linspace(df["Elapsed_s"].min(), df["Elapsed_s"].max(), 500)
    y_fit = exp_decay(t_fit, *popt)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.semilogy(df["Elapsed_s"], df["Pressure"], 'o', color='black', markersize=4, label='Data')
    plt.semilogy(t_fit, y_fit, 'r-', label=f'Fit: y0={y0_fit:.3e}, t0={t0_fit:.3g} s, C={C_fit:.3e}')
    plt.xlabel("Time (s)")
    plt.ylabel("Pressure (mbar)")
    plt.title("Exponential Decay Fit")
    plt.grid(True, which='major', axis='both', linestyle='--')
    plt.minorticks_on()
    plt.grid(True, which='minor', axis='both', linestyle=':', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()

    # Print results
    print("Fitted parameters:")
    print(f"y0 = {y0_fit:.6e}")
    print(f"t0 = {t0_fit:.6g} seconds")
    print(f"C  = {C_fit:.6e}")
    print(f"Plot saved to: {output_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit data to exponential decay: y = y0 * exp(-t/t0) + C")
    parser.add_argument("csv_file", help="Input CSV file")
    parser.add_argument("--output", default="fit_result.png", help="Output PNG file")
    args = parser.parse_args()
    main(args.csv_file, args.output)
