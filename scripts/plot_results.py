#!/usr/bin/env python3
"""Generate benchmark charts from CSV results."""

import sys
import csv
import os


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <benchmark.csv>")
        sys.exit(1)

    filepath = sys.argv[1]

    # Read CSV
    versions, times, losses = [], [], []
    try:
        with open(filepath) as f:
            for row in csv.DictReader(f):
                versions.append(row["Version"])
                times.append(float(row["Time_ms"]) / 1000)
                losses.append(row.get("Loss", "N/A"))
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # Calculate speedups
    base = times[0] if times else 1
    speedups = [base / t if t > 0 else 0 for t in times]

    # Print table
    print("\n[benchmark] Results")
    print(f"{'Version':<15} {'Time(s)':<10} {'Speedup':<10}")
    print("-" * 35)
    for i, v in enumerate(versions):
        print(f"{v:<15} {times[i]:<10.2f} {speedups[i]:<10.1f}x")

    # Generate chart
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.bar(versions, times)
        ax1.set_ylabel("Time (s)")
        ax1.set_title("Training Time")
        ax1.tick_params(axis="x", rotation=45)

        ax2.bar(versions, speedups)
        ax2.set_ylabel("Speedup")
        ax2.set_title("Speedup vs Baseline")
        ax2.axhline(y=20, color="r", linestyle="--", alpha=0.5)
        ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        out = os.path.join(os.path.dirname(filepath) or ".", "benchmark.png")
        plt.savefig(out, dpi=150)
        print(f"\n[OK] Chart: {out}")
    except ImportError:
        print("\n[INFO] Install matplotlib for charts: pip install matplotlib")


if __name__ == "__main__":
    main()
