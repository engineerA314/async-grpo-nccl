#!/usr/bin/env python3
"""
Collect weight sync timing from benchmark logs and generate comparison charts.

Usage:
    python benchmark/collect_and_plot.py ./benchmark/results/
    python benchmark/collect_and_plot.py ./benchmark/results/ --output ./benchmark/charts/
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Log patterns
ORIGINAL_PATTERN = re.compile(
    r"Updated weights on .* in (\d+\.?\d*) seconds Rank: 0"
)
NCCL_PATTERN = re.compile(
    r"\[WEIGHT_SYNC_TIME\] (\d+\.?\d*) seconds"
)

MODEL_SIZES = ["0.5B", "3B", "7B", "14B"]


def parse_log(filepath: str, pattern: re.Pattern) -> list[float]:
    """Extract weight sync times from a log file."""
    times = []
    if not os.path.isfile(filepath):
        return times
    with open(filepath, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                times.append(float(match.group(1)))
    return times


def collect_results(results_dir: str) -> dict:
    """Collect all benchmark results from log files."""
    results = {}
    for size in MODEL_SIZES:
        orig_log = os.path.join(results_dir, f"original_{size}.log")
        nccl_log = os.path.join(results_dir, f"nccl_{size}.log")

        orig_times = parse_log(orig_log, ORIGINAL_PATTERN)
        nccl_times = parse_log(nccl_log, NCCL_PATTERN)

        if not orig_times and not nccl_times:
            continue

        entry = {"model": size}
        if orig_times:
            entry["original"] = {
                "times": orig_times,
                "mean": np.mean(orig_times),
                "std": np.std(orig_times),
                "min": np.min(orig_times),
                "max": np.max(orig_times),
                "count": len(orig_times),
            }
        if nccl_times:
            entry["nccl"] = {
                "times": nccl_times,
                "mean": np.mean(nccl_times),
                "std": np.std(nccl_times),
                "min": np.min(nccl_times),
                "max": np.max(nccl_times),
                "count": len(nccl_times),
            }
        if orig_times and nccl_times:
            entry["speedup"] = np.mean(orig_times) / np.mean(nccl_times)

        results[size] = entry

    return results


def print_summary(results: dict):
    """Print a formatted summary table."""
    print("\n" + "=" * 75)
    print(f"{'Model':>6}  {'Original (s)':>14}  {'NCCL (s)':>14}  {'Speedup':>10}")
    print("-" * 75)
    for size in MODEL_SIZES:
        if size not in results:
            continue
        r = results[size]
        orig_str = f"{r['original']['mean']:.3f} +/- {r['original']['std']:.3f}" if "original" in r else "N/A"
        nccl_str = f"{r['nccl']['mean']:.3f} +/- {r['nccl']['std']:.3f}" if "nccl" in r else "N/A"
        speedup_str = f"{r['speedup']:.1f}x" if "speedup" in r else "N/A"
        print(f"{size:>6}  {orig_str:>14}  {nccl_str:>14}  {speedup_str:>10}")
    print("=" * 75)


def plot_bar_chart(results: dict, output_path: str):
    """Grouped bar chart: model size vs sync time."""
    sizes = [s for s in MODEL_SIZES if s in results]
    if not sizes:
        print("No data to plot.")
        return

    orig_means = []
    orig_stds = []
    nccl_means = []
    nccl_stds = []
    speedups = []

    for s in sizes:
        r = results[s]
        orig_means.append(r.get("original", {}).get("mean", 0))
        orig_stds.append(r.get("original", {}).get("std", 0))
        nccl_means.append(r.get("nccl", {}).get("mean", 0))
        nccl_stds.append(r.get("nccl", {}).get("std", 0))
        speedups.append(r.get("speedup", 0))

    x = np.arange(len(sizes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_orig = ax.bar(
        x - width / 2, orig_means, width,
        yerr=orig_stds, capsize=4,
        label="Ray Object Store (Original)",
        color="#E74C3C", alpha=0.85,
        edgecolor="black", linewidth=0.5,
    )
    bars_nccl = ax.bar(
        x + width / 2, nccl_means, width,
        yerr=nccl_stds, capsize=4,
        label="NCCL Broadcast (Ours)",
        color="#3498DB", alpha=0.85,
        edgecolor="black", linewidth=0.5,
    )

    # Speedup annotations
    for i, sp in enumerate(speedups):
        if sp > 0:
            max_h = max(orig_means[i] + orig_stds[i], nccl_means[i] + nccl_stds[i])
            ax.annotate(
                f"{sp:.1f}x",
                xy=(x[i], max_h),
                ha="center", va="bottom",
                fontsize=12, fontweight="bold", color="#2C3E50",
            )

    ax.set_xlabel("Model Size", fontsize=13)
    ax.set_ylabel("Weight Sync Time (seconds)", fontsize=13)
    ax.set_title(
        "Weight Synchronization: NCCL Broadcast vs Ray Object Store\n"
        "(A100 80GB x8, lower is better)",
        fontsize=14, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def plot_box(results: dict, output_path: str):
    """Box plot showing distribution of sync times."""
    sizes = [s for s in MODEL_SIZES if s in results and "original" in results[s] and "nccl" in results[s]]
    if not sizes:
        return

    fig, axes = plt.subplots(1, len(sizes), figsize=(4 * len(sizes), 5), sharey=False)
    if len(sizes) == 1:
        axes = [axes]

    for i, s in enumerate(sizes):
        r = results[s]
        data = [r["original"]["times"], r["nccl"]["times"]]
        bp = axes[i].boxplot(
            data,
            tick_labels=["Ray\nObject Store", "NCCL\nBroadcast"],
            patch_artist=True,
            widths=0.5,
        )
        bp["boxes"][0].set_facecolor("#E74C3C")
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor("#3498DB")
        bp["boxes"][1].set_alpha(0.7)
        axes[i].set_title(f"Qwen2.5-{s}", fontsize=12, fontweight="bold")
        axes[i].set_ylabel("Time (s)" if i == 0 else "")
        axes[i].grid(axis="y", alpha=0.3)

    fig.suptitle("Weight Sync Time Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Collect and plot weight sync benchmark results")
    parser.add_argument("results_dir", help="Directory containing benchmark log files")
    parser.add_argument("--output", "-o", default=None, help="Output directory for charts")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"ERROR: {args.results_dir} is not a directory")
        sys.exit(1)

    output_dir = args.output or args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    # Collect
    results = collect_results(args.results_dir)
    if not results:
        print("No benchmark results found. Expected files: original_*.log, nccl_*.log")
        sys.exit(1)

    # Summary
    print_summary(results)

    # Save JSON
    json_path = os.path.join(output_dir, "results.json")
    serializable = {}
    for k, v in results.items():
        serializable[k] = {}
        for kk, vv in v.items():
            if isinstance(vv, dict):
                serializable[k][kk] = {kkk: float(vvv) if isinstance(vvv, (np.floating, float)) else vvv for kkk, vvv in vv.items()}
            elif isinstance(vv, (np.floating, float)):
                serializable[k][kk] = float(vv)
            else:
                serializable[k][kk] = vv
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results JSON: {json_path}")

    # Plots
    plot_bar_chart(results, os.path.join(output_dir, "weight_sync_comparison.png"))
    plot_bar_chart(results, os.path.join(output_dir, "weight_sync_comparison.svg"))
    plot_box(results, os.path.join(output_dir, "weight_sync_boxplot.png"))
    plot_box(results, os.path.join(output_dir, "weight_sync_boxplot.svg"))


if __name__ == "__main__":
    main()
