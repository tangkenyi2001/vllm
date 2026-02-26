"""
Generate comparison chart: Worker Controller vs Standard vLLM cold start.

Two simple bar charts side by side:
  Left:  Phase durations (Model Loading, Engine Creation, etc.)
  Right: Total cold start per run

Usage:
    python plot_benchmark.py              # show interactively
    python plot_benchmark.py --save       # save to benchmark_chart.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Benchmark data (facebook/opt-125m, 3 runs)
# ---------------------------------------------------------------------------
RUNS = {
    "run-1": {
        "std": {
            "engine_creation_start": 6.442,
            "model_loading_start": 7.292,
            "model_loading_end": 9.274,
            "engine_creation_end": 10.293,
            "api_server_ready": 13.024,
            "first_inference": 0.134,
            "total": 13.159,
        },
        "wc": {
            "engine_creation_start": 5.980,
            "model_loading_start": 5.980,
            "model_loading_end": 7.294,
            "engine_creation_end": 8.171,
            "api_server_ready": 10.239,
            "first_inference": 0.090,
            "total": 10.334,
        },
    },
    "run-2": {
        "std": {
            "engine_creation_start": 9.423,
            "model_loading_start": 10.259,
            "model_loading_end": 12.249,
            "engine_creation_end": 13.282,
            "api_server_ready": 15.976,
            "first_inference": 0.147,
            "total": 16.124,
        },
        "wc": {
            "engine_creation_start": 5.995,
            "model_loading_start": 5.995,
            "model_loading_end": 7.285,
            "engine_creation_end": 8.220,
            "api_server_ready": 10.234,
            "first_inference": 0.073,
            "total": 10.311,
        },
    },
    "run-3": {
        "std": {
            "engine_creation_start": 6.424,
            "model_loading_start": 7.265,
            "model_loading_end": 9.304,
            "engine_creation_end": 10.335,
            "api_server_ready": 13.023,
            "first_inference": 0.148,
            "total": 13.172,
        },
        "wc": {
            "engine_creation_start": 8.963,
            "model_loading_start": 8.963,
            "model_loading_end": 10.250,
            "engine_creation_end": 11.622,
            "api_server_ready": 14.201,
            "first_inference": 0.068,
            "total": 14.274,
        },
    },
}

STD_COLOR = "#e74c3c"
WC_COLOR = "#3498db"


def avg(values):
    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    # Compute average durations
    std_model_load = avg([r["std"]["model_loading_end"] - r["std"]["model_loading_start"] for r in RUNS.values()])
    std_engine     = avg([r["std"]["engine_creation_end"] - r["std"]["engine_creation_start"] for r in RUNS.values()])
    std_startup    = avg([r["std"]["api_server_ready"] for r in RUNS.values()])
    std_inference  = avg([r["std"]["first_inference"] for r in RUNS.values()])

    wc_model_load = avg([r["wc"]["model_loading_end"] - r["wc"]["model_loading_start"] for r in RUNS.values()])
    wc_engine     = avg([r["wc"]["engine_creation_end"] - r["wc"]["engine_creation_start"] for r in RUNS.values()])
    wc_startup    = avg([r["wc"]["api_server_ready"] for r in RUNS.values()])
    wc_inference  = avg([r["wc"]["first_inference"] for r in RUNS.values()])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Cold Start: Standard vLLM vs Worker Controller\nfacebook/opt-125m  •  1 GPU  •  avg of 3 runs",
        fontsize=14, fontweight="bold",
    )

    # ── Left: Phase durations ──
    phases = ["Model\nLoading", "Engine\nCreation", "API Server\nStartup", "First\nInference"]
    std_vals = [std_model_load, std_engine, std_startup, std_inference]
    wc_vals  = [wc_model_load,  wc_engine,  wc_startup,  wc_inference]

    x = np.arange(len(phases))
    w = 0.32

    bars_s = ax1.bar(x - w/2, std_vals, w, color=STD_COLOR, label="Std vLLM", alpha=0.85)
    bars_w = ax1.bar(x + w/2, wc_vals,  w, color=WC_COLOR,  label="Worker Controller", alpha=0.85)

    for bars in (bars_s, bars_w):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, h + 0.15,
                     f"{h:.2f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(phases, fontsize=10)
    ax1.set_ylabel("Seconds", fontsize=11)
    ax1.set_title("Phase Durations (avg)", fontsize=12, fontweight="bold")
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax1.grid(axis="y", alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(fontsize=10)

    # ── Right: Total cold start per run ──
    run_names = list(RUNS.keys())
    x2 = np.arange(len(run_names))

    std_totals = [RUNS[r]["std"]["total"] for r in run_names]
    wc_totals  = [RUNS[r]["wc"]["total"]  for r in run_names]

    bars_s2 = ax2.bar(x2 - w/2, std_totals, w, color=STD_COLOR, label="Std vLLM", alpha=0.85)
    bars_w2 = ax2.bar(x2 + w/2, wc_totals,  w, color=WC_COLOR,  label="Worker Controller", alpha=0.85)

    for bars in (bars_s2, bars_w2):
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.15,
                     f"{h:.1f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")

    for i, (s, wc) in enumerate(zip(std_totals, wc_totals)):
        if wc > 0:
            spd = s / wc
            ax2.text(x2[i], max(s, wc) + 1.0, f"{spd:.1f}x",
                     ha="center", fontsize=10, fontweight="bold",
                     color="#27ae60" if spd > 1 else "#e74c3c")

    ax2.set_xticks(x2)
    ax2.set_xticklabels([f"Run {i+1}" for i in range(len(run_names))], fontsize=10)
    ax2.set_ylabel("Seconds", fontsize=11)
    ax2.set_title("Total Cold Start (per run)", fontsize=12, fontweight="bold")
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    if args.save or args.output:
        out = args.output or os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_chart.png")
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
