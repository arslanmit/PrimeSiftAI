"""Generate the model-ranking chart used by the project README."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = ROOT / "breast_cancer_dataset_edit_result.csv"
OUTPUT_FILE = ROOT / "output" / "playwright" / "latest-run-tuned-auc.png"


def main() -> None:
    results = pd.read_csv(RESULTS_FILE, index_col=0)
    latest_timestamp = results["timestamp"].max()
    latest = results.loc[results["timestamp"] == latest_timestamp].copy()

    if latest.empty:
        raise ValueError("No model results were found in the latest run.")

    ranking = latest.nlargest(10, "AUC (Tuned)").sort_values("AUC (Tuned)")
    search_method = latest["Selected search method"].iloc[0]
    scaling_method = latest["Selected scaling method"].iloc[0]

    colors = ["#5B8FF9"] * len(ranking)
    colors[-1] = "#173F7A"

    fig, ax = plt.subplots(figsize=(14, 8), facecolor="white")
    bars = ax.barh(
        ranking.index,
        ranking["AUC (Tuned)"],
        color=colors,
        edgecolor="#173F7A",
        linewidth=0.7,
    )

    ax.set_title(
        "Top 10 Models by Tuned ROC AUC",
        loc="left",
        fontsize=22,
        fontweight="bold",
        color="#172033",
        pad=30,
    )
    ax.text(
        0,
        1.015,
        f"Latest recorded run: {latest_timestamp} | {search_method} search | {scaling_method}",
        transform=ax.transAxes,
        fontsize=11,
        color="#5B6472",
    )

    ax.set_xlim(0, 1.04)
    ax.set_xlabel("ROC AUC (0-1)", fontsize=11, color="#364152")
    ax.set_ylabel("")
    ax.xaxis.grid(True, color="#E4E8EF", linewidth=0.8)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, ranking["AUC (Tuned)"], strict=True):
        ax.text(
            value + 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#172033",
        )

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#AAB2BF")
    ax.tick_params(axis="y", length=0, labelsize=11, colors="#172033")
    ax.tick_params(axis="x", colors="#5B6472")

    fig.text(
        0.01,
        0.01,
        "Source: breast_cancer_dataset_edit_result.csv",
        fontsize=9,
        color="#6B7280",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
