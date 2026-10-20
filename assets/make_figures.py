"""
Regenerates the README figures from the evaluation results.

Reads:  data/evaluation/ragas_scores.json, data/evaluation/cost_estimates.json
Writes: assets/{quality_vs_cost,ragas_metrics,ablation,cost_per_query}.png

Run from the project root:  python assets/make_figures.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
EVAL = ROOT / "data/evaluation"
OUT = ROOT / "assets"

SYSTEMS = {1: "S1 BM25", 2: "S2 Vector", 3: "S3 Hybrid", 4: "S4 Hybrid+KG", 5: "S5 Agentic"}
COST_KEYS = {1: "S1 BM25", 2: "S2 Vector", 3: "S3 Hybrid", 4: "S4 Hybrid+Graph", 5: "S5 Agentic"}
COLORS = ["#9aa5b1", "#7b8794", "#52606d", "#3e7cb1", "#d9822b"]
METRICS = ["faithfulness", "context_recall", "answer_correctness"]
METRIC_LABELS = ["Faithfulness", "Context recall", "Answer correctness"]

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 140, "savefig.bbox": "tight"})

ragas = json.load(open(EVAL / "ragas_scores.json"))
costs = json.load(open(EVAL / "cost_estimates.json"))["summary_common_set"]

scores = {s["system_id"]: {m: s["metrics"][m]["score"] for m in METRICS}
          for s in ragas["systems"].values()}


def quality_vs_cost():
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for sid, color in zip(SYSTEMS, COLORS):
        x = costs[COST_KEYS[sid]]["mean_total_tokens"]
        y = scores[sid]["answer_correctness"]
        ax.scatter(x, y, s=120, color=color, zorder=3)
        label = SYSTEMS[sid]
        if sid == 5:
            label += f"\n({costs[COST_KEYS[sid]]['mean_llm_calls']:.1f} LLM calls)"
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.set_xlim(0, 22000)
    ax.set_ylim(0.25, 0.5)
    ax.set_xlabel("Tokens per query")
    ax.set_ylabel("Answer correctness")
    ax.set_title("Quality vs. cost per query", loc="left", fontweight="bold")
    ax.grid(alpha=0.3)
    fig.savefig(OUT / "quality_vs_cost.png")
    plt.close(fig)


def ragas_metrics():
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    width = 0.15
    x = np.arange(len(METRICS))
    for k, (sid, color) in enumerate(zip(SYSTEMS, COLORS)):
        ax.bar(x + (k - 2) * width, [scores[sid][m] for m in METRICS], width,
               color=color, label=SYSTEMS[sid], edgecolor="white")
    ax.set_xticks(x, METRIC_LABELS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("RAGAS metrics (35 validated questions)", loc="left", fontweight="bold", pad=30)
    ax.legend(ncol=5, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.savefig(OUT / "ragas_metrics.png")
    plt.close(fig)


def ablation():
    steps = [("BM25 → Vector", 1, 2), ("Vector → Hybrid", 2, 3), ("Hybrid → +KG", 3, 4),
             ("+KG → Agent", 4, 5), ("BM25 → Agent\n(baseline → solution)", 1, 5)]
    delta = np.array([[(scores[b][m] - scores[a][m]) * 100 for m in METRICS] for _, a, b in steps])

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.imshow(delta, cmap="RdYlGn", vmin=-50, vmax=50, aspect="auto")
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            v = delta[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=11,
                    color="white" if abs(v) > 30 else "black",
                    fontweight="bold" if abs(v) > 30 else "normal")
    ax.axhline(3.5, color="white", linewidth=6)
    ax.set_xticks(range(3), METRIC_LABELS)
    ax.set_yticks(range(len(steps)), [s[0] for s in steps])
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title("Ablation: change when adding each component (percentage points)",
                 loc="left", fontweight="bold")
    fig.savefig(OUT / "ablation.png")
    plt.close(fig)


def cost_per_query():
    panels = [("mean_total_tokens", "Tokens per query", "{:,.0f}"),
              ("mean_llm_calls", "LLM calls per query", "{:.1f}"),
              ("mean_latency_s", "Mean latency (s)", "{:.1f}")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (key, title, fmt) in zip(axes, panels):
        values = [costs[COST_KEYS[sid]][key] for sid in SYSTEMS]
        bars = ax.bar(range(5), values, color=COLORS)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(5), [f"S{sid}" for sid in SYSTEMS])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_title(title, fontsize=11)
    fig.suptitle(f"Cost per query ({costs['S1 BM25']['n_questions']} common questions)",
                 y=1.06, fontweight="bold", fontsize=13)
    fig.savefig(OUT / "cost_per_query.png")
    plt.close(fig)


if __name__ == "__main__":
    quality_vs_cost()
    ragas_metrics()
    ablation()
    cost_per_query()
    print(f"Figures written to {OUT}")
