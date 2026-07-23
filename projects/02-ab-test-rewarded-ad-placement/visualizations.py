"""
Generate all charts used in the notebook / README for the A/B test project.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 130
plt.rcParams["font.size"] = 10

COLORS = {"control": "#6B7280", "treatment": "#2563EB"}

import os
BASE = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(BASE, "..", "data", "gaming_adtech_ab_test_dataset.csv"))
IMG_DIR = os.path.join(BASE, "..", "images")

# ------------------------------------------------------------------
# 1. Sample Ratio Mismatch bar chart
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 4))
counts = df["experiment_group"].value_counts().reindex(["control", "treatment"])
bars = ax.bar(counts.index, counts.values, color=[COLORS["control"], COLORS["treatment"]])
ax.axhline(len(df) / 2, color="black", linestyle="--", linewidth=1, label="Expected (50/50)")
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 300, f"{val:,}", ha="center", fontweight="bold")
ax.set_ylabel("Users")
ax.set_title("Sample Ratio Check: Control vs Treatment\n(χ² p = 0.096 — no SRM detected)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/01_srm_check.png")
plt.close()

# ------------------------------------------------------------------
# 2. Primary metric: Ad revenue per user with CI
# ------------------------------------------------------------------
means = df.groupby("experiment_group")["ad_revenue_usd"].mean().reindex(["control", "treatment"])
sems = df.groupby("experiment_group")["ad_revenue_usd"].sem().reindex(["control", "treatment"])
ci95 = sems * 1.96

fig, ax = plt.subplots(figsize=(5, 4.5))
bars = ax.bar(means.index, means.values, yerr=ci95.values, capsize=8,
               color=[COLORS["control"], COLORS["treatment"]], alpha=0.9)
for bar, val in zip(bars, means.values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.0015, f"${val:.4f}", ha="center", fontweight="bold")
ax.set_ylabel("Mean Ad Revenue / User (USD)")
ax.set_title("Primary Metric: Ad Revenue per User\n+26.95% relative lift (p < 0.001)")
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/02_primary_metric_revenue.png")
plt.close()

# ------------------------------------------------------------------
# 3. Guardrail metric: Day-1 retention with CI
# ------------------------------------------------------------------
ret = df.groupby("experiment_group")["day1_retained"].mean().reindex(["control", "treatment"])
n = df.groupby("experiment_group")["day1_retained"].count().reindex(["control", "treatment"])
se_ret = np.sqrt(ret * (1 - ret) / n)
ci95_ret = se_ret * 1.96

fig, ax = plt.subplots(figsize=(5, 4.5))
bars = ax.bar(ret.index, ret.values, yerr=ci95_ret.values, capsize=8,
               color=[COLORS["control"], COLORS["treatment"]], alpha=0.9)
for bar, val in zip(bars, ret.values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012, f"{val:.1%}", ha="center", fontweight="bold")
ax.set_ylabel("Day-1 Retention Rate")
ax.set_title("Guardrail Metric: Day-1 Retention\n-2.21pp absolute (p < 0.001) — GUARDRAIL VIOLATED")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_ylim(0, 0.8)
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/03_guardrail_retention.png")
plt.close()

# ------------------------------------------------------------------
# 4. Secondary metrics grid
# ------------------------------------------------------------------
metrics = [
    ("ad_impressions", "Ad Impressions / User", "{:.2f}"),
    ("ad_completion_rate", "Ad Completion Rate", "{:.1%}"),
    ("total_revenue_usd", "Total Revenue / User ($)", "${:.3f}"),
    ("uninstall_after_ad_flag", "Uninstall-after-Ad Rate", "{:.2%}"),
]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, (col, title, fmt) in zip(axes.flat, metrics):
    vals = df.groupby("experiment_group")[col].mean().reindex(["control", "treatment"])
    bars = ax.bar(vals.index, vals.values, color=[COLORS["control"], COLORS["treatment"]], alpha=0.9)
    for bar, val in zip(bars, vals.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.02, fmt.format(val), ha="center", fontweight="bold", fontsize=9)
    ax.set_title(title, fontsize=11)
fig.suptitle("Secondary / Diagnostic Metrics", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/04_secondary_metrics.png")
plt.close()

# ------------------------------------------------------------------
# 5. Segmentation: relative lift in primary metric by segment
# ------------------------------------------------------------------
import json
with open(os.path.join(BASE, "..", "reports", "analysis_results.json")) as f:
    results = json.load(f)

seg_df = pd.DataFrame(results["segmentation"])
seg_df["label"] = seg_df["dimension"].str.replace("_", " ").str.title() + ": " + seg_df["segment"]
seg_df = seg_df.sort_values("rel_lift_pct")

fig, ax = plt.subplots(figsize=(8, 5))
colors_bar = ["#2563EB" if sig else "#93C5FD" for sig in seg_df["sig_bonferroni"]]
bars = ax.barh(seg_df["label"], seg_df["rel_lift_pct"], color=colors_bar)
for bar, val in zip(bars, seg_df["rel_lift_pct"]):
    ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:+.1f}%", va="center", fontsize=9)
ax.set_xlabel("Relative Lift in Ad Revenue / User (%)")
ax.set_title("Heterogeneity: Ad Revenue Lift by Segment\n(all segments significant after Bonferroni correction)")
ax.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/05_segmentation_lift.png")
plt.close()

# ------------------------------------------------------------------
# 6. Distribution of ad revenue (log scale) - shows skew, justifies MW test
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
for grp, color in COLORS.items():
    vals = df.loc[df.experiment_group == grp, "ad_revenue_usd"]
    vals_nonzero = vals[vals > 0]
    ax.hist(vals_nonzero, bins=60, alpha=0.55, label=grp.title(), color=color, density=True)
ax.set_xlabel("Ad Revenue per User (USD, excl. zeros)")
ax.set_ylabel("Density")
ax.set_title("Distribution of Ad Revenue per User\n(right-skewed — motivates Mann-Whitney U as robustness check)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/06_revenue_distribution.png")
plt.close()

print("All charts saved to", IMG_DIR)
