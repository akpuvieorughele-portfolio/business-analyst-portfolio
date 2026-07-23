"""
End-to-end statistical analysis for the Rewarded Ad Placement A/B Test.

Covers:
  - Sample Ratio Mismatch (SRM) check
  - Sample size / power sanity check
  - Primary metric test (ad revenue per user) - Mann-Whitney (skewed $ data)
  - Guardrail metric test (Day-1 retention) - two-proportion z-test
  - Secondary/diagnostic metrics
  - Segmentation / heterogeneity analysis with Bonferroni correction
  - Confidence intervals for all key comparisons

Run: python3 analysis.py
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.stats.api as sms
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from statsmodels.stats.power import NormalIndPower, TTestIndPower
import json

pd.set_option("display.width", 120)

import os
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "gaming_adtech_ab_test_dataset.csv")
df = pd.read_csv(DATA_PATH)

results = {}  # collect everything for the writeup / JSON export

print("=" * 70)
print("1. DATA OVERVIEW")
print("=" * 70)
print(f"Total users: {len(df):,}")
print(df["experiment_group"].value_counts())
print()

# ------------------------------------------------------------------
# 2. SAMPLE RATIO MISMATCH (SRM) CHECK
# ------------------------------------------------------------------
print("=" * 70)
print("2. SAMPLE RATIO MISMATCH (SRM) CHECK")
print("=" * 70)

n_control = (df["experiment_group"] == "control").sum()
n_treatment = (df["experiment_group"] == "treatment").sum()
n_total = n_control + n_treatment
expected = n_total / 2

chi2_srm, p_srm = stats.chisquare([n_control, n_treatment], f_exp=[expected, expected])

print(f"Control: {n_control:,} | Treatment: {n_treatment:,} | Expected each: {expected:,.0f}")
print(f"Chi-square: {chi2_srm:.4f}, p-value: {p_srm:.4f}")
srm_flag = p_srm < 0.001  # standard SRM threshold is stricter than 0.05
print(f"SRM detected (p < 0.001)?  {'YES - INVESTIGATE' if srm_flag else 'No - split is healthy'}")
print()

results["srm_check"] = {
    "n_control": int(n_control),
    "n_treatment": int(n_treatment),
    "chi2": round(chi2_srm, 4),
    "p_value": round(p_srm, 4),
    "srm_detected": bool(srm_flag),
}

# ------------------------------------------------------------------
# 3. RANDOMIZATION BALANCE CHECK (pre-treatment covariates)
# ------------------------------------------------------------------
print("=" * 70)
print("3. RANDOMIZATION BALANCE CHECK (covariate balance)")
print("=" * 70)

balance_checks = {}
for col in ["platform", "country_tier", "user_type", "acquisition_channel"]:
    ct = pd.crosstab(df[col], df["experiment_group"])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    balance_checks[col] = round(p, 4)
    print(f"{col:25s} chi2 p-value = {p:.4f}  {'(imbalance flag)' if p < 0.05 else '(balanced)'}")

# continuous covariate
t_stat, p_dev = stats.ttest_ind(
    df.loc[df.experiment_group == "control", "device_age_months"],
    df.loc[df.experiment_group == "treatment", "device_age_months"],
)
balance_checks["device_age_months"] = round(p_dev, 4)
print(f"{'device_age_months':25s} t-test  p-value = {p_dev:.4f}  {'(imbalance flag)' if p_dev < 0.05 else '(balanced)'}")
results["balance_checks"] = balance_checks
print()

# ------------------------------------------------------------------
# 4. SAMPLE SIZE / POWER SANITY CHECK (retrospective)
# ------------------------------------------------------------------
print("=" * 70)
print("4. SAMPLE SIZE / POWER CHECK")
print("=" * 70)

# Primary metric power (using retention as the binary reference metric,
# since that's what the original design would have powered on)
baseline_retention = df.loc[df.experiment_group == "control", "day1_retained"].mean()
mde_abs = 0.02  # 2 percentage point MDE - typical for mature mobile games
alpha = 0.05
power_target = 0.80

effect_size_h = sms.proportion_effectsize(baseline_retention, baseline_retention + mde_abs)
analysis_power = NormalIndPower()
required_n_per_group = analysis_power.solve_power(
    effect_size=effect_size_h, alpha=alpha, power=power_target, ratio=1.0
)

print(f"Baseline Day-1 retention (control): {baseline_retention:.4f}")
print(f"Target MDE: {mde_abs*100:.1f} percentage points (absolute)")
print(f"Alpha: {alpha}, Target power: {power_target}")
print(f"Required sample size per group: {required_n_per_group:,.0f}")
print(f"Actual sample size per group: control={n_control:,}, treatment={n_treatment:,}")
print(f"Sufficiently powered? {'YES' if min(n_control, n_treatment) >= required_n_per_group else 'NO'}")

results["power_analysis"] = {
    "baseline_retention": round(baseline_retention, 4),
    "mde_abs": mde_abs,
    "alpha": alpha,
    "power_target": power_target,
    "required_n_per_group": round(required_n_per_group, 0),
    "actual_n_control": int(n_control),
    "actual_n_treatment": int(n_treatment),
}
print()

# ------------------------------------------------------------------
# 5. PRIMARY METRIC: Ad Revenue per User
# ------------------------------------------------------------------
print("=" * 70)
print("5. PRIMARY METRIC: Ad Revenue per User (USD)")
print("=" * 70)

rev_c = df.loc[df.experiment_group == "control", "ad_revenue_usd"]
rev_t = df.loc[df.experiment_group == "treatment", "ad_revenue_usd"]

mean_c, mean_t = rev_c.mean(), rev_t.mean()
rel_lift = (mean_t - mean_c) / mean_c * 100

# Revenue data is right-skewed (lots of zeros/small values) -> Mann-Whitney U
# in addition to Welch's t-test (CLT still applies reasonably at n=25k/group,
# but we report both for robustness)
u_stat, p_mw = stats.mannwhitneyu(rev_t, rev_c, alternative="two-sided")
t_stat_rev, p_ttest_rev = stats.ttest_ind(rev_t, rev_c, equal_var=False)

# Bootstrap CI for the difference in means (robust to skew)
rng = np.random.default_rng(42)
n_boot = 5000
boot_diffs = np.empty(n_boot)
rev_c_arr, rev_t_arr = rev_c.values, rev_t.values
for i in range(n_boot):
    bc = rng.choice(rev_c_arr, size=len(rev_c_arr), replace=True)
    bt = rng.choice(rev_t_arr, size=len(rev_t_arr), replace=True)
    boot_diffs[i] = bt.mean() - bc.mean()
ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])

print(f"Control mean ad revenue/user:   ${mean_c:.4f}")
print(f"Treatment mean ad revenue/user: ${mean_t:.4f}")
print(f"Absolute lift: ${mean_t - mean_c:.4f}  |  Relative lift: {rel_lift:+.2f}%")
print(f"95% Bootstrap CI on difference: [${ci_low:.4f}, ${ci_high:.4f}]")
print(f"Welch's t-test: t={t_stat_rev:.3f}, p={p_ttest_rev:.6f}")
print(f"Mann-Whitney U test: U={u_stat:.0f}, p={p_mw:.6f}")
print(f"Statistically significant (alpha=0.05)? {'YES' if p_ttest_rev < 0.05 else 'NO'}")
print()

results["primary_metric_ad_revenue"] = {
    "control_mean": round(mean_c, 4),
    "treatment_mean": round(mean_t, 4),
    "abs_lift": round(mean_t - mean_c, 4),
    "rel_lift_pct": round(rel_lift, 2),
    "ci_95_low": round(ci_low, 4),
    "ci_95_high": round(ci_high, 4),
    "ttest_p": p_ttest_rev,
    "mannwhitney_p": p_mw,
}

# ------------------------------------------------------------------
# 6. GUARDRAIL METRIC: Day-1 Retention
# ------------------------------------------------------------------
print("=" * 70)
print("6. GUARDRAIL METRIC: Day-1 Retention")
print("=" * 70)

ret_c_count = df.loc[df.experiment_group == "control", "day1_retained"].sum()
ret_t_count = df.loc[df.experiment_group == "treatment", "day1_retained"].sum()
ret_c_rate = ret_c_count / n_control
ret_t_rate = ret_t_count / n_treatment

z_stat, p_prop = proportions_ztest(
    [ret_t_count, ret_c_count], [n_treatment, n_control], alternative="two-sided"
)

ci_c = proportion_confint(ret_c_count, n_control, alpha=0.05, method="wilson")
ci_t = proportion_confint(ret_t_count, n_treatment, alpha=0.05, method="wilson")

diff_ret = ret_t_rate - ret_c_rate
se_diff = np.sqrt(ret_c_rate*(1-ret_c_rate)/n_control + ret_t_rate*(1-ret_t_rate)/n_treatment)
ci_diff = (diff_ret - 1.96*se_diff, diff_ret + 1.96*se_diff)

print(f"Control Day-1 retention:   {ret_c_rate:.4f}  (95% CI: {ci_c[0]:.4f} - {ci_c[1]:.4f})")
print(f"Treatment Day-1 retention: {ret_t_rate:.4f}  (95% CI: {ci_t[0]:.4f} - {ci_t[1]:.4f})")
print(f"Absolute difference: {diff_ret*100:+.2f} pp  (95% CI: {ci_diff[0]*100:+.2f} to {ci_diff[1]*100:+.2f} pp)")
print(f"Two-proportion z-test: z={z_stat:.3f}, p={p_prop:.6f}")
guardrail_violated = (p_prop < 0.05) and (diff_ret < 0)
print(f"Guardrail violated (significant DECREASE)? {'YES - FLAG' if guardrail_violated else 'No'}")
print()

results["guardrail_retention"] = {
    "control_rate": round(ret_c_rate, 4),
    "treatment_rate": round(ret_t_rate, 4),
    "abs_diff_pp": round(diff_ret * 100, 3),
    "ci_diff_low_pp": round(ci_diff[0] * 100, 3),
    "ci_diff_high_pp": round(ci_diff[1] * 100, 3),
    "z_stat": round(z_stat, 4),
    "p_value": p_prop,
    "guardrail_violated": bool(guardrail_violated),
}

# ------------------------------------------------------------------
# 7. SECONDARY / DIAGNOSTIC METRICS
# ------------------------------------------------------------------
print("=" * 70)
print("7. SECONDARY / DIAGNOSTIC METRICS")
print("=" * 70)

secondary_results = {}
secondary_metrics = [
    ("ad_impressions", "mean", "Ad impressions/user"),
    ("ad_completion_rate", "mean", "Ad completion rate"),
    ("total_revenue_usd", "mean", "Total revenue/user (ads+IAP)"),
    ("iap_revenue_usd", "mean", "IAP revenue/user"),
    ("day1_sessions", "mean", "Day-1 sessions"),
    ("uninstall_after_ad_flag", "mean", "Uninstall-after-ad rate"),
]

for col, agg, label in secondary_metrics:
    c_vals = df.loc[df.experiment_group == "control", col]
    t_vals = df.loc[df.experiment_group == "treatment", col]
    c_mean, t_mean = c_vals.mean(), t_vals.mean()
    t_stat, p_val = stats.ttest_ind(t_vals, c_vals, equal_var=False)
    rel_change = (t_mean - c_mean) / c_mean * 100 if c_mean != 0 else np.nan
    sig = "***" if p_val < 0.05 else ""
    print(f"{label:32s} control={c_mean:8.4f}  treatment={t_mean:8.4f}  "
          f"rel_change={rel_change:+7.2f}%  p={p_val:.5f} {sig}")
    secondary_results[col] = {
        "control_mean": round(c_mean, 4),
        "treatment_mean": round(t_mean, 4),
        "rel_change_pct": round(rel_change, 2),
        "p_value": p_val,
    }
results["secondary_metrics"] = secondary_results
print()

# ------------------------------------------------------------------
# 8. SEGMENTATION / HETEROGENEITY ANALYSIS
#    (with Bonferroni correction for multiple comparisons)
# ------------------------------------------------------------------
print("=" * 70)
print("8. SEGMENTATION / HETEROGENEITY ANALYSIS (Primary metric: ad revenue)")
print("=" * 70)

segment_cols = ["platform", "country_tier", "user_type"]
segment_results = []

n_tests = sum(df[c].nunique() for c in segment_cols)
bonferroni_alpha = 0.05 / n_tests
print(f"Number of segment tests: {n_tests}  ->  Bonferroni-adjusted alpha = {bonferroni_alpha:.5f}")
print()

for col in segment_cols:
    print(f"--- By {col} ---")
    for level in sorted(df[col].unique()):
        sub = df[df[col] == level]
        c_vals = sub.loc[sub.experiment_group == "control", "ad_revenue_usd"]
        t_vals = sub.loc[sub.experiment_group == "treatment", "ad_revenue_usd"]
        if len(c_vals) < 30 or len(t_vals) < 30:
            continue
        c_mean, t_mean = c_vals.mean(), t_vals.mean()
        rel_lift_seg = (t_mean - c_mean) / c_mean * 100 if c_mean != 0 else np.nan
        t_stat_seg, p_seg = stats.ttest_ind(t_vals, c_vals, equal_var=False)
        sig_raw = p_seg < 0.05
        sig_bonf = p_seg < bonferroni_alpha
        flag = "**" if sig_bonf else ("*" if sig_raw else "")
        print(f"  {level:20s} n_c={len(c_vals):5d} n_t={len(t_vals):5d}  "
              f"control=${c_mean:.4f}  treatment=${t_mean:.4f}  "
              f"lift={rel_lift_seg:+6.2f}%  p={p_seg:.5f} {flag}")
        segment_results.append({
            "dimension": col, "segment": level,
            "n_control": len(c_vals), "n_treatment": len(t_vals),
            "control_mean": round(c_mean, 4), "treatment_mean": round(t_mean, 4),
            "rel_lift_pct": round(rel_lift_seg, 2), "p_value": p_seg,
            "sig_raw_005": bool(sig_raw), "sig_bonferroni": bool(sig_bonf),
        })
    print()

results["segmentation"] = segment_results
results["bonferroni_alpha"] = bonferroni_alpha
print("* = significant at raw alpha=0.05   ** = significant after Bonferroni correction")
print()

# ------------------------------------------------------------------
# 9. SAVE RESULTS
# ------------------------------------------------------------------
def convert(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError

with open(os.path.join(os.path.dirname(__file__), "..", "reports", "analysis_results.json"), "w") as f:
    json.dump(results, f, indent=2, default=convert)

print("=" * 70)
print("Analysis complete. Results saved to reports/analysis_results.json")
print("=" * 70)
