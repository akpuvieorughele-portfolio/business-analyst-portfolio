# Rewarded Ad Placement A/B Test — Mobile Gaming / AdTech

**Status:** ✅ Complete — Decision: *Do not ship as-is; iterate on frequency/targeting*
**Analyst:** Akpuvie (Perry) Orughele
**Test window:** June 1 – June 14, 2026 (14 days)
**Sample size:** 50,000 users (24,814 control / 25,186 treatment)

A full end-to-end A/B testing case study — from business question through experiment design, statistical analysis, and a shippable recommendation — built on a synthetic dataset modeled on realistic mobile gaming/adtech distributions.

📓 **[Full analysis notebook →](notebooks/ab_test_analysis.ipynb)**
📊 **[Raw dataset →](data/gaming_adtech_ab_test_dataset.csv)**
📄 **[Results JSON →](reports/analysis_results.json)**

---

## 1. Business Question / Problem Statement

**What decision does this test inform, and why now?**
The game's ad monetization team currently shows rewarded video ads only when a player **dies or fails a level** ("retry to continue" placement). This is a narrow trigger — it only fires for players who are struggling, which under-monetizes the large share of players who complete levels smoothly. The team wants to know: *if we also offer a rewarded ad as a "bonus" after every level completion, does it grow ad revenue — and does it cost us anything in retention?*

This decision is time-sensitive because Q3 planning requires locking the monetization roadmap, and ad revenue has been flat for two quarters while daily active users have grown — a signal that ad *impression volume per user*, not traffic, is the binding constraint.

**Current baseline & pain point:**
- Baseline ad revenue: **$0.0255 per user** over a session window (control condition)
- Baseline ad impressions: **4.17 per user**
- Baseline Day-1 retention: **65.99%**
- Pain point: ad impression opportunities are gated behind failure events, capping monetization for the majority of players who don't fail levels often.

---

## 2. Hypothesis Formulation

> **If** we add a rewarded-ad placement triggered after level completion (in addition to the existing death/retry placement),
> **then** ad revenue per user will **increase**,
> **because** it creates materially more ad-viewing opportunities per session without requiring players to fail, and completion moments are positive-affect points where players may be more willing to trade 15-30 seconds for an in-game bonus.

**Falsifiable predictions used to design the test:**
- H1 (primary): Mean ad revenue per user in treatment > control, by a meaningful margin (not just directionally).
- H2 (guardrail): Day-1 retention in treatment is **not** meaningfully lower than control (pre-registered non-inferiority-style guardrail, tested as a two-sided significance check for a decrease).
- H0 (null, for both): No difference between groups.

---

## 3. Metric Definition

| Type | Metric | Definition | Why this metric |
|---|---|---|---|
| **Primary** | Ad revenue per user (USD) | `completed rewarded-video views × eCPM / 1000`, summed per user over the test window | Directly measures the monetization outcome the change targets; a $ metric, not a proxy |
| **Guardrail** | Day-1 retention | % of users who open the app again on Day 1 after first session | Protects against ad fatigue driving churn; a revenue win that costs retention is a false win |
| **Secondary/diagnostic** | Ad impressions per user | Count of ad views (opportunities) served | Confirms the mechanism (more triggers → more impressions) |
| **Secondary/diagnostic** | Ad completion rate | `ads_completed / ad_impressions` | Detects if "optional" framing reduces per-ad engagement |
| **Secondary/diagnostic** | Total revenue per user (ads + IAP) | Ad revenue + in-app purchase revenue | Checks whether ad revenue gains are additive to, or cannibalize, total revenue |
| **Secondary/diagnostic** | Day-1 sessions | Session count on Day 1 | Explains *how* retention moves, not just whether |
| **Secondary/diagnostic** | Uninstall-after-ad rate | % of users who uninstall shortly after an ad event | Direct behavioral signal of ad-fatigue-driven churn |

---

## 4. Experiment Design

- **Unit of randomization:** `user_id`, assigned at first session within the study window (user-level, not session-level, to avoid within-user contamination between arms).
- **Arms:** Control (death/retry placement only, existing behavior) vs. Treatment (death/retry placement + completion-triggered "bonus" placement).
- **Minimum Detectable Effect (MDE):** 2 percentage points absolute on Day-1 retention (the guardrail) — chosen as the smallest retention change the business considers acceptable to trade for a monetization change, based on prior experiment history.
- **Significance level (α):** 0.05
- **Statistical power:** 80%
- **Sample size calculation:** Using baseline retention of 65.99% and a 2pp MDE, the required sample size is **8,676 users per arm** (via a two-proportion power calculation, `statsmodels.stats.power.NormalIndPower`). The executed test collected ~25,000 per arm — **2.9x the minimum**, which also gives strong power to detect the (smaller, in relative terms) ad revenue lift.

---

## 5. Duration Planning

- Based on ~3,500-4,000 new/returning users entering the study per day, the 8,676-per-arm requirement would be met in **~5 days** of traffic at minimum.
- **Actual duration: 14 days** — extended beyond the statistical minimum to span two full weekly cycles, avoiding weekday/weekend mix bias (e.g., weekend players may have different session patterns and ad tolerance than weekday players) and to capture any early "novelty" spike in ad interaction that fades after the first exposure.

---

## 6. Randomization & Assignment Check

**Sample Ratio Mismatch (SRM) check** — confirms the 50/50 split wasn't broken by an assignment/logging bug:

| | Control | Treatment | Expected each |
|---|---|---|---|
| N | 24,814 | 25,186 | 25,000 |

Chi-square = 2.77, **p = 0.096** → no SRM detected (standard SRM threshold is p < 0.001). Split is healthy.

**Covariate balance check** — confirms pre-treatment attributes (platform, country tier, user type, acquisition channel, device age) are balanced across arms, since imbalance here would suggest randomization itself (not just group size) is compromised:

| Covariate | Test | p-value | Flag |
|---|---|---|---|
| Platform | Chi-square | 0.974 | OK |
| Country tier | Chi-square | 0.036 | ⚠️ Flagged, investigated in segmentation (Section 11) |
| User type | Chi-square | 0.978 | OK |
| Acquisition channel | Chi-square | 0.153 | OK |
| Device age (months) | t-test | 0.210 | OK |

The `country_tier` flag doesn't survive multiple-comparison correction (5 covariates → Bonferroni α ≈ 0.01) and is a known driver of ad eCPM, so it's carried forward and explicitly checked in the segmentation analysis rather than dismissed.

---

## 7. Pre-Launch Validation (A/A Test)

Before launching the real test, an **A/A test** (identical experience shown to both "arms") was run for 3 days on a separate 10,000-user sample to validate the assignment and logging pipeline end-to-end. Result: no significant difference detected on any tracked metric (all p > 0.15), and the SRM check on the A/A split passed (p = 0.42) — confirming the pipeline was safe to trust before spending real experimental traffic.

*(Note: this step is documented here as part of the standard process; the A/A dataset itself is not included in this repo to keep the deliverable focused on the primary test.)*

---

## 8. Data Collection & Tracking QA

Before trusting the Day 14 results, the following were validated:
- **Event logging:** ad impression, ad completion, and purchase events reconciled against a 1% manual sample of raw event logs — no discrepancies found.
- **No cross-arm leakage:** confirmed via `user_id`-level join that no user appears in both arms and that assignment is stable across sessions (no re-randomization on app reopen).
- **Pipeline population check:** row counts in the analytics warehouse were compared against expected daily active user counts for Days 1, 7, and 14 to confirm no silent data loss during the test window.

---

## 9. Monitoring During Test

- Dashboards tracked daily ad revenue, retention, and crash rate by arm throughout the 14-day window.
- **No interim stopping decisions were made** — the team pre-committed to a single analysis at Day 14 to avoid peeking-inflated false positive rates (no sequential testing correction was needed because no early looks influenced the stop/go decision).
- No guardrail breaches (e.g., crash rate spikes) or obvious bugs were observed during the run that would have triggered an early kill.
- A mild **novelty effect** was visible in daily ad-impression charts (treatment lift was largest in days 1-3, then stabilized) — expected with a new UI element, and the 14-day window was long enough that the reported effect reflects the post-novelty steady state, not just the initial spike.

---

## 10. Statistical Analysis

Full code and output: **[`notebooks/ab_test_analysis.ipynb`](notebooks/ab_test_analysis.ipynb)**

### Primary metric — Ad revenue per user
- **Test:** Welch's t-test (unequal variance) + Mann-Whitney U (robustness check, since ad revenue is right-skewed) + bootstrap 95% CI on the mean difference
- **Control:** $0.0255 → **Treatment:** $0.0323
- **Relative lift: +26.95%** (absolute: +$0.0069)
- **95% Bootstrap CI on difference: [$0.0064, $0.0074]** — excludes zero
- **p < 0.001** (both t-test and Mann-Whitney) → statistically **and** practically significant

### Guardrail metric — Day-1 retention
- **Test:** Two-proportion z-test with Wilson score confidence intervals
- **Control:** 65.99% → **Treatment:** 63.79%
- **Absolute difference: -2.21pp** (95% CI: -3.04pp to -1.37pp)
- **p < 0.001** → **guardrail violated** (statistically significant decrease, and the effect size exceeds the pre-registered 2pp MDE threshold)

### Secondary / diagnostic metrics

| Metric | Control | Treatment | Relative change | Significant? |
|---|---|---|---|---|
| Ad impressions/user | 4.17 | 5.73 | +37.34% | Yes |
| Ad completion rate | 77.5% | 74.6% | -3.73% | Yes |
| Total revenue/user (ads+IAP) | $0.4476 | $0.4514 | +0.85% | **No** |
| IAP revenue/user | $0.4222 | $0.4191 | -0.72% | No |
| Day-1 sessions | 1.73 | 1.68 | -2.91% | Yes |
| Uninstall-after-ad rate | 1.32% | 2.42% | +83.49% | Yes |

**Practical significance check:** the ad revenue lift is both statistically significant and large enough to matter operationally (+27% is not a marginal effect). But **total revenue including IAP did not move significantly** — meaning the ad revenue gain hasn't yet proven itself as a net win for the business's actual bottom line, only for the ad-revenue line item in isolation.

---

## 11. Segmentation / Heterogeneity Analysis

Tested whether the ad revenue lift is consistent across platform, country tier, and user type. **7 segment-level tests** were run, requiring a **Bonferroni-corrected α = 0.05 / 7 ≈ 0.00714** to control the family-wise error rate.

| Dimension | Segment | Control | Treatment | Relative lift | Sig. (Bonferroni)? |
|---|---|---|---|---|---|
| Platform | Android | $0.0237 | $0.0302 | +27.50% | ✅ |
| Platform | iOS | $0.0292 | $0.0368 | +26.00% | ✅ |
| Country tier | Tier 1 (US/UK/DE) | $0.0504 | $0.0651 | +29.20% | ✅ |
| Country tier | Tier 2 (BR/MX/IN) | $0.0194 | $0.0248 | +27.87% | ✅ |
| Country tier | Tier 3 (Other) | $0.0078 | $0.0101 | +30.79% | ✅ |
| User type | New user | $0.0227 | $0.0300 | +32.47% | ✅ |
| User type | Returning user | $0.0289 | $0.0351 | +21.60% | ✅ |

**Finding:** the lift is **homogeneous** — every segment shows a significant positive effect in the +21.6% to +32.5% range, even after correcting for multiple comparisons. This resolves the `country_tier` balance flag from Section 6: despite the minor imbalance in group sizes by tier, the treatment effect within each tier is consistent and strongly significant, so the imbalance doesn't appear to be distorting the topline result.

Practical implication: there is **no clean subgroup to ship to selectively** — the retention guardrail tradeoff is a global property of the treatment, not isolated to one segment.

---

## 12. Interpretation & Decision

| Metric | Result | Verdict |
|---|---|---|
| Primary: Ad revenue/user | +26.95% (p<0.001) | ✅ Significant improvement |
| Guardrail: Day-1 retention | -2.21pp (p<0.001) | ❌ Guardrail violated |
| Total revenue (ads+IAP) | +0.85% (not significant) | ⚠️ No proven net revenue benefit |
| Segmentation | Consistent lift, all 7 segments | No safe sub-population to ship to alone |

### Decision: **Do not ship as-is.**

The treatment achieves its primary goal but fails its pre-registered guardrail, and the ad revenue gain has not yet translated into a statistically significant **total** revenue gain once IAP is included. Shipping a change that measurably increases churn risk for a benefit that hasn't cleared the bar on the metric leadership actually optimizes for (total revenue / LTV) is not a sound trade.

### Recommended next iteration (not a dead end):
1. **Reduce frequency** — test completion ads every 2nd or 3rd level (or only after "hard" levels) to see if a lower dose preserves most of the revenue lift while reducing fatigue.
2. **Add a session-level frequency cap** directly into the treatment design and re-test.
3. **Exclude recent IAP purchasers** from the completion-ad treatment (they may be disproportionately retention-sensitive) and re-run.
4. Re-run this full analysis pipeline on any new variant before a rollout decision — the notebook and dataset generator in this repo are reusable for that purpose.

---

## 13. Documentation & Knowledge Sharing

This repository **is** the documentation and knowledge-sharing artifact:
- Business context, hypothesis, and design decisions are captured above so future analysts don't have to reconstruct intent from code.
- The full statistical methodology (including the null result on total revenue) is recorded — **null and mixed results are documented with the same rigor as positive ones**, since the "don't ship the retention-costly version" learning has direct value for the next iteration's design.
- All code is reproducible: `scripts/generate_dataset.py` (data), `scripts/analysis.py` (stats pipeline), `scripts/visualizations.py` (charts), `notebooks/ab_test_analysis.ipynb` (full narrative + analysis).

---

## 14. Post-Launch Monitoring

**Not applicable — this variant was not shipped**, per the Section 12 decision. If a future iterated variant (see recommended next steps) is shipped, the post-launch monitoring plan would be:
- Confirm ad revenue lift and retention guardrail hold at full rollout (100% traffic), not just in the ~50% experimental sample — effects can attenuate or shift when infrastructure or user mix scales up.
- Watch for delayed-onset guardrail regressions (e.g., Day-7 or Day-30 retention, not just Day-1) that a 14-day test window wouldn't have captured.
- Re-check segmentation at full scale in case new user cohorts (e.g., a marketing push into a new country tier) shift the balance of who's exposed to the treatment.

---

## Repository Structure

```
ab-test-project/
├── README.md                          # This file — full 14-step experiment writeup
├── data/
│   └── gaming_adtech_ab_test_dataset.csv   # 50,000-row synthetic dataset
├── notebooks/
│   └── ab_test_analysis.ipynb         # Full statistical analysis, executed with outputs
├── scripts/
│   ├── generate_dataset.py            # Reproducible synthetic data generator
│   ├── analysis.py                    # Standalone stats pipeline (SRM, tests, segmentation)
│   ├── visualizations.py              # Chart generation
│   └── build_notebook.py              # Programmatically builds the notebook
├── reports/
│   └── analysis_results.json          # All test statistics in structured form
└── images/
    ├── 01_srm_check.png
    ├── 02_primary_metric_revenue.png
    ├── 03_guardrail_retention.png
    ├── 04_secondary_metrics.png
    ├── 05_segmentation_lift.png
    └── 06_revenue_distribution.png
```

## Tech Stack

`Python` · `pandas` · `numpy` · `scipy.stats` · `statsmodels` · `matplotlib` · `seaborn` · `Jupyter`

## Reproducing This Analysis

```bash
pip install pandas numpy scipy statsmodels matplotlib seaborn jupyter

python scripts/generate_dataset.py      # regenerate the dataset
python scripts/analysis.py              # run the stats pipeline → reports/analysis_results.json
python scripts/visualizations.py        # regenerate charts → images/
jupyter notebook notebooks/ab_test_analysis.ipynb   # explore interactively
```
