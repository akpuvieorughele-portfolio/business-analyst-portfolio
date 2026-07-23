"""
Generate a synthetic A/B test dataset for a gaming/adtech company.

SCENARIO:
A mobile puzzle game currently shows rewarded video ads only when a player
dies/fails a level ("Control - Death Placement"). Product wants to test a new
placement that offers a rewarded ad after every level COMPLETION, framed as a
"bonus reward" ("Treatment - Completion Placement"), hypothesizing it will
increase ad impressions/revenue per user without hurting Day-1 retention or
session length.

Business questions this dataset lets you A/B test:
1. Does the treatment increase ad impressions per user and eCPM-driven revenue?
2. Does it hurt Day-1 retention or session count (ad fatigue / annoyance)?
3. Is the revenue lift statistically significant and practically meaningful net
   of any retention loss (guardrail metric)?
4. Are effects consistent across platform (iOS/Android), country tier, and
   new-vs-returning users (heterogeneous treatment effects)?

Unit of randomization: user_id (assigned at first session in the study window)
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(42)

N = 50000

# ---------------------------------------------------------------
# 1. Experiment assignment (50/50 randomization, slight imbalance
#    to mimic real-world SRM-free but not perfectly exact split)
# ---------------------------------------------------------------
group = rng.choice(
    ["control", "treatment"], size=N, p=[0.5, 0.5]
)

# ---------------------------------------------------------------
# 2. User attributes (pre-treatment covariates, independent of assignment
#    since this is randomized - but we add realistic distributions)
# ---------------------------------------------------------------
platforms = rng.choice(["Android", "iOS"], size=N, p=[0.68, 0.32])

country_tiers = rng.choice(
    ["Tier1_US_UK_DE", "Tier2_BR_MX_IN", "Tier3_Other"],
    size=N,
    p=[0.30, 0.40, 0.30],
)

user_type = rng.choice(
    ["new_user", "returning_user"], size=N, p=[0.55, 0.45]
)

acquisition_channel = rng.choice(
    ["organic", "paid_social", "paid_search", "influencer", "cross_promo"],
    size=N,
    p=[0.35, 0.30, 0.15, 0.10, 0.10],
)

device_age_months = np.clip(rng.exponential(scale=14, size=N), 0, 60).round(1)

# Study dates: a 14-day experiment window in June 2026
start_date = datetime(2026, 6, 1)
assignment_date = start_date + pd.to_timedelta(
    rng.integers(0, 14, size=N), unit="D"
)

# ---------------------------------------------------------------
# 3. Simulate underlying "true" behavior with a treatment effect
#    baked in, plus realistic covariate effects & noise.
# ---------------------------------------------------------------

# --- baseline session count (day 0) ---
# returning users play more; iOS slightly higher engagement typical in games
base_sessions = (
    2.3
    + (user_type == "returning_user") * 0.9
    + (platforms == "iOS") * 0.15
    + (country_tiers == "Tier1_US_UK_DE") * 0.25
)
sessions_day0 = rng.poisson(lam=np.clip(base_sessions, 0.3, None))

# --- levels completed day 0 ---
base_levels = 3.0 + (user_type == "returning_user") * 1.5 + sessions_day0 * 0.8
levels_completed_day0 = rng.poisson(lam=np.clip(base_levels, 0.2, None))

# --- avg session length (minutes) ---
session_length_min = np.clip(
    rng.normal(
        loc=8.5 + (user_type == "returning_user") * 1.2,
        scale=3.0,
        size=N,
    ),
    0.5,
    None,
).round(2)

# --- TREATMENT EFFECT: ad impressions per user ---
# Control: ads mostly on death events (fewer natural triggers)
# Treatment: ads after every level completion (more triggers) -> real lift
base_ad_impressions = 1.4 + levels_completed_day0 * 0.35 + sessions_day0 * 0.25
treatment_lift_impressions = np.where(group == "treatment", 1.55, 0.0)  # ~+35-45% lift
ad_impressions = rng.poisson(
    lam=np.clip(base_ad_impressions + treatment_lift_impressions, 0.1, None)
)

# --- ad completion rate (rewarded video watched to completion) ---
# Treatment ads are "optional bonus" framing -> slightly lower completion rate
# per-impression (some users skip more readily when not needed to continue)
base_completion_rate = 0.82 - (platforms == "Android") * 0.03
treatment_completion_penalty = np.where(group == "treatment", -0.05, 0.0)
completion_rate_true = np.clip(
    base_completion_rate + treatment_completion_penalty + rng.normal(0, 0.05, N),
    0.3,
    0.98,
)
ads_completed = rng.binomial(ad_impressions, completion_rate_true)

# --- eCPM varies by country tier and platform (typical adtech pattern) ---
ecpm_base = np.select(
    [
        country_tiers == "Tier1_US_UK_DE",
        country_tiers == "Tier2_BR_MX_IN",
        country_tiers == "Tier3_Other",
    ],
    [14.0, 5.5, 2.2],
)
ecpm = np.clip(
    ecpm_base * (1.0 + (platforms == "iOS") * 0.18) + rng.normal(0, 1.2, N),
    0.3,
    None,
).round(2)

# --- ad revenue = completed rewarded views * eCPM / 1000 ---
ad_revenue_usd = (ads_completed * ecpm / 1000).round(4)

# --- IAP revenue (in-app purchases) - treatment should NOT meaningfully
#     change this except a tiny possible cannibalization effect ---
iap_propensity = rng.random(N) < (
    0.045 + (user_type == "returning_user") * 0.02 + (country_tiers == "Tier1_US_UK_DE") * 0.02
)
iap_amount = np.where(
    iap_propensity,
    np.clip(rng.lognormal(mean=1.6, sigma=0.9, size=N), 0.99, 199.99),
    0.0,
).round(2)
# tiny cannibalization: treatment users slightly less likely to spend real money
# since they're getting "free" rewards more often (~-3% relative)
iap_amount = np.where(
    (group == "treatment") & iap_propensity,
    (iap_amount * rng.uniform(0.90, 1.02, N)).round(2),
    iap_amount,
)

# --- GUARDRAIL METRIC: Day-1 retention ---
# Treatment slightly reduces retention due to ad fatigue (ads after every win
# can feel like "punishment" for completing a level) -> small negative effect
base_retention_logit = (
    -0.6
    + (user_type == "returning_user") * 1.1
    + (platforms == "iOS") * 0.12
    + (country_tiers == "Tier1_US_UK_DE") * 0.15
    + sessions_day0 * 0.18
    + levels_completed_day0 * 0.05
)
treatment_retention_effect = np.where(group == "treatment", -0.12, 0.0)  # small negative
retention_logit = base_retention_logit + treatment_retention_effect + rng.normal(0, 0.4, N)
retention_prob = 1 / (1 + np.exp(-retention_logit))
day1_retained = (rng.random(N) < retention_prob).astype(int)

# --- Day-1 sessions for retained users (0 if not retained) ---
day1_sessions = np.where(
    day1_retained == 1,
    rng.poisson(lam=np.clip(sessions_day0 * 0.85, 0.2, None)),
    0,
)

# --- ad-related churn flag: did user uninstall right after an ad (proxy) ---
uninstall_after_ad = (
    (rng.random(N) < (0.015 + np.where(group == "treatment", 0.008, 0.0)))
    & (ad_impressions > 0)
).astype(int)

# --- total revenue ---
total_revenue_usd = (ad_revenue_usd + iap_amount).round(4)

# ---------------------------------------------------------------
# 4. Assemble DataFrame
# ---------------------------------------------------------------
df = pd.DataFrame(
    {
        "user_id": [f"U{100000+i}" for i in range(N)],
        "assignment_date": assignment_date.strftime("%Y-%m-%d"),
        "experiment_group": group,  # control / treatment
        "platform": platforms,
        "country_tier": country_tiers,
        "user_type": user_type,
        "acquisition_channel": acquisition_channel,
        "device_age_months": device_age_months,
        "sessions_day0": sessions_day0,
        "levels_completed_day0": levels_completed_day0,
        "avg_session_length_min": session_length_min,
        "ad_impressions": ad_impressions,
        "ads_completed": ads_completed,
        "ad_completion_rate": np.where(
            ad_impressions > 0, (ads_completed / np.maximum(ad_impressions, 1)).round(3), 0.0
        ),
        "ecpm_usd": ecpm,
        "ad_revenue_usd": ad_revenue_usd,
        "iap_revenue_usd": iap_amount,
        "total_revenue_usd": total_revenue_usd,
        "day1_retained": day1_retained,
        "day1_sessions": day1_sessions,
        "uninstall_after_ad_flag": uninstall_after_ad,
    }
)

# Shuffle rows so group order isn't sorted
df = df.sample(frac=1, random_state=7).reset_index(drop=True)

out_path = os.path.join(os.path.dirname(__file__), "..", "data", "gaming_adtech_ab_test_dataset.csv")
df.to_csv(out_path, index=False)

print(df.shape)
print(df["experiment_group"].value_counts())
print(df.groupby("experiment_group")[["ad_impressions", "ad_revenue_usd", "day1_retained", "total_revenue_usd"]].mean())
