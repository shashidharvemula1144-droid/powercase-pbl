"""
Step 1 - Generate a synthetic consumer survey for PowerCase (phone case with built-in battery).

Why synthetic: the real survey takes about a month. Generating realistic data first lets us build
and test the whole analytics pipeline now, then plug in the real responses later.

What is built in on purpose (so the algorithms have something to find):
  * Battery anxiety, hours away from a charger and number of problems raise the chance of buying.
  * Income, iPhone ownership and price sensitivity drive how much a person will pay.
  * Features cluster into three "wish groups" (power, style, rugged) so association rules exist.
  * Four groups of respondents behave differently (used later by clustering).

Noise added on purpose (so the cleaning step has real work):
  * missing values, duplicate submissions, outliers, impossible values, inconsistent labels,
    mixed date formats, careless "speeder" respondents and about 6% flipped purchase answers.

Run:  python 02_generate_survey_data.py      -> data/powercase_survey_raw.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
(HERE / "data").mkdir(exist_ok=True)
rng = np.random.default_rng(2026)
N = 1500

# ---------------------------------------------------------------- demographics
occupation = rng.choice(["Student", "Employee", "Self-employed", "Other"], N, p=[.24, .55, .14, .07])
age = np.where(occupation == "Student", rng.normal(21, 2.2, N),
      np.where(occupation == "Employee", rng.normal(34, 8, N),
      np.where(occupation == "Self-employed", rng.normal(38, 9, N), rng.normal(40, 12, N))))
age = np.clip(np.round(age), 18, 68).astype(int)
gender = rng.choice(["Male", "Female"], N, p=[.54, .46])
emirate = rng.choice(["Dubai", "Abu Dhabi", "Sharjah", "Ajman", "Other emirates"], N, p=[.42, .2, .2, .06, .12])

income_med = {"Student": 2500, "Employee": 11000, "Self-employed": 15000, "Other": 6000}
income_sd = {"Student": .45, "Employee": .55, "Self-employed": .70, "Other": .6}
income = np.array([rng.lognormal(np.log(income_med[o]), income_sd[o]) for o in occupation])
income = np.clip(np.round(income, -2), 800, 80000)
z_inc = (np.log(income) - np.log(8000)) / 1.0

p_iphone = np.clip(0.50 + 0.18 * z_inc, 0.15, 0.9)
r = rng.random(N)
phone_brand = np.where(r < p_iphone, "iPhone", np.where(rng.random(N) < 0.72, "Samsung", "Other Android"))
phone_age = np.clip(rng.gamma(2.2, 0.9, N), 0.2, 6).round(1)

# ---------------------------------------------------------------- battery habits
screen = np.clip(rng.gamma(6, 1.0, N) + (age < 25) * 1.3 - (age > 45) * 1.2, 1.5, 14).round(1)
low_batt = rng.poisson(np.clip(0.4 + 0.28 * screen + 0.35 * (phone_age > 2.5), 0.2, None))
anxiety = np.clip(np.round(1.0 + 0.16 * low_batt + 0.12 * screen + rng.normal(0, 0.8, N)), 1, 5).astype(int)
trips = rng.poisson(np.clip(1.5 + 1.2 * z_inc + 1.0 * (occupation == "Self-employed"), 0.3, None))
away = np.clip(rng.gamma(3, 1.3, N) + 1.5 * (occupation == "Employee") + 1.0 * (occupation == "Student"), 1, 16).round(1)
carries_pb = rng.random(N) < np.clip(0.12 + 0.11 * anxiety, 0, 0.8)
pb_spend = np.where(carries_pb, np.clip(rng.lognormal(np.log(90), 0.5, N), 20, 600), 0).round(0)
uses_case = rng.random(N) < 0.9
case_spend = np.where(uses_case, np.clip(rng.lognormal(np.log(60) + 0.25 * z_inc, 0.6, N), 10, 600), 0).round(0)

# ---------------------------------------------------------------- problems faced (multi-select)
def sig(x): return 1 / (1 + np.exp(-x))
ch_p = {
    "Phone died while navigating or on a ride app": sig(-2.0 + 0.15 * low_batt + 0.10 * trips),
    "Phone died before the end of the workday": sig(-1.8 + 0.18 * low_batt + 0.12 * away + 0.5 * (occupation == "Employee")),
    "No outlet or charger when I needed one": sig(-2.2 + 0.20 * away + 0.1 * anxiety),
    "Forgot or lost my power bank": np.where(carries_pb, 0.40, 0.05),
    "Power bank and cables too bulky to carry": np.where(carries_pb, 0.35, 0.06),
    "Battery drains fast while travelling": sig(-2.0 + 0.25 * trips + 0.1 * screen),
    "Missed an important call or OTP, phone was dead": sig(-3.0 + 0.45 * anxiety),
}
ch_cols = list(ch_p)
ch_mat = np.column_stack([rng.random(N) < ch_p[c] for c in ch_cols])
challenge_count = ch_mat.sum(axis=1)

# ---------------------------------------------------------------- desired features (multi-select) with 3 wish groups
power = rng.normal(0, 1, N) + 0.35 * (anxiety - 3)
style = rng.normal(0, 1, N) + 0.5 * (phone_brand == "iPhone") + 0.2 * (gender == "Female")
rugged = rng.normal(0, 1, N) + 0.3 * (occupation == "Self-employed") + 0.2 * trips / 3
ft_p = {
    "Slim design": sig(0.0 + 0.9 * style + 0.9 * ch_mat[:, 4]),
    "High capacity (4,000 mAh or more)": sig(-0.3 + 1.0 * power + 0.7 * ch_mat[:, 2] + 0.6 * ch_mat[:, 5]),
    "Fast charging": sig(-0.1 + 0.9 * power + 0.6 * ch_mat[:, 6] + 0.5 * ch_mat[:, 2]),
    "Wireless or MagSafe pass-through": sig(-1.2 + 1.0 * style + 0.5 * (phone_brand == "iPhone")),
    "Drop protection": sig(-0.5 + 1.0 * rugged + 0.6 * ch_mat[:, 0]),
    "LED battery indicator": sig(-1.4 + 0.7 * power + 0.8 * ch_mat[:, 3] + 0.5 * ch_mat[:, 6]),
    "Card slot or kickstand": sig(-1.7 + 0.9 * rugged + 0.4 * ch_mat[:, 0]),
    "Eco-friendly material": sig(-1.9 + 0.8 * style),
    "2-year warranty": sig(-1.0 + 0.6 * rugged + 0.3 * power + 0.4 * ch_mat[:, 1]),
}
ft_cols = list(ft_p)
ft_mat = np.column_stack([rng.random(N) < ft_p[c] for c in ft_cols])
no_pick = ft_mat.sum(axis=1) == 0
ft_mat[no_pick, rng.integers(0, len(ft_cols), no_pick.sum())] = True
feat_count = ft_mat.sum(axis=1)

def pick_top(row):
    idx = np.flatnonzero(row)
    return ft_cols[rng.choice(idx)]
top_feature = [pick_top(row) for row in ft_mat]

# ---------------------------------------------------------------- other concept questions
concern = np.array([rng.choice(["Bulk or weight", "Price", "Safety (heat, swelling)", "Fits only one phone model", "Brand trust"],
                               p=[.32, .22 + .0, .22, .14, .10]) for _ in range(N)])
price_sens = np.clip(np.round(3.2 - 0.55 * z_inc + 0.3 * (occupation == "Student") + rng.normal(0, 0.9, N)), 1, 5).astype(int)
brand_imp = np.clip(np.round(3.0 + 0.25 * z_inc + 0.4 * (phone_brand == "iPhone") + rng.normal(0, 0.9, N)), 1, 5).astype(int)
acc_year = rng.poisson(np.clip(1.0 + 0.9 * z_inc + 0.4 * (age < 30), 0.2, None))

fmt = np.array([rng.choice(["Built-in battery case", "Detachable magnetic pack", "Normal power bank"],
                           p=[.40, .42, .18] if b == "iPhone" else [.40, .09, .51]) for b in phone_brand])
capacity = np.array([rng.choice(["2,000-3,000 mAh", "3,000-4,000 mAh", "4,000-5,000 mAh", "5,000+ mAh"],
                                p=[.10, .32, .34, .24] if a >= 4 else [.24, .40, .24, .12]) for a in anxiety])
def _norm(p):
    p = np.array(p, float)
    return p / p.sum()
channel = np.array([rng.choice(["Amazon.ae", "Noon", "Mall electronics store", "Instagram or TikTok shop", "Telecom store", "Brand website"],
                               p=_norm([.29, .24, .17, .14 + .06 * (a < 28), .10, .06])) for a in age])
discovery = np.array([rng.choice(["Instagram", "TikTok", "YouTube reviews", "Friends and family", "In-store"],
                                 p=[.30, .26, .16, .12, .16] if a < 30 else [.22, .08, .22, .22, .26]) for a in age])

# ---------------------------------------------------------------- willingness to pay (regression target)
log_wtp = (np.log(165) + 0.26 * z_inc + 0.12 * (phone_brand == "iPhone") + 0.07 * (anxiety - 3)
           - 0.09 * (price_sens - 3) + 0.0015 * case_spend + 0.04 * (brand_imp - 3) + rng.normal(0, 0.24, N))
wtp = np.clip(np.round(np.exp(log_wtp) / 5) * 5, 40, 500)

# ---------------------------------------------------------------- purchase intent (classification target)
z = (-2.1 + 0.50 * (anxiety - 3) + 0.10 * (screen - 6) + 0.10 * low_batt + 0.17 * challenge_count
     + 0.05 * (away - 5) - 0.40 * (price_sens - 3) + 0.30 * (phone_brand == "iPhone")
     + 0.28 * (wtp >= 150) + 0.10 * (feat_count - 3) - 0.60 * (age >= 45) + 0.10 * (acc_year - 1)
     + rng.normal(0, 0.55, N))
q_yes, q_maybe = np.quantile(z, [0.90, 0.68])
intent = np.where(z >= q_yes, "Yes", np.where(z >= q_maybe, "Maybe", "No"))

# careless respondents: random answers, very short completion time
careless = rng.random(N) < 0.04
intent = np.where(careless, rng.choice(["Yes", "Maybe", "No"], N), intent)
# label noise: some people say one thing and mean another
flip = rng.random(N) < 0.06
intent = np.where(flip, rng.choice(["Yes", "Maybe", "No"], N), intent)

preorder = np.where(intent == "Yes", rng.random(N) < 0.5, np.where(intent == "Maybe", rng.random(N) < 0.12, rng.random(N) < 0.01))
completion = np.where(careless, rng.uniform(0.4, 1.2, N), np.clip(rng.gamma(6, 1.1, N), 2.5, 40)).round(1)
dates = pd.to_datetime("2026-09-01") + pd.to_timedelta(rng.integers(0, 40, N), unit="D")

df = pd.DataFrame({
    "respondent_id": [f"R{i:04d}" for i in range(1, N + 1)],
    "responded_at": dates, "completion_min": completion,
    "age": age, "gender": gender, "emirate": emirate, "occupation": occupation, "monthly_income_aed": income,
    "phone_brand": phone_brand, "phone_age_years": phone_age,
    "screen_hrs": screen, "low_battery_per_week": low_batt, "battery_anxiety": anxiety, "hrs_away_from_charger": away,
    "carries_power_bank": np.where(carries_pb, "Yes", "No"), "powerbank_spend_aed": pb_spend,
    "uses_case": np.where(uses_case, "Yes", "No"), "case_spend_aed": case_spend,
    "challenges": [";".join(np.array(ch_cols)[row]) for row in ch_mat],
    "desired_features": [";".join(np.array(ft_cols)[row]) for row in ft_mat],
    "top_feature": top_feature, "top_concern": concern, "preferred_format": fmt, "preferred_capacity": capacity,
    "purchase_channel": channel, "discovery_channel": discovery,
    "price_sensitivity": price_sens, "brand_importance": brand_imp, "accessories_per_year": acc_year,
    "trips_per_year": trips, "max_price_aed": wtp, "preorder_signup": np.where(preorder, "Yes", "No"), "intent": intent,
})

# ================================================================ inject realistic mess
def hit(col, frac, value=np.nan, seed=0):
    idx = df.sample(frac=frac, random_state=seed).index
    df[col] = df[col].astype(object)
    df.loc[idx, col] = value

# missing values
for col, frac, s in [("monthly_income_aed", .07, 1), ("age", .03, 2), ("screen_hrs", .03, 3), ("max_price_aed", .04, 4),
                     ("phone_age_years", .04, 5), ("emirate", .02, 6), ("occupation", .02, 7), ("hrs_away_from_charger", .03, 8),
                     ("case_spend_aed", .03, 9)]:
    hit(col, frac, np.nan, s)

# outliers and impossible values
df["monthly_income_aed"] = df["monthly_income_aed"].astype(object)
df.loc[df.sample(n=6, random_state=11).index, "monthly_income_aed"] = 250000
df.loc[df.sample(n=5, random_state=12).index, "monthly_income_aed"] = 0
df["age"] = df["age"].astype(object)
df.loc[df.sample(n=5, random_state=13).index, "age"] = 120
df.loc[df.sample(n=3, random_state=14).index, "age"] = 9
df["screen_hrs"] = df["screen_hrs"].astype(object)
df.loc[df.sample(n=5, random_state=15).index, "screen_hrs"] = 26
df["max_price_aed"] = df["max_price_aed"].astype(object)
df.loc[df.sample(n=6, random_state=16).index, "max_price_aed"] = 5000
df.loc[df.sample(n=4, random_state=17).index, "max_price_aed"] = 0
df.loc[df.sample(n=4, random_state=18).index, "low_battery_per_week"] = 60
df.loc[df.sample(n=4, random_state=19).index, "accessories_per_year"] = 90

# inconsistent labels
em = df["emirate"].astype(object)
e_idx = df[df["emirate"] == "Dubai"].sample(frac=.12, random_state=20).index
em.loc[e_idx] = rng.choice(["dubai ", "DXB", "DUBAI", "Dubai."], len(e_idx))
df["emirate"] = em
b_idx = df[df["phone_brand"] == "iPhone"].sample(frac=.15, random_state=21).index
df.loc[b_idx, "phone_brand"] = rng.choice(["iphone", "Apple", "IPHONE", "Iphone "], len(b_idx))
s_idx = df[df["phone_brand"] == "Samsung"].sample(frac=.10, random_state=22).index
df.loc[s_idx, "phone_brand"] = rng.choice(["samsung", "SAMSUNG"], len(s_idx))
for col, seed in [("carries_power_bank", 23), ("uses_case", 24), ("preorder_signup", 25)]:
    idx = df[df[col] == "Yes"].sample(frac=.1, random_state=seed).index
    df.loc[idx, col] = rng.choice(["yes", "Y", "YES", "1"], len(idx))
    idx = df[df[col] == "No"].sample(frac=.1, random_state=seed + 10).index
    df.loc[idx, col] = rng.choice(["no", "N", "NO", "0"], len(idx))

# mixed date formats stored as text
d = pd.to_datetime(df["responded_at"])
fmt_mask = rng.random(N) < 0.5
df["responded_at"] = np.where(fmt_mask, d.dt.strftime("%d/%m/%Y"), d.dt.strftime("%Y-%m-%d"))

# duplicate submissions (same person submitted twice)
dups = df.sample(n=45, random_state=30)
df = pd.concat([df, dups], ignore_index=True).sample(frac=1, random_state=31).reset_index(drop=True)

out = HERE / "data" / "powercase_survey_raw.csv"
df.to_csv(out, index=False)
print(f"Saved {out.name}: {df.shape[0]} rows, {df.shape[1]} columns")
print(df["intent"].value_counts(normalize=True).round(3).to_dict())
