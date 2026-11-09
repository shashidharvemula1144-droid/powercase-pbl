"""
Step 2 - Data cleaning and transformation (10 marks).
Every action is written to data/cleaning_log.csv with the reason, so it can be quoted in the report.

Run:  python 03_clean_transform.py      -> data/powercase_survey_clean.csv, data/cleaning_log.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE / "data" / "powercase_survey_raw.csv")
log = []


def note(step, action, rows_affected, reason):
    log.append({"step": step, "action": action, "rows_affected": int(rows_affected), "reason": reason})
    print(f"[{step}] {action}: {int(rows_affected)} rows - {reason}")


raw_shape = df.shape
print("Raw shape:", raw_shape)
print("Missing values before:\n", df.isna().sum()[df.isna().sum() > 0].to_string(), "\n")

# 1 ------------------------------------------------------------ duplicates
n0 = len(df)
df = df.drop_duplicates(subset="respondent_id", keep="first").copy()
note("1 Duplicates", "Dropped repeated respondent_id", n0 - len(df), "Double submissions would count one person twice and inflate buy rates")

# 2 ------------------------------------------------------------ types
df["responded_at"] = pd.to_datetime(df["responded_at"], format="mixed", dayfirst=True)
note("2 Types", "Parsed responded_at from two text formats to datetime", len(df), "Dates arrived as dd/mm/yyyy and yyyy-mm-dd text")
num_cols = ["age", "monthly_income_aed", "phone_age_years", "screen_hrs", "low_battery_per_week", "battery_anxiety",
            "hrs_away_from_charger", "powerbank_spend_aed", "case_spend_aed", "price_sensitivity", "brand_importance",
            "accessories_per_year", "trips_per_year", "max_price_aed", "completion_min"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
note("2 Types", f"Converted {len(num_cols)} numeric columns with to_numeric", len(df), "Columns holding outlier text or mixed types must be numeric for statistics")

# 3 ------------------------------------------------------------ label standardisation
def canon(series, mapping, default=None):
    key = series.astype("string").str.strip().str.rstrip(".").str.lower()
    out = key.map(mapping)
    return out if default is None else out.fillna(default)

before = df["emirate"].nunique()
df["emirate"] = canon(df["emirate"], {"dubai": "Dubai", "dxb": "Dubai", "abu dhabi": "Abu Dhabi", "sharjah": "Sharjah",
                                       "ajman": "Ajman", "other emirates": "Other emirates"})
note("3 Labels", "Standardised emirate spellings (dubai, DXB, DUBAI, Dubai.)", before - df["emirate"].nunique(),
     "Different spellings of one city split groups in every chart and model")
before = df["phone_brand"].nunique()
df["phone_brand"] = canon(df["phone_brand"], {"iphone": "iPhone", "apple": "iPhone", "samsung": "Samsung", "other android": "Other Android"})
note("3 Labels", "Standardised phone brand (iphone, Apple, IPHONE, samsung)", before - df["phone_brand"].nunique(),
     "Same brand written several ways")
yn = {"yes": "Yes", "y": "Yes", "1": "Yes", "no": "No", "n": "No", "0": "No"}
for c in ["carries_power_bank", "uses_case", "preorder_signup"]:
    df[c] = canon(df[c], yn)
note("3 Labels", "Mapped yes/Y/YES/1 and no/N/NO/0 to Yes/No in 3 columns", len(df), "Models need one consistent code per answer")

# 4 ------------------------------------------------------------ careless respondents
careless = df["completion_min"] < 1.5
note("4 Quality", "Removed respondents who finished in under 1.5 minutes", careless.sum(),
     "A 30-question survey cannot be read in 90 seconds; these answers are random noise")
df = df[~careless].copy()

# 5 ------------------------------------------------------------ impossible values and outliers
rules = {
    "age": (16, 80), "monthly_income_aed": (500, 100000), "screen_hrs": (0.5, 18), "low_battery_per_week": (0, 25),
    "accessories_per_year": (0, 30), "max_price_aed": (20, 1000), "hrs_away_from_charger": (0, 20),
}
for c, (lo, hi) in rules.items():
    bad = (df[c] < lo) | (df[c] > hi)
    df.loc[bad, c] = np.nan
    note("5 Outliers", f"Set impossible {c} (outside {lo} to {hi}) to missing", bad.sum(),
         "Values like age 120, income 250,000 or 5,000 AED for a phone case are entry errors, not real customers")
for c in ["monthly_income_aed", "powerbank_spend_aed", "case_spend_aed"]:
    q1, q3 = df[c].quantile([.25, .75])
    cap = q3 + 3 * (q3 - q1)
    n_cap = (df[c] > cap).sum()
    df[c] = df[c].clip(upper=cap)
    note("5 Outliers", f"Capped {c} at Q3 + 3*IQR ({cap:,.0f})", n_cap, "Keeps real high spenders but stops extreme tails dominating means and distances")

# 6 ------------------------------------------------------------ missing values
miss_before = df.isna().sum().sum()
df["wtp_missing"] = df["max_price_aed"].isna().astype(int)
df["monthly_income_aed"] = df["monthly_income_aed"].fillna(df.groupby("occupation")["monthly_income_aed"].transform("median"))
for c in ["monthly_income_aed", "age", "screen_hrs", "phone_age_years", "hrs_away_from_charger", "case_spend_aed",
          "low_battery_per_week", "accessories_per_year"]:
    df[c] = df[c].fillna(df[c].median())
for c in ["emirate", "occupation"]:
    df[c] = df[c].fillna(df[c].mode()[0])
note("6 Missing", "Filled numeric gaps with the median (income by occupation) and categories with the mode",
     miss_before - df.isna().sum().sum(), "Median is robust to skew; keeps rows instead of throwing away 20% of respondents")
note("6 Missing", "Left max_price_aed (regression target) unfilled and flagged with wtp_missing", df["wtp_missing"].sum(),
     "Inventing the answer we want to predict would make the regression look better than it is")

# 7 ------------------------------------------------------------ multi-select answers to 0/1 columns
ch_map = {"Phone died while navigating or on a ride app": "ch_navigation", "Phone died before the end of the workday": "ch_workday",
          "No outlet or charger when I needed one": "ch_no_charger", "Forgot or lost my power bank": "ch_forgot_pb",
          "Power bank and cables too bulky to carry": "ch_bulky", "Battery drains fast while travelling": "ch_travel_drain",
          "Missed an important call or OTP, phone was dead": "ch_missed_call"}
ft_map = {"Slim design": "ft_slim", "High capacity (4,000 mAh or more)": "ft_capacity", "Fast charging": "ft_fast",
          "Wireless or MagSafe pass-through": "ft_wireless", "Drop protection": "ft_drop", "LED battery indicator": "ft_led",
          "Card slot or kickstand": "ft_card_kickstand", "Eco-friendly material": "ft_eco", "2-year warranty": "ft_warranty"}
n_none = df["challenges"].isna().sum()
df["challenges"] = df["challenges"].fillna("None of these")
note("7 Transform", "Labelled blank 'problems faced' answers as 'None of these'", n_none,
     "A blank multi-select means the person ticked nothing. That is a real answer (zero problems), not missing data")
for col, mp in [("challenges", ch_map), ("desired_features", ft_map)]:
    s = df[col].fillna("")
    for text, short in mp.items():
        df[short] = s.str.contains(text, regex=False).astype(int)
df["challenge_count"] = df[list(ch_map.values())].sum(axis=1)
df["feature_count"] = df[list(ft_map.values())].sum(axis=1)
note("7 Transform", "Split 2 multi-select answers into 16 yes/no columns (ch_*, ft_*)", len(df),
     "Association rule mining and classification need one 0/1 column per option")

# 8 ------------------------------------------------------------ feature engineering
for c in ["carries_power_bank", "uses_case", "preorder_signup"]:
    df[c + "_flag"] = (df[c] == "Yes").astype(int)
df["age_group"] = pd.cut(df["age"], [0, 24, 34, 44, 120], labels=["18-24", "25-34", "35-44", "45+"]).astype(str)
df["income_band"] = pd.cut(df["monthly_income_aed"], [0, 5000, 10000, 20000, 1e9], labels=["Under 5k", "5k-10k", "10k-20k", "20k+"]).astype(str)
need = ["screen_hrs", "low_battery_per_week", "battery_anxiety", "hrs_away_from_charger"]
df["battery_need_score"] = ((df[need] - df[need].mean()) / df[need].std()).mean(axis=1).round(3)
df["log_income"] = np.log(df["monthly_income_aed"]).round(3)
df["intent_yes"] = (df["intent"] == "Yes").astype(int)
df["intent_interested"] = df["intent"].isin(["Yes", "Maybe"]).astype(int)
note("8 Transform", "Created age_group, income_band, battery_need_score, log_income, challenge_count, feature_count, binary flags",
     len(df), "Derived columns turn raw answers into business-ready segments and model inputs")

df = df.sort_values("respondent_id").reset_index(drop=True)
df.to_csv(HERE / "data" / "powercase_survey_clean.csv", index=False)
pd.DataFrame(log).to_csv(HERE / "data" / "cleaning_log.csv", index=False)
left = df.drop(columns=["max_price_aed", "wtp_missing"]).isna().sum().sum()
print(f"\nRaw {raw_shape[0]} rows x {raw_shape[1]} cols  ->  clean {df.shape[0]} rows x {df.shape[1]} cols")
print("Missing values left (excluding the unfilled target):", int(left))
