"""
Step 3 - Descriptive, diagnostic, predictive and prescriptive analytics for PowerCase.

  Descriptive  : what do respondents look like?
  Diagnostic   : why do some want it and others not? (correlations, chi-square, group rates)
  Predictive   : 4 algorithms
                   1. Classification  -> will a person buy? (Yes / Maybe / No)
                   2. Clustering      -> which customer segments exist? (high / medium / low propensity)
                   3. Association     -> which problems and features go together? (launch roadmap)
                   4. Regression      -> how much will a person pay? (AED)
  Prescriptive : price, target segments, feature roadmap, break-even, edge cases, final verdict

Run:  python 04_analysis.py
Writes: outputs/results.json, outputs/*.csv, charts/*.png, models/*.joblib
"""
import json
import warnings
from itertools import combinations
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_predict, cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
for sub in ["outputs", "charts", "models"]:
    (HERE / sub).mkdir(exist_ok=True)
SEED = 42
sns.set_theme(style="whitegrid", font_scale=0.95)
df = pd.read_csv(HERE / "data" / "powercase_survey_clean.csv")
R = {}  # everything the dashboard needs

CH = {"ch_navigation": "Phone died while navigating", "ch_workday": "Phone died before end of workday",
      "ch_no_charger": "No charger or outlet available", "ch_forgot_pb": "Forgot or lost power bank",
      "ch_bulky": "Power bank and cables too bulky", "ch_travel_drain": "Battery drains while travelling",
      "ch_missed_call": "Missed important call or OTP"}
FT = {"ft_slim": "Slim design", "ft_capacity": "High capacity (4,000 mAh+)", "ft_fast": "Fast charging",
      "ft_wireless": "Wireless / MagSafe pass-through", "ft_drop": "Drop protection", "ft_led": "LED battery indicator",
      "ft_card_kickstand": "Card slot or kickstand", "ft_eco": "Eco-friendly material", "ft_warranty": "2-year warranty"}
LABEL = {**CH, **FT}

NUM = ["age", "monthly_income_aed", "phone_age_years", "screen_hrs", "low_battery_per_week", "battery_anxiety",
       "hrs_away_from_charger", "powerbank_spend_aed", "case_spend_aed", "price_sensitivity", "brand_importance",
       "accessories_per_year", "trips_per_year", "challenge_count", "feature_count",
       "carries_power_bank_flag", "uses_case_flag"] + list(CH) + list(FT)
CAT = ["gender", "emirate", "occupation", "phone_brand"]
FEATURES = NUM + CAT
CLASSES = ["No", "Maybe", "Yes"]


def preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])


def save_fig(name):
    plt.tight_layout()
    plt.savefig(HERE / "charts" / name, dpi=140)
    plt.close()


# =============================================================================================== 1. DESCRIPTIVE
print("1. Descriptive analytics")
desc = df[["age", "monthly_income_aed", "screen_hrs", "low_battery_per_week", "battery_anxiety", "hrs_away_from_charger",
           "max_price_aed", "challenge_count", "feature_count"]].describe().T.round(2)
desc.to_csv(HERE / "outputs" / "descriptive_summary.csv")
R["n"] = len(df)
R["intent_share"] = (df["intent"].value_counts(normalize=True).reindex(CLASSES) * 100).round(1).to_dict()
R["wtp_median"] = float(df["max_price_aed"].median())
R["wtp_mean"] = round(float(df["max_price_aed"].mean()), 1)
R["pct_carry_powerbank"] = round(df["carries_power_bank_flag"].mean() * 100, 1)
R["pct_anxiety_4_5"] = round((df["battery_anxiety"] >= 4).mean() * 100, 1)
R["pct_low_batt_3plus"] = round((df["low_battery_per_week"] >= 3).mean() * 100, 1)
R["problem_prevalence"] = {CH[c]: round(df[c].mean() * 100, 1) for c in CH}
R["feature_wishes"] = {FT[c]: round(df[c].mean() * 100, 1) for c in FT}

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
sns.countplot(data=df, x="intent", order=CLASSES, palette=["#9aa5ab", "#f0a35e", "#e4501d"], ax=ax[0])
ax[0].set_title("Would you buy PowerCase?")
sns.histplot(df["max_price_aed"].dropna(), bins=30, color="#0c8473", ax=ax[1])
ax[1].axvline(169, color="k", ls="--"); ax[1].set_title("Highest price people would pay (AED)")
sns.histplot(df["screen_hrs"], bins=25, color="#546b73", ax=ax[2]); ax[2].set_title("Daily screen hours")
save_fig("01_descriptive.png")

# =============================================================================================== 2. DIAGNOSTIC
print("2. Diagnostic analytics")
corr_cols = ["age", "monthly_income_aed", "screen_hrs", "low_battery_per_week", "battery_anxiety", "hrs_away_from_charger",
             "challenge_count", "price_sensitivity", "brand_importance", "case_spend_aed", "max_price_aed", "intent_yes", "intent_interested"]
corr = df[corr_cols].corr().round(2)
corr.to_csv(HERE / "outputs" / "correlation_matrix.csv")
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, cbar_kws={"shrink": .8})
plt.title("Correlation heatmap")
save_fig("02_correlation_heatmap.png")
R["corr_with_interest"] = corr["intent_interested"].drop(["intent_interested", "intent_yes"]).sort_values(ascending=False).round(2).to_dict()

tests = {}
for col in ["phone_brand", "occupation", "age_group", "income_band", "emirate", "gender"]:
    ct = pd.crosstab(df[col], df["intent"])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    n = ct.values.sum()
    v = float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))
    tests[col] = {"chi2": round(chi2, 1), "p_value": float(p), "cramers_v": round(v, 3), "significant": bool(p < 0.05)}
groups = [g["battery_anxiety"].values for _, g in df.groupby("intent")]
h, p = stats.kruskal(*groups)
tests["battery_anxiety (Kruskal-Wallis)"] = {"H": round(float(h), 1), "p_value": float(p), "significant": bool(p < 0.05)}
rho, p = stats.spearmanr(df["monthly_income_aed"], df["max_price_aed"], nan_policy="omit")
tests["income vs max price (Spearman)"] = {"rho": round(float(rho), 2), "p_value": float(p), "significant": bool(p < 0.05)}
R["hypothesis_tests"] = tests

diag = {}
for col in ["battery_anxiety", "phone_brand", "age_group", "income_band", "occupation", "challenge_count", "price_sensitivity"]:
    g = df.groupby(col).agg(respondents=("intent", "size"), pct_yes=("intent_yes", "mean"), pct_interested=("intent_interested", "mean"),
                            median_wtp=("max_price_aed", "median")).reset_index()
    g[["pct_yes", "pct_interested"]] = (g[["pct_yes", "pct_interested"]] * 100).round(1)
    diag[col] = json.loads(g.to_json(orient="records"))
R["diagnostic_groups"] = diag
a = df.groupby("battery_anxiety")["intent_interested"].mean() * 100
plt.figure(figsize=(6, 4)); plt.bar(a.index.astype(str), a.values, color="#e4501d")
plt.xlabel("Battery anxiety (1 to 5)"); plt.ylabel("% Yes or Maybe"); plt.title("Interest rises with battery anxiety")
save_fig("03_interest_by_anxiety.png")

# =============================================================================================== 3. CLASSIFICATION
print("3. Classification")
X, y = df[FEATURES], df["intent"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
models = {
    "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, class_weight="balanced", random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
    "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
}
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
clf_rows, fitted = [], {}
for name, m in models.items():
    pipe = Pipeline([("pre", preprocessor()), ("m", m)]).fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    proba = pipe.predict_proba(Xte)
    cv = cross_validate(Pipeline([("pre", preprocessor()), ("m", m)]), X, y, cv=skf,
                        scoring={"acc": "accuracy", "f1": "f1_macro"}, n_jobs=1)
    clf_rows.append({"model": name, "test_accuracy": accuracy_score(yte, pred), "test_macro_f1": f1_score(yte, pred, average="macro"),
                     "test_roc_auc": roc_auc_score(yte, proba, multi_class="ovr", labels=pipe.classes_),
                     "cv_accuracy": cv["test_acc"].mean(), "cv_macro_f1": cv["test_f1"].mean(), "cv_f1_std": cv["test_f1"].std()})
    fitted[name] = pipe
clf = pd.DataFrame(clf_rows).round(3)
baseline = float((yte == y.mode()[0]).mean())
clf.to_csv(HERE / "outputs" / "classification_model_comparison.csv", index=False)
best_name = clf.sort_values("cv_macro_f1", ascending=False).iloc[0]["model"]
best = fitted[best_name]
pred = best.predict(Xte)
cm = confusion_matrix(yte, pred, labels=CLASSES)
rep = classification_report(yte, pred, labels=CLASSES, output_dict=True, zero_division=0)
R["classification"] = {"models": json.loads(clf.to_json(orient="records")), "best_model": best_name,
                       "baseline_accuracy": round(baseline, 3), "confusion_matrix": cm.tolist(), "labels": CLASSES,
                       "report": {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in rep.items() if isinstance(v, dict)},
                       "why_these_columns": "All questions answered before the purchase question, except max price and pre-order (not available for a new prospect)."}
print(clf.to_string(index=False), f"\nBest: {best_name}  | majority-class baseline accuracy on test: {baseline:.3f}")

pi = permutation_importance(best, Xte, yte, n_repeats=8, random_state=SEED, scoring="f1_macro", n_jobs=1)
imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values(ascending=False)
imp_top = imp.head(12)
R["classification"]["importance"] = {LABEL.get(k, k): round(float(v), 4) for k, v in imp_top.items()}
joblib.dump(best, HERE / "models" / "intent_classifier.joblib")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax[0], cbar=False)
ax[0].set_title(f"Confusion matrix ({best_name}, test set)"); ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("Actual")
clf.set_index("model")[["test_accuracy", "test_macro_f1", "cv_macro_f1"]].plot.bar(ax=ax[1], color=["#546b73", "#0c8473", "#e4501d"])
ax[1].axhline(baseline, color="k", ls="--", lw=1); ax[1].set_title("Model comparison (dashed = majority-class guess)")
ax[1].set_ylim(0, 1); ax[1].tick_params(axis="x", rotation=20)
imp_top.rename(index=lambda k: LABEL.get(k, k)).sort_values().plot.barh(ax=ax[2], color="#0c8473")
ax[2].set_title("What drives the prediction")
save_fig("04_classification.png")

# out-of-fold probabilities and edge cases
oof_proba = cross_val_predict(Pipeline([("pre", preprocessor()), ("m", models[best_name])]), X, y, cv=skf, method="predict_proba")
cls_order = list(best.classes_)
oof_pred = np.array(cls_order)[oof_proba.argmax(axis=1)]
df["p_yes"] = oof_proba[:, cls_order.index("Yes")]
df["p_maybe"] = oof_proba[:, cls_order.index("Maybe")]
df["p_no"] = oof_proba[:, cls_order.index("No")]
df["oof_correct"] = (oof_pred == y.values).astype(int)
seg_acc = {}
for col in ["phone_brand", "age_group", "occupation", "income_band"]:
    g = df.groupby(col)["oof_correct"].agg(["mean", "size"])
    seg_acc[col] = {k: {"accuracy": round(float(r["mean"]), 3), "n": int(r["size"])} for k, r in g.iterrows()}
rng = np.random.default_rng(SEED)
noise_rows = []
for flip in [0.0, 0.05, 0.10, 0.20, 0.30]:
    ytr_n = ytr.copy().values
    idx = rng.random(len(ytr_n)) < flip
    ytr_n[idx] = rng.choice(CLASSES, idx.sum())
    pn = Pipeline([("pre", preprocessor()), ("m", models[best_name])]).fit(Xtr, ytr_n)
    noise_rows.append({"extra_label_noise": flip, "test_accuracy": round(accuracy_score(yte, pn.predict(Xte)), 3),
                       "test_macro_f1": round(f1_score(yte, pn.predict(Xte), average="macro"), 3)})
R["classification"]["segment_accuracy"] = seg_acc
R["classification"]["noise_robustness"] = noise_rows

# =============================================================================================== 4. REGRESSION
print("4. Regression")
dr = df[df["wtp_missing"] == 0].copy()
Xr, yr = dr[FEATURES], dr["max_price_aed"]
Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(Xr, yr, test_size=0.2, random_state=SEED)
reg_models = {
    "Ridge Regression": Ridge(alpha=10),
    "Random Forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=SEED),
    "Gradient Boosting": GradientBoostingRegressor(random_state=SEED),
}
kf = KFold(5, shuffle=True, random_state=SEED)
reg_rows, reg_fitted = [], {}
for name, m in reg_models.items():
    pipe = Pipeline([("pre", preprocessor()), ("m", m)]).fit(Xtr_r, ytr_r)
    p = pipe.predict(Xte_r)
    cv = cross_validate(Pipeline([("pre", preprocessor()), ("m", m)]), Xr, yr, cv=kf, scoring="r2")
    reg_rows.append({"model": name, "test_r2": r2_score(yte_r, p), "test_mae_aed": mean_absolute_error(yte_r, p),
                     "test_rmse_aed": float(np.sqrt(mean_squared_error(yte_r, p))), "cv_r2": cv["test_score"].mean()})
    reg_fitted[name] = pipe
reg = pd.DataFrame(reg_rows).round(3)
reg.to_csv(HERE / "outputs" / "regression_model_comparison.csv", index=False)
reg_best_name = reg.sort_values("cv_r2", ascending=False).iloc[0]["model"]
reg_best = reg_fitted[reg_best_name]
pte = reg_best.predict(Xte_r)
pd.DataFrame({"actual": yte_r.values, "predicted": pte.round(1)}).to_csv(HERE / "outputs" / "regression_test_predictions.csv", index=False)
pir = permutation_importance(reg_best, Xte_r, yte_r, n_repeats=8, random_state=SEED, scoring="r2", n_jobs=1)
imp_r = pd.Series(pir.importances_mean, index=FEATURES).sort_values(ascending=False).head(10)
oof_wtp = cross_val_predict(Pipeline([("pre", preprocessor()), ("m", reg_models[reg_best_name])]), Xr, yr, cv=kf)
dr["wtp_pred_oof"] = oof_wtp
mae_band = dr.assign(abs_err=(dr["wtp_pred_oof"] - dr["max_price_aed"]).abs()).groupby("income_band")["abs_err"].mean().round(1).to_dict()
R["regression"] = {"models": json.loads(reg.to_json(orient="records")), "best_model": reg_best_name,
                   "baseline_mae_aed": round(float((yte_r - ytr_r.mean()).abs().mean()), 1),
                   "importance": {LABEL.get(k, k): round(float(v), 4) for k, v in imp_r.items()},
                   "mae_by_income_band": mae_band, "n_train": int(len(Xtr_r)), "n_test": int(len(Xte_r))}
print(reg.to_string(index=False))
joblib.dump(reg_best, HERE / "models" / "wtp_regressor.joblib")
df["wtp_final"] = df["max_price_aed"].fillna(pd.Series(reg_best.predict(df[FEATURES]), index=df.index))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
ax[0].scatter(yte_r, pte, alpha=.5, color="#0c8473", s=18); lim = [30, 500]; ax[0].plot(lim, lim, "k--", lw=1)
ax[0].set_xlabel("Actual max price (AED)"); ax[0].set_ylabel("Predicted (AED)")
ax[0].set_title(f"{reg_best_name}: R2 = {reg.sort_values('cv_r2').iloc[-1]['test_r2']:.2f}")
imp_r.rename(index=lambda k: LABEL.get(k, k)).sort_values().plot.barh(ax=ax[1], color="#e4501d")
ax[1].set_title("What drives willingness to pay")
save_fig("05_regression.png")

# =============================================================================================== 5. CLUSTERING
print("5. Clustering")
CL = ["screen_hrs", "low_battery_per_week", "battery_anxiety", "hrs_away_from_charger", "log_income", "price_sensitivity",
      "case_spend_aed", "accessories_per_year", "challenge_count", "carries_power_bank_flag"]
Z = StandardScaler().fit(df[CL])
Zx = Z.transform(df[CL])
ks = list(range(2, 8))
sil, inertia = [], []
for k in ks:
    km = KMeans(k, n_init=10, random_state=SEED).fit(Zx)
    sil.append(silhouette_score(Zx, km.labels_)); inertia.append(km.inertia_)
k_best = ks[3 - 2 + int(np.argmax(sil[1:5]))]  # choose among k = 3..6 so that we get high / medium / low groups
km = KMeans(k_best, n_init=20, random_state=SEED).fit(Zx)
df["cluster"] = km.labels_
cent = pd.DataFrame(km.cluster_centers_, columns=CL)
prof = df.groupby("cluster").agg(size=("intent", "size"), pct_yes=("intent_yes", "mean"), pct_interested=("intent_interested", "mean"),
                                 median_wtp=("wtp_final", "median"), mean_age=("age", "mean"), median_income=("monthly_income_aed", "median"),
                                 screen_hrs=("screen_hrs", "mean"), low_battery_per_week=("low_battery_per_week", "mean"),
                                 battery_anxiety=("battery_anxiety", "mean"), hrs_away=("hrs_away_from_charger", "mean"),
                                 price_sensitivity=("price_sensitivity", "mean"), pct_iphone=("phone_brand", lambda s: (s == "iPhone").mean()),
                                 pct_powerbank=("carries_power_bank_flag", "mean"), problems=("challenge_count", "mean"))
prof["share"] = prof["size"] / len(df)
order = prof.sort_values("pct_interested", ascending=False).index.tolist()
tiers = {}
for rank, c in enumerate(order):
    tiers[c] = "High" if rank == 0 else ("Low" if rank == len(order) - 1 else ("Medium" if len(order) == 3 else ("Medium-high" if rank == 1 else "Medium-low")))
phrases = {"screen_hrs": ("heavy screen users", "light screen users"), "low_battery_per_week": ("frequent low battery", "rarely low battery"),
           "battery_anxiety": ("high battery anxiety", "relaxed about battery"), "hrs_away_from_charger": ("long hours away from a charger", "always near a charger"),
           "log_income": ("higher income", "lower income"), "price_sensitivity": ("price sensitive", "price insensitive"),
           "case_spend_aed": ("spend more on cases", "spend little on cases"), "accessories_per_year": ("frequent accessory buyers", "rare accessory buyers"),
           "challenge_count": ("many battery problems", "few battery problems"), "carries_power_bank_flag": ("already carry a power bank", "no power bank")}
names = {}
for c in range(k_best):
    row = cent.loc[c]
    top = row.abs().sort_values(ascending=False).index[0]
    names[c] = f"{tiers[c]} propensity: {phrases[top][0] if row[top] > 0 else phrases[top][1]}"
df["cluster_name"] = df["cluster"].map(names)
prof_out = prof.copy(); prof_out["tier"] = [tiers[c] for c in prof.index]; prof_out["name"] = [names[c] for c in prof.index]
prof_out = prof_out.round(3).reset_index()
prof_out.to_csv(HERE / "outputs" / "cluster_profiles.csv", index=False)
R["clustering"] = {"k_chosen": int(k_best), "k_range": ks, "silhouette": [round(float(s), 3) for s in sil], "inertia": [round(float(i), 1) for i in inertia],
                   "profiles": json.loads(prof_out.to_json(orient="records")),
                   "centroids_z": json.loads(cent.round(2).assign(cluster=range(k_best)).to_json(orient="records")), "features": CL,
                   "reason": "k was chosen as the best silhouette among 3 to 6 clusters, because the business needs at least a high, medium and low group."}
joblib.dump({"scaler": Z, "kmeans": km, "features": CL, "names": names}, HERE / "models" / "kmeans_segments.joblib")
pc = PCA(2, random_state=SEED).fit_transform(Zx)
df["pc1"], df["pc2"] = pc[:, 0], pc[:, 1]
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
for c in range(k_best):
    m = df["cluster"] == c
    ax[0].scatter(df.loc[m, "pc1"], df.loc[m, "pc2"], s=12, alpha=.6, label=names[c])
ax[0].set_title(f"Customer segments (k = {k_best}, PCA view)"); ax[0].legend(fontsize=7)
ax[1].plot(ks, sil, "o-", color="#0c8473"); ax[1].axvline(k_best, color="k", ls="--"); ax[1].set_title("Silhouette score by k"); ax[1].set_xlabel("k")
hm = cent.copy(); hm.index = [names[c] for c in range(k_best)]
sns.heatmap(hm.round(1), annot=True, cmap="RdBu_r", center=0, ax=ax[2], cbar=False, annot_kws={"size": 7}); ax[2].set_title("Segment profiles (z-scores)")
ax[2].tick_params(axis="y", labelsize=7); ax[2].tick_params(axis="x", labelsize=7, rotation=45)
save_fig("06_clustering.png")

# =============================================================================================== 6. ASSOCIATION RULES
print("6. Association rule mining")
B = df[list(CH) + list(FT)].astype(bool)


def apriori(B, min_support, max_len=3):
    n = len(B); sup = {}
    for c in B.columns:
        s = B[c].mean()
        if s >= min_support: sup[(c,)] = s
    for k in range(2, max_len + 1):
        base = [i for (i,) in [x for x in sup if len(x) == 1]]
        for combo in combinations(base, k):
            if k == 3 and any(tuple(sorted(sub)) not in sup and tuple(sub) not in sup for sub in combinations(combo, 2)):
                continue
            s = B[list(combo)].all(axis=1).mean()
            if s >= min_support: sup[tuple(combo)] = s
    return sup


def rules_from(sup, min_conf, min_lift, ante_ok, cons_ok):
    out = []
    for iset, s in sup.items():
        if len(iset) < 2: continue
        for cons in iset:
            ante = tuple(i for i in iset if i != cons)
            key = ante if ante in sup else tuple(sorted(ante))
            if key not in sup or (cons,) not in sup: continue
            if not (all(ante_ok(a) for a in ante) and cons_ok(cons)): continue
            conf = s / sup[key]; lift = conf / sup[(cons,)]
            if conf >= min_conf and lift >= min_lift:
                out.append({"if_customer": " + ".join(LABEL[a] for a in ante), "then_also": LABEL[cons],
                            "support": round(s, 3), "confidence": round(conf, 3), "lift": round(lift, 2)})
    return pd.DataFrame(out).sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True) if out else pd.DataFrame(
        columns=["if_customer", "then_also", "support", "confidence", "lift"])


sup = apriori(B, 0.04, 3)
r_ch_ft = rules_from(sup, 0.45, 1.10, lambda a: a.startswith("ch_"), lambda c: c.startswith("ft_"))
r_ft_ft = rules_from(sup, 0.50, 1.15, lambda a: a.startswith("ft_"), lambda c: c.startswith("ft_"))
r_ch_ch = rules_from(sup, 0.35, 1.15, lambda a: a.startswith("ch_"), lambda c: c.startswith("ch_"))
r_ch_ft.to_csv(HERE / "outputs" / "rules_problem_to_feature.csv", index=False)
r_ft_ft.to_csv(HERE / "outputs" / "rules_feature_to_feature.csv", index=False)
r_ch_ch.to_csv(HERE / "outputs" / "rules_problem_to_problem.csv", index=False)

buyers = df[df["intent_interested"] == 1]
fsup = (buyers[list(FT)].mean() * 100).sort_values(ascending=False)
p1, p2, p3 = list(fsup.index[:3]), list(fsup.index[3:6]), list(fsup.index[6:])
both = buyers[buyers[p1].all(axis=1)]
def attach(feats):
    return [{"feature": FT[f], "wanted_by_interested_pct": round(float(fsup[f]), 1),
             "also_wanted_given_phase1_pct": round(float(both[f].mean() * 100), 1) if len(both) else None} for f in feats]
R["association"] = {"n_itemsets": len(sup), "min_support": 0.04,
                    "problem_to_feature": json.loads(r_ch_ft.head(12).to_json(orient="records")),
                    "feature_to_feature": json.loads(r_ft_ft.head(12).to_json(orient="records")),
                    "problem_to_problem": json.loads(r_ch_ch.head(8).to_json(orient="records")),
                    "roadmap": {"Phase 1 (launch)": attach(p1), "Phase 2 (+3 to 6 months)": attach(p2), "Phase 3 (later)": attach(p3)},
                    "phase1_group_size": int(len(both)),
                    "columns_used": "Only the 7 problem columns (ch_*) and 9 feature columns (ft_*), not the other survey questions."}
print("Phase 1:", [FT[f] for f in p1], "| Phase 2:", [FT[f] for f in p2])
top_r = pd.concat([r_ft_ft.head(6).assign(kind="feature -> feature"), r_ch_ft.head(6).assign(kind="problem -> feature")]) if len(r_ft_ft) or len(r_ch_ft) else pd.DataFrame()
if len(top_r):
    plt.figure(figsize=(9, 5))
    lab = (top_r["if_customer"].str.slice(0, 32) + "  ->  " + top_r["then_also"].str.slice(0, 26))
    plt.barh(lab[::-1], top_r["lift"][::-1], color=["#e4501d" if k == "problem -> feature" else "#0c8473" for k in top_r["kind"][::-1]])
    plt.axvline(1, color="k", lw=1); plt.xlabel("Lift (above 1 means they go together)"); plt.title("Strongest association rules")
    plt.yticks(fontsize=7)
    save_fig("07_association_rules.png")

# =============================================================================================== 7. PRESCRIPTIVE
print("7. Prescriptive analytics")
A = dict(smartphone_users=7_460_000, target_age_share=0.627, model_coverage=0.35, vat=0.05, landed_cost=51.0, platform_fee=0.18,
         fulfillment=10.0, returns_pct=0.08, w_yes=0.70, w_maybe=0.20, one_off=105_000, fixed_year=24_000, growth=[1.0, 1.5, 1.95],
         exposure=0.03, ad_cost_per_exposure=2.0, targeting_premium=1.2, survey_to_market=0.5, targeting_accuracy=0.5)
R["assumptions"] = A
df["intent_value"] = df["intent"].map({"Yes": A["w_yes"], "Maybe": A["w_maybe"], "No": 0.0})


def conv(price, frame=df, w=None):
    v = frame["intent_value"] if w is None else frame["intent"].map({"Yes": w[0], "Maybe": w[1], "No": 0.0})
    # survey_to_market: people who answer a battery survey are keener than the general public, so halve the buy rate
    return float((v * (frame["wtp_final"] >= price)).mean()) * A["survey_to_market"]


def economics(price, conversion, landed=None, ad_cost=None, exposure=None, premium=1.0, growth=None):
    landed = A["landed_cost"] if landed is None else landed
    ad_cost = A["ad_cost_per_exposure"] if ad_cost is None else ad_cost
    exposure = A["exposure"] if exposure is None else exposure
    growth = A["growth"] if growth is None else growth
    audience = A["smartphone_users"] * A["target_age_share"] * A["model_coverage"]
    exposures = audience * exposure
    units = exposures * conversion
    net = price / (1 + A["vat"])
    pre_marketing = net - landed - A["platform_fee"] * net - A["fulfillment"] - A["returns_pct"] * net
    cac = (ad_cost * premium / conversion) if conversion > 0 else float("inf")
    cm = pre_marketing - cac
    months, cum, cashflow = [], -A["one_off"], []
    breakeven_month = None
    for yr, g in enumerate(growth):
        for m in range(12):
            u = units * g / 12
            cum += u * cm - A["fixed_year"] / 12
            cashflow.append(cum)
            if breakeven_month is None and cum >= 0: breakeven_month = yr * 12 + m + 1
    ycum = [cashflow[11], cashflow[23], cashflow[35]]
    return {"price": price, "conversion": conversion, "units_y1": units, "net_price": net, "pre_marketing_margin": pre_marketing,
            "cac": cac, "cm_unit": cm, "cm_pct": cm / net, "ad_spend_y1": exposures * ad_cost * premium,
            "breakeven_units_y1": (A["one_off"] + A["fixed_year"]) / cm if cm > 0 else None, "breakeven_month": breakeven_month,
            "cum_y1": ycum[0], "cum_y2": ycum[1], "cum_y3": ycum[2], "cashflow": cashflow}


# price curve
grid = list(range(99, 300, 10))
curve = []
for p in grid:
    e = economics(p, conv(p))
    curve.append({"price": p, "conversion_pct": round(e["conversion"] * 100, 2), "units_y1": round(e["units_y1"]), "cm_unit": round(e["cm_unit"], 1),
                  "cum_y3": round(e["cum_y3"]), "breakeven_month": e["breakeven_month"]})
curve_df = pd.DataFrame(curve); curve_df.to_csv(HERE / "outputs" / "price_curve.csv", index=False)
best_row = curve_df.sort_values("cum_y3", ascending=False).iloc[0]
price_best = int(best_row["price"])
price_rec = min(max(price_best, 149), 249)
R["price_curve"] = json.loads(curve_df.to_json(orient="records"))
R["price_best_by_profit"] = price_best
R["price_recommended"] = price_rec
print(f"Best price by 3-year profit: AED {price_best}; recommended: AED {price_rec}")

# strategies
tiers_by_conv = sorted(range(k_best), key=lambda c: -conv(price_rec, df[df["cluster"] == c]))
strategies = {}
mass = economics(price_rec, conv(price_rec))
strategies["Mass market (everyone)"] = mass
for n in [1, 2]:
    sel = tiers_by_conv[:n]
    sub = df[df["cluster"].isin(sel)]
    # ads cannot target "battery anxiety" exactly, so only half of the exposures land on the chosen segment
    eff = A["targeting_accuracy"] * conv(price_rec, sub) + (1 - A["targeting_accuracy"]) * conv(price_rec)
    e = economics(price_rec, eff, premium=A["targeting_premium"])
    e["segment_conversion"] = conv(price_rec, sub)
    e["share_of_market"] = len(sub) / len(df); e["segments"] = [names[c] for c in sel]
    strategies[f"Target top {n} segment{'s' if n > 1 else ''}"] = e
mass["share_of_market"] = 1.0; mass["segments"] = ["All"]
strategies_out = {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "cashflow"} for k, v in strategies.items()}
R["strategies"] = strategies_out
R["cashflows"] = {k: [round(x) for x in v["cashflow"]] for k, v in strategies.items()}
best_strategy = max(strategies, key=lambda k: strategies[k]["cum_y3"])
rec = strategies[best_strategy]

# 169 comparison (website price) for reference
R["reference_169"] = {k: round(v, 3) if isinstance(v, float) else v for k, v in economics(169, conv(169)).items() if k != "cashflow"}

# edge cases applied to the recommended strategy
sub_rec = df[df["cluster"].isin(tiers_by_conv[:int(best_strategy.split()[2])])] if best_strategy.startswith("Target") else df
prem = A["targeting_premium"] if best_strategy.startswith("Target") else 1.0
def eff_conv(p):
    if not best_strategy.startswith("Target"):
        return conv(p)
    return A["targeting_accuracy"] * conv(p, sub_rec) + (1 - A["targeting_accuracy"]) * conv(p)
base_conv = eff_conv(price_rec)
edge = []
def add_edge(name, **kw):
    e = economics(kw.pop("price", price_rec), kw.pop("conversion", base_conv), premium=prem, **kw)
    edge.append({"scenario": name, "units_y1": round(e["units_y1"]), "cm_unit": round(e["cm_unit"], 1), "cum_y3": round(e["cum_y3"]),
                 "breakeven_month": e["breakeven_month"], "status": "Pays back" if e["cum_y3"] > 0 else "Loses money"})
add_edge("Base (recommended strategy)")
add_edge("Competitor magnetic packs cut conversion by 25%", conversion=base_conv * 0.75)
add_edge("Survey bias is worse than assumed (another 50% cut)", conversion=base_conv * 0.5)
add_edge("Supplier cost rises 20%", landed=A["landed_cost"] * 1.2)
add_edge("Advertising costs double", ad_cost=A["ad_cost_per_exposure"] * 2)
add_edge("Reach is half as large (1.5% exposure)", exposure=A["exposure"] / 2)
add_edge("Price cut to AED 149", price=149, conversion=eff_conv(149))
add_edge("Growth stalls after Year 1", growth=[1.0, 1.0, 1.0])
R["edge_cases"] = edge

# sensitivity grid: price x advertising cost per exposure
sens = []
for p in [129, 149, 169, 189, 209, 229, 249]:
    row = {"price": p}
    for ad in [1.0, 1.5, 2.0, 3.0, 4.0]:
        row[f"ad_{ad}"] = round(economics(p, conv(p), ad_cost=ad)["cum_y3"])
    sens.append(row)
pd.DataFrame(sens).to_csv(HERE / "outputs" / "sensitivity_price_vs_ad_cost.csv", index=False)
R["sensitivity"] = sens

# sales and marketing channels for the target segments
tgt = df[df["cluster"].isin(tiers_by_conv[:2])]
R["channels_target"] = {"purchase": (tgt["purchase_channel"].value_counts(normalize=True) * 100).round(1).to_dict(),
                        "discovery": (tgt["discovery_channel"].value_counts(normalize=True) * 100).round(1).to_dict()}
R["format_by_brand"] = {b: (g["preferred_format"].value_counts(normalize=True) * 100).round(1).to_dict() for b, g in df.groupby("phone_brand")}
R["concerns"] = (df["top_concern"].value_counts(normalize=True) * 100).round(1).to_dict()

all_pass = all(e["status"] == "Pays back" for e in edge)
fast = rec["cum_y3"] > 0 and (rec["breakeven_month"] or 99) <= 24
verdict = ("SUSTAINABLE" if fast and all_pass else "CONDITIONALLY SUSTAINABLE" if fast else
           "MARGINAL" if rec["cum_y3"] > 0 else "NOT SUSTAINABLE")
R["verdict"] = {"label": verdict, "strategy": best_strategy, "price": price_rec, "breakeven_month": rec["breakeven_month"],
                "cum_y3": round(rec["cum_y3"]), "cm_unit": round(rec["cm_unit"], 1), "units_y1": round(rec["units_y1"]),
                "total_launch_cash": A["one_off"] + 1000 * A["landed_cost"],
                "target_segments": rec["segments"], "survives_edge_cases": int(sum(1 for e in edge if e["status"] == "Pays back")), "edge_cases_total": len(edge)}
print("Verdict:", R["verdict"])

fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
ax[0].plot(curve_df["price"], curve_df["conversion_pct"], "o-", color="#0c8473"); ax[0].set_title("Buy rate by price"); ax[0].set_xlabel("Price (AED)"); ax[0].set_ylabel("% of respondents")
ax[1].plot(curve_df["price"], curve_df["cum_y3"] / 1000, "o-", color="#e4501d"); ax[1].axhline(0, color="k", lw=1)
ax[1].axvline(price_rec, color="k", ls="--"); ax[1].set_title("3-year cumulative profit by price (AED thousand)"); ax[1].set_xlabel("Price (AED)")
for k, v in strategies.items():
    ax[2].plot(range(1, 37), np.array(v["cashflow"]) / 1000, label=k)
ax[2].axhline(0, color="k", lw=1); ax[2].set_xlabel("Month"); ax[2].set_title("Cumulative cash position (AED thousand)"); ax[2].legend(fontsize=8)
save_fig("08_price_and_breakeven.png")

# =============================================================================================== SAVE
scores = df[["respondent_id", "intent", "p_yes", "p_maybe", "p_no", "cluster", "cluster_name", "wtp_final"]].round(3)
scores.to_csv(HERE / "outputs" / "respondent_scores.csv", index=False)
joblib.dump(FEATURES, HERE / "models" / "feature_list.joblib")


def default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.ndarray,)): return o.tolist()
    raise TypeError(type(o))


json.dump(R, open(HERE / "outputs" / "results.json", "w"), indent=1, default=default)
print("\nSaved outputs/results.json, charts/*.png, models/*.joblib")
