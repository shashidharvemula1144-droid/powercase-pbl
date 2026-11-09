"""
Step 4 - Streamlit dashboard for the PowerCase validation.
Run (after steps 02, 03, 04):   streamlit run 05_dashboard.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
st.set_page_config(page_title="PowerCase Validation Dashboard", layout="wide")

TEAL, ORANGE, SLATE = "#0c8473", "#e4501d", "#546b73"
CLASSES = ["No", "Maybe", "Yes"]


def plot(fig, **kw):
    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), height=kw.pop("height", 380))
    try:
        st.plotly_chart(fig, width="stretch", **kw)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, **kw)


def table(d, **kw):
    try:
        st.dataframe(d, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(d, use_container_width=True, hide_index=True, **kw)


@st.cache_data
def load():
    clean = pd.read_csv(HERE / "data" / "powercase_survey_clean.csv")
    raw = pd.read_csv(HERE / "data" / "powercase_survey_raw.csv")
    log = pd.read_csv(HERE / "data" / "cleaning_log.csv")
    res = json.load(open(HERE / "outputs" / "results.json"))
    scores = pd.read_csv(HERE / "outputs" / "respondent_scores.csv")
    rules = {k: pd.read_csv(HERE / "outputs" / f"rules_{k}.csv") for k in ["problem_to_feature", "feature_to_feature", "problem_to_problem"]}
    reg_pred = pd.read_csv(HERE / "outputs" / "regression_test_predictions.csv")
    return clean, raw, log, res, scores, rules, reg_pred


@st.cache_resource
def load_models():
    return (joblib.load(HERE / "models" / "intent_classifier.joblib"), joblib.load(HERE / "models" / "wtp_regressor.joblib"),
            joblib.load(HERE / "models" / "kmeans_segments.joblib"), joblib.load(HERE / "models" / "feature_list.joblib"))


try:
    clean, raw, log, R, scores, rules, reg_pred = load()
    clf, reg, seg, FEATURES = load_models()
except FileNotFoundError:
    st.error("Run steps 02, 03 and 04 first (see README.md). The dashboard reads the files they create.")
    st.stop()

df = clean.merge(scores[["respondent_id", "p_yes", "p_maybe", "p_no", "cluster", "cluster_name", "wtp_final"]], on="respondent_id")
A = R["assumptions"]
CH = {"ch_navigation": "Phone died while navigating", "ch_workday": "Phone died before end of workday", "ch_no_charger": "No charger or outlet available",
      "ch_forgot_pb": "Forgot or lost power bank", "ch_bulky": "Power bank and cables too bulky", "ch_travel_drain": "Battery drains while travelling",
      "ch_missed_call": "Missed important call or OTP"}
FT = {"ft_slim": "Slim design", "ft_capacity": "High capacity (4,000 mAh+)", "ft_fast": "Fast charging", "ft_wireless": "Wireless / MagSafe pass-through",
      "ft_drop": "Drop protection", "ft_led": "LED battery indicator", "ft_card_kickstand": "Card slot or kickstand", "ft_eco": "Eco-friendly material",
      "ft_warranty": "2-year warranty"}

st.title("PowerCase: will a phone case with a built-in battery make money in the UAE?")
st.caption("Synthetic survey of {:,} respondents. Replace data/powercase_survey_raw.csv with real responses and rerun steps 03 and 04.".format(R["n"]))

# sidebar filters (descriptive and diagnostic tabs)
st.sidebar.header("Filters (Descriptive and Diagnostic tabs)")
brands = st.sidebar.multiselect("Phone brand", sorted(df["phone_brand"].unique()), sorted(df["phone_brand"].unique()))
emirates = st.sidebar.multiselect("Emirate", sorted(df["emirate"].unique()), sorted(df["emirate"].unique()))
ages = st.sidebar.multiselect("Age group", sorted(df["age_group"].unique()), sorted(df["age_group"].unique()))
f = df[df["phone_brand"].isin(brands) & df["emirate"].isin(emirates) & df["age_group"].isin(ages)]
st.sidebar.write(f"{len(f):,} of {len(df):,} respondents selected")

tabs = st.tabs(["Overview", "Data and cleaning", "Descriptive", "Diagnostic", "1 Classification", "2 Clustering",
                "3 Association rules", "4 Regression", "Strategy and break-even"])

# ------------------------------------------------------------------------------------------------ Overview
with tabs[0]:
    v = R["verdict"]
    msg = (f"**Verdict: {v['label']}.** Sell at AED {v['price']} and market to the segment '{v['target_segments'][0]}'. "
           f"Base case earns AED {v['cm_unit']} per case, breaks even in month {v['breakeven_month']} and reaches AED {v['cum_y3']:,} cumulative profit after 3 years. "
           f"It still pays back in {v['survives_edge_cases']} of {v['edge_cases_total']} stress tests.")
    (st.success if v["label"] == "SUSTAINABLE" else st.warning)(msg)
    c = st.columns(5)
    c[0].metric("Would buy (Yes)", f"{R['intent_share']['Yes']}%")
    c[1].metric("Yes or Maybe", f"{R['intent_share']['Yes'] + R['intent_share']['Maybe']:.1f}%")
    c[2].metric("Median price they would pay", f"AED {R['wtp_median']:.0f}")
    c[3].metric("Launch cash needed", f"AED {v['total_launch_cash'] / 1000:.0f}k")
    c[4].metric("Classifier accuracy", f"{[m for m in R['classification']['models'] if m['model'] == R['classification']['best_model']][0]['test_accuracy']:.0%}")
    st.subheader("What the four algorithms found")
    cl = R["clustering"]["profiles"]
    hi = max(cl, key=lambda p: p["pct_interested"])
    st.markdown(f"""
- **Classification** ({R['classification']['best_model']}) predicts Yes, Maybe or No with {[m for m in R['classification']['models'] if m['model'] == R['classification']['best_model']][0]['test_accuracy']:.0%} accuracy. The majority-class guess scores {R['classification']['baseline_accuracy']:.0%}, so look at the confusion matrix and macro F1 too.
- **Clustering** finds {R['clustering']['k_chosen']} segments. The best one ('{hi['name']}') is {hi['share']:.0%} of respondents and {hi['pct_interested']:.0%} of it says Yes or Maybe.
- **Association rules** set the launch order: {", ".join(x['feature'] for x in R['association']['roadmap']['Phase 1 (launch)'])} first.
- **Regression** predicts willingness to pay with an average error of AED {[m for m in R['regression']['models'] if m['model'] == R['regression']['best_model']][0]['test_mae_aed']:.0f}.
""")

# ------------------------------------------------------------------------------------------------ Data and cleaning
with tabs[1]:
    st.subheader("From raw to clean data")
    c = st.columns(4)
    c[0].metric("Raw rows", f"{len(raw):,}"); c[1].metric("Clean rows", f"{len(clean):,}")
    c[2].metric("Raw missing cells", f"{int(raw.isna().sum().sum()):,}"); c[3].metric("Clean columns", len(clean.columns))
    st.markdown("Every cleaning action with its reason:")
    table(log)
    a, b = st.columns(2)
    a.markdown("**Raw data (first 8 rows)**"); a.dataframe(raw.head(8))
    b.markdown("**Clean data (first 8 rows)**"); b.dataframe(clean.head(8))

# ------------------------------------------------------------------------------------------------ Descriptive
with tabs[2]:
    st.subheader("Descriptive analytics: who answered and what they said")
    a, b = st.columns(2)
    d = f["intent"].value_counts(normalize=True).reindex(CLASSES).mul(100).reset_index(); d.columns = ["Answer", "%"]
    a_fig = px.bar(d, x="Answer", y="%", color="Answer", color_discrete_map={"No": SLATE, "Maybe": "#f0a35e", "Yes": ORANGE}, title="Would you buy PowerCase?")
    with a: plot(a_fig)
    b_fig = px.histogram(f.dropna(subset=["max_price_aed"]), x="max_price_aed", nbins=30, title="Highest price people would pay (AED)", color_discrete_sequence=[TEAL])
    b_fig.add_vline(x=189, line_dash="dash")
    with b: plot(b_fig)
    a, b = st.columns(2)
    p = pd.DataFrame({"Problem": list(CH.values()), "%": [f[k].mean() * 100 for k in CH]}).sort_values("%")
    with a: plot(px.bar(p, x="%", y="Problem", orientation="h", title="Battery problems faced in the last 3 months", color_discrete_sequence=[ORANGE]))
    q = pd.DataFrame({"Feature": list(FT.values()), "%": [f[k].mean() * 100 for k in FT]}).sort_values("%")
    with b: plot(px.bar(q, x="%", y="Feature", orientation="h", title="Features respondents want", color_discrete_sequence=[TEAL]))
    a, b, c = st.columns(3)
    with a: plot(px.histogram(f, x="screen_hrs", nbins=25, title="Daily screen hours", color_discrete_sequence=[SLATE]), height=300)
    with b: plot(px.histogram(f, x="battery_anxiety", title="Battery anxiety (1 to 5)", color_discrete_sequence=[ORANGE]), height=300)
    with c: plot(px.pie(f, names="phone_brand", title="Phone brand", hole=.5), height=300)
    st.markdown("**Summary statistics**")
    st.dataframe(f[["age", "monthly_income_aed", "screen_hrs", "low_battery_per_week", "battery_anxiety", "hrs_away_from_charger", "max_price_aed"]].describe().T.round(1))

# ------------------------------------------------------------------------------------------------ Diagnostic
with tabs[3]:
    st.subheader("Diagnostic analytics: why do some people want it?")
    cols = ["age", "monthly_income_aed", "screen_hrs", "low_battery_per_week", "battery_anxiety", "hrs_away_from_charger", "challenge_count",
            "price_sensitivity", "brand_importance", "case_spend_aed", "max_price_aed", "intent_interested"]
    cm = f[cols].corr().round(2)
    plot(px.imshow(cm, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation heatmap"), height=560)
    st.caption("Look at the last column: battery anxiety, low-battery frequency and number of problems move with interest. Price sensitivity moves against it.")
    by = st.selectbox("Compare interest by", ["battery_anxiety", "phone_brand", "age_group", "income_band", "occupation", "challenge_count", "price_sensitivity", "emirate"])
    g = f.groupby(by).agg(respondents=("intent", "size"), pct_yes=("intent_yes", "mean"), pct_interested=("intent_interested", "mean")).reset_index()
    g[["pct_yes", "pct_interested"]] *= 100
    plot(px.bar(g, x=by, y=["pct_yes", "pct_interested"], barmode="group", title=f"% saying Yes and % saying Yes or Maybe, by {by}",
                color_discrete_sequence=[ORANGE, TEAL]))
    st.markdown("**Hypothesis tests** (a p-value below 0.05 means the difference is unlikely to be chance)")
    ht = pd.DataFrame(R["hypothesis_tests"]).T.reset_index().rename(columns={"index": "test"})
    st.dataframe(ht)

# ------------------------------------------------------------------------------------------------ Classification
with tabs[4]:
    C = R["classification"]
    st.subheader("Algorithm 1: Classification. Will this person buy? (Yes / Maybe / No)")
    st.markdown("**Why:** the final survey question is the label. A model that predicts it lets us score new prospects and decide whom to advertise to. "
                f"**Columns used:** {C['why_these_columns']}")
    mdl = pd.DataFrame(C["models"])
    table(mdl)
    plot(px.bar(mdl.melt(id_vars="model", value_vars=["test_accuracy", "test_macro_f1", "cv_macro_f1"]), x="model", y="value", color="variable", barmode="group",
                title=f"Model comparison. Majority-class guess = {C['baseline_accuracy']:.2f} accuracy", color_discrete_sequence=[SLATE, TEAL, ORANGE]))
    a, b = st.columns(2)
    cmx = np.array(C["confusion_matrix"])
    with a: plot(px.imshow(cmx, x=CLASSES, y=CLASSES, text_auto=True, color_continuous_scale="Oranges", labels=dict(x="Predicted", y="Actual"),
                           title=f"Confusion matrix: {C['best_model']} (test set)"))
    with b: plot(px.imshow((cmx / cmx.sum(axis=1, keepdims=True)).round(2), x=CLASSES, y=CLASSES, text_auto=True, color_continuous_scale="Teal",
                           labels=dict(x="Predicted", y="Actual"), title="Same matrix as share of each actual answer (recall)"))
    st.markdown("**Precision, recall and F1 per answer**")
    st.dataframe(pd.DataFrame(C["report"]).T.round(2))
    a, b = st.columns(2)
    imp = pd.Series(C["importance"]).sort_values()
    with a: plot(px.bar(imp, orientation="h", title="What drives the prediction", color_discrete_sequence=[TEAL], labels={"value": "Importance", "index": ""}))
    with b:
        st.markdown("**Edge cases: where does the model struggle?**")
        seg_rows = [{"group": f"{k}: {kk}", "accuracy": vv["accuracy"], "n": vv["n"]} for k, v in C["segment_accuracy"].items() for kk, vv in v.items()]
        st.dataframe(pd.DataFrame(seg_rows))
        st.markdown("**If 5% to 30% more answers in the training data are wrong:**")
        st.dataframe(pd.DataFrame(C["noise_robustness"]))

    st.markdown("---")
    st.subheader("Score a new prospect")
    c = st.columns(4)
    age = c[0].slider("Age", 18, 65, 30); income = c[1].slider("Monthly income (AED)", 1000, 40000, 9000, 500)
    brand = c[2].selectbox("Phone", ["iPhone", "Samsung", "Other Android"]); occ = c[3].selectbox("Occupation", ["Employee", "Student", "Self-employed", "Other"])
    c = st.columns(4)
    screen = c[0].slider("Screen hours per day", 1.0, 14.0, 6.0, 0.5); low = c[1].slider("Low battery moments per week", 0, 15, 3)
    anx = c[2].slider("Battery anxiety (1-5)", 1, 5, 3); away = c[3].slider("Hours away from a charger", 1.0, 16.0, 5.0, 0.5)
    c = st.columns(3)
    pb = c[0].checkbox("Carries a power bank"); sens = c[1].slider("Price sensitivity (1-5)", 1, 5, 3); gen = c[2].selectbox("Gender", ["Male", "Female"])
    probs = st.multiselect("Problems faced", list(CH.values()), [])
    row = {k: (df[k].median() if pd.api.types.is_numeric_dtype(df[k]) else df[k].mode()[0]) for k in FEATURES}
    row.update(age=age, monthly_income_aed=income, phone_brand=brand, occupation=occ, screen_hrs=screen, low_battery_per_week=low, battery_anxiety=anx,
               hrs_away_from_charger=away, carries_power_bank_flag=int(pb), price_sensitivity=sens, gender=gen, feature_count=3)
    for k, lab in CH.items():
        row[k] = int(lab in probs)
    for k in FT:
        row[k] = 0
    row["challenge_count"] = len(probs)
    X1 = pd.DataFrame([row])[FEATURES]
    pr = dict(zip(clf.classes_, clf.predict_proba(X1)[0]))
    wtp = float(reg.predict(X1)[0])
    r1, r2, r3 = st.columns(3)
    r1.metric("Most likely answer", max(pr, key=pr.get)); r2.metric("Chance of Yes", f"{pr['Yes']:.0%}"); r3.metric("Predicted max price", f"AED {wtp:.0f}")
    plot(px.bar(pd.Series(pr).reindex(CLASSES).mul(100).reset_index().rename(columns={"index": "Answer", 0: "%"}), x="Answer", y="%",
                color_discrete_sequence=[ORANGE], title="Predicted answer probabilities"), height=260)

# ------------------------------------------------------------------------------------------------ Clustering
with tabs[5]:
    K = R["clustering"]
    st.subheader("Algorithm 2: Clustering. Which customer groups exist?")
    st.markdown("**Why:** split respondents into high, medium and low propensity groups so each gets its own marketing. **Columns used:** " + ", ".join(K["features"]) + ".  \n"
                f"**Choice of k:** {K['reason']}")
    Zs = seg["scaler"].transform(df[K["features"]])
    pc = PCA(2, random_state=42).fit_transform(Zs)
    pdf = pd.DataFrame({"PC1": pc[:, 0], "PC2": pc[:, 1], "Segment": df["cluster_name"]})
    a, b = st.columns([3, 2])
    with a: plot(px.scatter(pdf, x="PC1", y="PC2", color="Segment", opacity=.6, title=f"Segments (k = {K['k_chosen']}) on the first two principal components"), height=420)
    with b: plot(px.line(x=K["k_range"], y=K["silhouette"], markers=True, labels={"x": "Number of clusters k", "y": "Silhouette"}, title="Silhouette score by k"), height=420)
    prof = pd.DataFrame(K["profiles"])
    show = prof[["name", "size", "share", "pct_yes", "pct_interested", "median_wtp", "mean_age", "median_income", "battery_anxiety", "hrs_away",
                 "price_sensitivity", "pct_iphone", "pct_powerbank"]]
    st.markdown("**Segment profiles**"); table(show.round(2))
    cen = pd.DataFrame(K["centroids_z"]).set_index("cluster")
    cen.index = [prof.loc[prof["cluster"] == i, "name"].iloc[0] for i in cen.index]
    plot(px.imshow(cen, text_auto=".1f", color_continuous_scale="RdBu_r", zmin=-2, zmax=2, aspect="auto", title="What makes each segment different (z-scores)"), height=300)

# ------------------------------------------------------------------------------------------------ Association
with tabs[6]:
    Aso = R["association"]
    st.subheader("Algorithm 3: Association rule mining. Which problems and features go together?")
    st.markdown("**Why:** instead of launching all 9 features at once, find which features people ask for together and release them in phases. "
                f"**Columns used:** {Aso['columns_used']}  \n"
                "**How to read a rule:** *support* = share of all respondents with both; *confidence* = of people with the left side, the share who also have the right side; "
                "*lift* above 1 means they go together more than chance.")
    st.markdown("#### Launch roadmap")
    c = st.columns(3)
    for col, (phase, items) in zip(c, Aso["roadmap"].items()):
        col.markdown(f"**{phase}**")
        col.dataframe(pd.DataFrame(items).rename(columns={"wanted_by_interested_pct": "Wanted by interested %", "also_wanted_given_phase1_pct": "Also wanted by Phase-1 fans %"}))
    st.markdown("#### Problem -> feature rules")
    (table(rules["problem_to_feature"].head(15)) if len(rules["problem_to_feature"]) else st.info("No rules reach the support, confidence and lift thresholds."))
    st.markdown("#### Feature -> feature rules")
    (table(rules["feature_to_feature"].head(15)) if len(rules["feature_to_feature"]) else st.info("No rules reach the support, confidence and lift thresholds."))
    st.markdown("#### Problem -> problem rules")
    (table(rules["problem_to_problem"].head(10)) if len(rules["problem_to_problem"]) else st.info("No rules reach the support, confidence and lift thresholds."))

# ------------------------------------------------------------------------------------------------ Regression
with tabs[7]:
    G = R["regression"]
    st.subheader("Algorithm 4: Regression. How much will they pay?")
    st.markdown("**Why:** price is a continuous number, so regression predicts each person's maximum price from their profile. This sets the price point per segment. "
                f"Respondents without a price answer ({int(df['wtp_missing'].sum())}) are left out of training and receive a predicted price afterwards.")
    mdl = pd.DataFrame(G["models"]); table(mdl)
    st.caption(f"Guessing the average price every time would be off by AED {G['baseline_mae_aed']} on average. Best model: {G['best_model']}.")
    a, b = st.columns(2)
    sc = px.scatter(reg_pred, x="actual", y="predicted", opacity=.5, title="Predicted vs actual price (test set)", color_discrete_sequence=[TEAL])
    sc.add_trace(go.Scatter(x=[40, 500], y=[40, 500], mode="lines", line=dict(dash="dash", color="black"), name="Perfect"))
    with a: plot(sc)
    with b: plot(px.bar(pd.Series(G["importance"]).sort_values(), orientation="h", title="What drives willingness to pay", color_discrete_sequence=[ORANGE], labels={"value": "Importance", "index": ""}))
    st.markdown("**Average error (AED) by income band, from cross-validation**")
    st.dataframe(pd.DataFrame(G["mae_by_income_band"], index=["MAE"]).T)

# ------------------------------------------------------------------------------------------------ Strategy
with tabs[8]:
    st.subheader("Prescriptive analytics: price, target customers, break-even and stress tests")
    st.markdown("Costs and market size are assumptions (see the table at the bottom). Change them and watch the verdict move.")
    c = st.columns(3)
    price = c[0].slider("Selling price incl. VAT (AED)", 99, 299, int(R["price_recommended"]), 5)
    ad = c[1].slider("Advertising cost per person reached (AED)", 0.5, 5.0, float(A["ad_cost_per_exposure"]), 0.1)
    expo = c[2].slider("Share of reachable people who see the offer (%)", 0.5, 8.0, float(A["exposure"] * 100), 0.5) / 100
    c = st.columns(3)
    landed = c[0].slider("Landed cost per case (AED)", 35, 90, int(A["landed_cost"]))
    s2m = c[1].slider("Survey-to-market factor", 0.2, 1.0, float(A["survey_to_market"]), 0.05)
    tacc = c[2].slider("Targeting accuracy (share of ads that land on the target segment)", 0.0, 1.0, float(A["targeting_accuracy"]), 0.05)
    prof = pd.DataFrame(R["clustering"]["profiles"])
    prof_sorted = prof.sort_values("pct_interested", ascending=False)
    opts = ["Everyone"] + [f"Top {n} segment{'s' if n > 1 else ''}" for n in range(1, len(prof_sorted))]
    rec_label = "Everyone" if not R["verdict"]["strategy"].startswith("Target") else R["verdict"]["strategy"].replace("Target top", "Top")
    strat = st.selectbox("Who do we advertise to?", opts, index=opts.index(rec_label) if rec_label in opts else 0)
    d2 = df.copy()
    d2["iv"] = d2["intent"].map({"Yes": A["w_yes"], "Maybe": A["w_maybe"], "No": 0.0})

    def conv(p, frame):
        return float((frame["iv"] * (frame["wtp_final"] >= p)).mean()) * s2m

    if strat == "Everyone":
        eff, prem = conv(price, d2), 1.0
    else:
        n = int(strat.split()[1]); sel = prof_sorted["cluster"].tolist()[:n]
        eff = tacc * conv(price, d2[d2["cluster"].isin(sel)]) + (1 - tacc) * conv(price, d2); prem = A["targeting_premium"]
    audience = A["smartphone_users"] * A["target_age_share"] * A["model_coverage"]
    units = audience * expo * eff
    net = price / (1 + A["vat"])
    pre = net - landed - A["platform_fee"] * net - A["fulfillment"] - A["returns_pct"] * net
    cac = ad * prem / eff if eff > 0 else float("inf")
    cm = pre - cac
    cum, cash, be = -A["one_off"], [], None
    for yr, g in enumerate(A["growth"]):
        for m in range(12):
            cum += units * g / 12 * cm - A["fixed_year"] / 12
            cash.append(cum)
            if be is None and cum >= 0: be = yr * 12 + m + 1
    c = st.columns(5)
    c[0].metric("Buy rate of people reached", f"{eff:.1%}"); c[1].metric("Cases in Year 1", f"{units:,.0f}")
    c[2].metric("Marketing cost per sale", f"AED {cac:,.0f}"); c[3].metric("Profit per case", f"AED {cm:,.1f}")
    c[4].metric("Break-even", f"Month {be}" if be else "Not within 3 years")
    (st.success if cash[-1] > 0 and be and be <= 24 else st.warning if cash[-1] > 0 else st.error)(
        f"Cumulative profit after 3 years: AED {cash[-1]:,.0f}. Launch cash needed: AED {A['one_off'] + 1000 * landed:,.0f} (one-off costs plus a first batch of 1,000 cases).")
    cf = go.Figure(go.Scatter(x=list(range(1, 37)), y=cash, mode="lines", line=dict(color=TEAL, width=3), name="Cumulative cash"))
    cf.add_hline(y=0, line_color="black"); cf.update_layout(title="Cumulative cash position (AED), month by month", xaxis_title="Month")
    plot(cf)

    st.markdown("#### Price: what the survey says about each price point")
    pc_df = pd.DataFrame(R["price_curve"])
    a, b = st.columns(2)
    with a: plot(px.line(pc_df, x="price", y="conversion_pct", markers=True, title="Buy rate by price (%), mass market", color_discrete_sequence=[TEAL]))
    with b: plot(px.line(pc_df, x="price", y=pc_df["cum_y3"] / 1000, markers=True, title="3-year cumulative profit by price (AED thousand), mass market",
                         color_discrete_sequence=[ORANGE], labels={"y": "AED thousand"}))
    st.markdown("#### Strategies compared at the recommended price")
    st_df = pd.DataFrame(R["strategies"]).T[["conversion", "units_y1", "cac", "cm_unit", "breakeven_month", "cum_y1", "cum_y2", "cum_y3"]].round(3)
    st.dataframe(st_df)
    cfs = go.Figure()
    for k, v in R["cashflows"].items():
        cfs.add_trace(go.Scatter(x=list(range(1, 37)), y=v, mode="lines", name=k))
    cfs.add_hline(y=0, line_color="black"); cfs.update_layout(title="Cumulative cash by strategy (AED)", xaxis_title="Month")
    plot(cfs)
    st.markdown("#### Edge cases: what if things go wrong? (recommended strategy)")
    st.dataframe(pd.DataFrame(R["edge_cases"]))
    st.markdown("#### Sensitivity: 3-year profit (AED) by price and advertising cost per person reached")
    sens = pd.DataFrame(R["sensitivity"]).set_index("price")
    sens.columns = [f"AED {c.split('_')[1]} per person" for c in sens.columns]
    plot(px.imshow(sens, text_auto=",.0f", color_continuous_scale="RdYlGn", color_continuous_midpoint=0, aspect="auto", title="Green = pays back, red = loses money"), height=380)
    st.markdown("#### Who to target, where to sell, which format")
    a, b, c = st.columns(3)
    with a:
        st.markdown("**Where the target segments want to buy**")
        st.dataframe(pd.Series(R["channels_target"]["purchase"], name="%"))
    with b:
        st.markdown("**Where they discover accessories**")
        st.dataframe(pd.Series(R["channels_target"]["discovery"], name="%"))
    with c:
        st.markdown("**Preferred format by phone brand (%)**")
        st.dataframe(pd.DataFrame(R["format_by_brand"]))
    st.markdown("#### Assumptions (replace with real quotes before relying on the numbers)")
    st.json({k: v for k, v in A.items()})
