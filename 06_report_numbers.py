"""Step 5 - Prints the key numbers from the analysis into outputs/report_numbers.md.
These are FACTS to look up while you write your own report. The report itself must be in your own words."""
import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
R = json.load(open(HERE / "outputs" / "results.json"))
raw = pd.read_csv(HERE / "data" / "powercase_survey_raw.csv")
clean = pd.read_csv(HERE / "data" / "powercase_survey_clean.csv")
log = pd.read_csv(HERE / "data" / "cleaning_log.csv")
C, G, K, S, V = R["classification"], R["regression"], R["clustering"], R["association"], R["verdict"]
L = []
L += ["# Key numbers for your report (facts only, write the explanations yourself)", ""]
L += ["## Data", f"- Raw: {len(raw)} rows x {raw.shape[1]} columns. Clean: {len(clean)} rows x {clean.shape[1]} columns.",
      f"- Purchase answers: {R['intent_share']}", f"- Median price people would pay: AED {R['wtp_median']}", ""]
L += ["## Cleaning log (step, rows affected)"] + [f"- {r.step}: {r.action} ({r.rows_affected} rows)" for r in log.itertuples()] + [""]
L += ["## Classification (target: Yes / Maybe / No)", f"- Best model by cross-validated macro F1: {C['best_model']}",
      f"- Majority-class guess accuracy on the test set: {C['baseline_accuracy']}"]
L += [f"- {m['model']}: test accuracy {m['test_accuracy']}, test macro F1 {m['test_macro_f1']}, ROC-AUC {m['test_roc_auc']}, CV macro F1 {m['cv_macro_f1']}" for m in C["models"]]
L += [f"- Confusion matrix (rows = actual No, Maybe, Yes; columns = predicted): {C['confusion_matrix']}", f"- Top drivers: {C['importance']}", ""]
L += ["## Clustering", f"- k chosen = {K['k_chosen']}; silhouette by k (2 to 7) = {K['silhouette']}"]
L += [f"- {p['name']}: {p['share']:.0%} of respondents, {p['pct_yes']:.0%} Yes, {p['pct_interested']:.0%} Yes or Maybe, median price AED {p['median_wtp']:.0f}, median income AED {p['median_income']:.0f}" for p in K["profiles"]] + [""]
L += ["## Association rules", f"- Itemsets above min support {S['min_support']}: {S['n_itemsets']}"]
L += [f"- {ph}: " + ", ".join(x['feature'] for x in items) for ph, items in S["roadmap"].items()]
L += [f"- Rule: if {r['if_customer']} then {r['then_also']} (support {r['support']}, confidence {r['confidence']}, lift {r['lift']})" for r in S["problem_to_feature"][:3] + S["feature_to_feature"][:3]] + [""]
L += ["## Regression (target: maximum price in AED)"]
L += [f"- {m['model']}: R2 {m['test_r2']}, MAE AED {m['test_mae_aed']}, RMSE AED {m['test_rmse_aed']}, CV R2 {m['cv_r2']}" for m in G["models"]]
L += [f"- Best: {G['best_model']}. Guessing the mean every time gives MAE AED {G['baseline_mae_aed']}", f"- Top drivers: {G['importance']}", ""]
L += ["## Strategy and break-even", f"- Verdict: {V['label']}", f"- Best price by 3-year profit (mass market): AED {R['price_best_by_profit']}; recommended price AED {R['price_recommended']}",
      f"- Recommended: {V['strategy']} -> {V['target_segments']}"]
for k, v in R["strategies"].items():
    L.append(f"- {k}: buy rate {v['conversion']:.1%}, {v['units_y1']:,.0f} cases in Year 1, marketing cost per sale AED {v['cac']:.0f}, profit per case AED {v['cm_unit']:.1f}, "
             f"break-even month {v['breakeven_month']}, cumulative profit Y3 AED {v['cum_y3']:,.0f}")
L += ["", "## Stress tests (recommended strategy)"] + [f"- {e['scenario']}: {e['status']}, Y3 profit AED {e['cum_y3']:,}, break-even month {e['breakeven_month']}" for e in R["edge_cases"]]
(HERE / "outputs" / "report_numbers.md").write_text("\n".join(L))
print("Saved outputs/report_numbers.md")
