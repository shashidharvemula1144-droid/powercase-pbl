# PowerCase: validating a phone case with a built-in battery (UAE)

Individual PBL, Data Analytics. Business idea: a slim phone case with a built-in rechargeable battery, sold to phone users in the UAE.
Objective: decide whether the idea is sustainable, whom to target, which features to launch first, what price to charge, and when it breaks even.

## How to run (Python 3.10 or newer)
```
pip install -r requirements.txt
python run_all.py                 # runs steps 02, 03, 04 and 06 in order (about 1 minute)
streamlit run 05_dashboard.py     # opens the dashboard in the browser
```
Everything is already generated in this folder, so the dashboard also runs without step `run_all.py`. All scripts use relative paths and a fixed random seed, so results repeat exactly.

## Folder contents
| File or folder | What it is |
|---|---|
| `01_survey_questionnaire.md` | The 30-question survey and which question feeds which algorithm |
| `02_generate_survey_data.py` | Creates the synthetic survey with deliberate noise, outliers, missing values and duplicates |
| `03_clean_transform.py` | Data cleaning and transformation, with a reason for every step |
| `04_analysis.py` | Descriptive, diagnostic, predictive (4 algorithms) and prescriptive analytics |
| `05_dashboard.py` | Streamlit dashboard |
| `06_report_numbers.py` | Prints the key numbers into `outputs/report_numbers.md` |
| `data/` | `powercase_survey_raw.csv`, `powercase_survey_clean.csv`, `cleaning_log.csv` |
| `outputs/` | Result tables (`results.json`, model comparisons, rules, cluster profiles, price curve, sensitivity) |
| `charts/` | PNG charts produced by the analysis |
| `models/` | Saved classifier, regressor and segment model used by the dashboard |

## What each algorithm does
| Algorithm | Business question | Columns | Accuracy check |
|---|---|---|---|
| Classification (logistic regression, decision tree, random forest, gradient boosting) | Will this person buy: Yes, Maybe or No? | Survey answers before the purchase question; max price and pre-order excluded | Accuracy, macro F1, ROC-AUC, 5-fold cross-validation, confusion matrix, comparison with a majority-class guess |
| Clustering (k-means) | Which customer segments exist? | Habit, need and spending columns | Silhouette score for k = 2 to 7, segment profiles |
| Association rule mining (own Apriori code) | Which problems and features go together, so what launches first? | Only the 7 problem and 9 feature yes/no columns | Support, confidence and lift |
| Regression (ridge, random forest, gradient boosting) | How much will each person pay? | Survey answers before the price question | R2, MAE, RMSE, 5-fold cross-validation, comparison with guessing the mean |

## Prescriptive layer
Price curve, three advertising strategies (everyone, top segment, top two segments), month-by-month break-even, 8 stress tests (edge cases) and a price by advertising-cost sensitivity table.

## Assumptions (all editable in `04_analysis.py`, dictionary `A`)
Market size comes from public sources: 7.46 million UAE smartphone users (Statista, 2025) and 62.7% aged 18 to 44 (DataReportal, 2025).
Everything else is an assumption to replace with real quotes: landed cost AED 51, marketplace fees 18%, fulfilment AED 10, returns 8%, one-off costs AED 105,000 (certification, tooling, brand, samples),
fixed costs AED 24,000 a year, advertising AED 2 per person reached, 3% of reachable people see the offer, 35% of users own the supported phone models.
Stated intent is discounted (70% of Yes, 20% of Maybe) and halved again because survey respondents are keener than the general public.

## Important
The survey data is synthetic. Relationships were designed so the algorithms have something to find, so the results show the method and not the real market.
When the real survey is done, put the responses in `data/powercase_survey_raw.csv` with the same columns and rerun `03_clean_transform.py`, `04_analysis.py` and the dashboard.

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repo (files at the repo root).
2. share.streamlit.io -> New app -> pick the repo, branch main, main file path: 05_dashboard.py
3. Deploy. Data, models and charts are committed, so no re-run is needed.
