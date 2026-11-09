# Key numbers for your report (facts only, write the explanations yourself)

## Data
- Raw: 1545 rows x 33 columns. Clean: 1440 rows x 61 columns.
- Purchase answers: {'No': 65.2, 'Maybe': 23.5, 'Yes': 11.3}
- Median price people would pay: AED 180.0

## Cleaning log (step, rows affected)
- 1 Duplicates: Dropped repeated respondent_id (45 rows)
- 2 Types: Parsed responded_at from two text formats to datetime (1500 rows)
- 2 Types: Converted 15 numeric columns with to_numeric (1500 rows)
- 3 Labels: Standardised emirate spellings (dubai, DXB, DUBAI, Dubai.) (4 rows)
- 3 Labels: Standardised phone brand (iphone, Apple, IPHONE, samsung) (6 rows)
- 3 Labels: Mapped yes/Y/YES/1 and no/N/NO/0 to Yes/No in 3 columns (1500 rows)
- 4 Quality: Removed respondents who finished in under 1.5 minutes (60 rows)
- 5 Outliers: Set impossible age (outside 16 to 80) to missing (8 rows)
- 5 Outliers: Set impossible monthly_income_aed (outside 500 to 100000) to missing (10 rows)
- 5 Outliers: Set impossible screen_hrs (outside 0.5 to 18) to missing (5 rows)
- 5 Outliers: Set impossible low_battery_per_week (outside 0 to 25) to missing (4 rows)
- 5 Outliers: Set impossible accessories_per_year (outside 0 to 30) to missing (4 rows)
- 5 Outliers: Set impossible max_price_aed (outside 20 to 1000) to missing (10 rows)
- 5 Outliers: Set impossible hrs_away_from_charger (outside 0 to 20) to missing (0 rows)
- 5 Outliers: Capped monthly_income_aed at Q3 + 3*IQR (44,600) (18 rows)
- 5 Outliers: Capped powerbank_spend_aed at Q3 + 3*IQR (292) (5 rows)
- 5 Outliers: Capped case_spend_aed at Q3 + 3*IQR (238) (18 rows)
- 6 Missing: Filled numeric gaps with the median (income by occupation) and categories with the mode (423 rows)
- 6 Missing: Left max_price_aed (regression target) unfilled and flagged with wtp_missing (66 rows)
- 7 Transform: Labelled blank 'problems faced' answers as 'None of these' (222 rows)
- 7 Transform: Split 2 multi-select answers into 16 yes/no columns (ch_*, ft_*) (1440 rows)
- 8 Transform: Created age_group, income_band, battery_need_score, log_income, challenge_count, feature_count, binary flags (1440 rows)

## Classification (target: Yes / Maybe / No)
- Best model by cross-validated macro F1: Random Forest
- Majority-class guess accuracy on the test set: 0.653
- Logistic Regression: test accuracy 0.719, test macro F1 0.628, ROC-AUC 0.853, CV macro F1 0.58
- Decision Tree: test accuracy 0.542, test macro F1 0.464, ROC-AUC 0.703, CV macro F1 0.522
- Random Forest: test accuracy 0.729, test macro F1 0.581, ROC-AUC 0.851, CV macro F1 0.582
- Gradient Boosting: test accuracy 0.729, test macro F1 0.609, ROC-AUC 0.86, CV macro F1 0.581
- Confusion matrix (rows = actual No, Maybe, Yes; columns = predicted): [[171, 14, 3], [34, 27, 7], [7, 13, 12]]
- Top drivers: {'battery_anxiety': 0.1454, 'price_sensitivity': 0.0473, 'screen_hrs': 0.0437, 'challenge_count': 0.0408, 'Missed important call or OTP': 0.0128, 'emirate': 0.0096, 'High capacity (4,000 mAh+)': 0.0091, 'low_battery_per_week': 0.0088, 'accessories_per_year': 0.0081, 'case_spend_aed': 0.0075, 'uses_case_flag': 0.0071, 'gender': 0.0057}

## Clustering
- k chosen = 3; silhouette by k (2 to 7) = [0.128, 0.12, 0.117, 0.115, 0.106, 0.107]
- High propensity: high battery anxiety: 26% of respondents, 31% Yes, 72% Yes or Maybe, median price AED 190, median income AED 10100
- Medium propensity: higher income: 38% of respondents, 7% Yes, 31% Yes or Maybe, median price AED 220, median income AED 13300
- Low propensity: lower income: 36% of respondents, 2% Yes, 12% Yes or Maybe, median price AED 130, median income AED 3750

## Association rules
- Itemsets above min support 0.04: 280
- Phase 1 (launch): Slim design, Fast charging, High capacity (4,000 mAh+)
- Phase 2 (+3 to 6 months): Wireless / MagSafe pass-through, Drop protection, 2-year warranty
- Phase 3 (later): Card slot or kickstand, LED battery indicator, Eco-friendly material
- Rule: if No charger or outlet available + Battery drains while travelling then High capacity (4,000 mAh+) (support 0.05, confidence 0.655, lift 1.45)
- Rule: if Phone died before end of workday + Battery drains while travelling then High capacity (4,000 mAh+) (support 0.061, confidence 0.62, lift 1.38)
- Rule: if Missed important call or OTP then Fast charging (support 0.071, confidence 0.607, lift 1.35)
- Rule: if High capacity (4,000 mAh+) + LED battery indicator then Fast charging (support 0.087, confidence 0.665, lift 1.48)
- Rule: if Slim design + Eco-friendly material then Wireless / MagSafe pass-through (support 0.079, confidence 0.588, lift 1.47)
- Rule: if Card slot or kickstand + 2-year warranty then Drop protection (support 0.058, confidence 0.638, lift 1.46)

## Regression (target: maximum price in AED)
- Ridge Regression: R2 0.629, MAE AED 40.197, RMSE AED 52.764, CV R2 0.613
- Random Forest: R2 0.624, MAE AED 39.98, RMSE AED 53.108, CV R2 0.577
- Gradient Boosting: R2 0.637, MAE AED 38.951, RMSE AED 52.191, CV R2 0.603
- Best: Ridge Regression. Guessing the mean every time gives MAE AED 66.6
- Top drivers: {'monthly_income_aed': 0.3744, 'case_spend_aed': 0.1035, 'price_sensitivity': 0.0927, 'phone_brand': 0.0712, 'battery_anxiety': 0.0183, 'brand_importance': 0.0149, 'occupation': 0.0058, 'Power bank and cables too bulky': 0.0029, 'screen_hrs': 0.0019, '2-year warranty': 0.0012}

## Strategy and break-even
- Verdict: CONDITIONALLY SUSTAINABLE
- Best price by 3-year profit (mass market): AED 189; recommended price AED 189
- Recommended: Target top 1 segment -> ['High propensity: high battery anxiety']
- Mass market (everyone): buy rate 4.0%, 1,965 cases in Year 1, marketing cost per sale AED 50, profit per case AED 22.2, break-even month 33, cumulative profit Y3 AED 17,075
- Target top 1 segment: buy rate 6.6%, 3,234 cases in Year 1, marketing cost per sale AED 36, profit per case AED 35.8, break-even month 14, cumulative profit Y3 AED 337,642
- Target top 2 segments: buy rate 5.0%, 2,454 cases in Year 1, marketing cost per sale AED 48, profit per case AED 24.2, break-even month 25, cumulative profit Y3 AED 86,932

## Stress tests (recommended strategy)
- Base (recommended strategy): Pays back, Y3 profit AED 337,642, break-even month 14
- Competitor magnetic packs cut conversion by 25%: Pays back, Y3 profit AED 77,850, break-even month 26
- Survey bias is worse than assumed (another 50% cut): Loses money, Y3 profit AED -181,942, break-even month None
- Supplier cost rises 20%: Pays back, Y3 profit AED 190,834, break-even month 18
- Advertising costs double: Loses money, Y3 profit AED -186,884, break-even month None
- Reach is half as large (1.5% exposure): Pays back, Y3 profit AED 80,321, break-even month 26
- Price cut to AED 149: Pays back, Y3 profit AED 189,822, break-even month 18
- Growth stalls after Year 1: Pays back, Y3 profit AED 169,950, break-even month 14