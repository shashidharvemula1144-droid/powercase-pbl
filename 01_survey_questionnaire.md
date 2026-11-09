# PowerCase Consumer Survey (30 questions)

**Business idea:** A phone case with a built-in battery ("PowerCase") sold to phone users in the UAE.
**Why a survey:** Secondary research gives market size and competitor prices. It cannot say who will buy, at what price, or which features to launch first. These 30 questions collect the columns the four algorithms need.

**Intro shown to respondents:** "PowerCase is a slim phone case with a built-in rechargeable battery. It gives your phone an extra charge without a separate power bank or cable. Please answer based on how you use your phone today." (Questions 17 to 30 are answered after reading this.)

| # | Question | Answer type | Column in dataset | Used by |
|---|----------|-------------|-------------------|---------|
| **A. About you** | | | | |
| 1 | What is your age? | Number | `age` | All |
| 2 | What is your gender? | Single choice | `gender` | Classification, clustering |
| 3 | Which emirate do you live in? | Single choice | `emirate` | Classification, descriptive |
| 4 | What is your occupation? | Single choice | `occupation` | Classification, clustering |
| 5 | What is your monthly income or allowance (AED)? | Number | `monthly_income_aed` | Regression, clustering |
| 6 | Which phone brand do you use? | Single choice | `phone_brand` | Classification, regression |
| 7 | How old is your current phone (years)? | Number | `phone_age_years` | Classification |
| **B. Battery habits** | | | | |
| 8 | Hours per day you use your phone screen? | Number | `screen_hrs` | All |
| 9 | How many times a week does your battery fall below 20%? | Number | `low_battery_per_week` | All |
| 10 | How worried are you about running out of battery? (1 to 5) | Rating | `battery_anxiety` | All |
| 11 | Hours per day you are away from a charger? | Number | `hrs_away_from_charger` | Classification, clustering |
| 12 | Do you carry a power bank? | Yes / No | `carries_power_bank` | Classification, clustering |
| 13 | How much did you spend on power banks and charging accessories last year (AED)? | Number | `powerbank_spend_aed` | Regression, clustering |
| 14 | Do you use a phone case? | Yes / No | `uses_case` | Classification |
| 15 | How much did you pay for your current case (AED)? | Number | `case_spend_aed` | Regression, clustering |
| **C. Problems faced** | | | | |
| 16 | Which of these have happened to you in the last 3 months? (tick all) | Multi-select, 7 options | `challenges` → `ch_*` columns | **Association rules**, classification |
| **D. The PowerCase concept** | | | | |
| 17 | Which features would you want? (tick all) | Multi-select, 9 options | `desired_features` → `ft_*` columns | **Association rules**, classification |
| 18 | Which single feature matters most? | Single choice | `top_feature` | Descriptive |
| 19 | What is your biggest concern? | Single choice | `top_concern` | Descriptive, diagnostic |
| 20 | Which format do you prefer? | Single choice | `preferred_format` | Descriptive |
| 21 | Which battery capacity do you prefer? | Single choice | `preferred_capacity` | Descriptive |
| 22 | Where would you buy it? | Single choice | `purchase_channel` | Prescriptive (sales channel) |
| 23 | Where do you usually discover new accessories? | Single choice | `discovery_channel` | Prescriptive (marketing channel) |
| 24 | "I always choose the cheapest option." (1 to 5) | Rating | `price_sensitivity` | Regression, clustering |
| 25 | "Brand name matters when I buy accessories." (1 to 5) | Rating | `brand_importance` | Regression, classification |
| 26 | How many phone accessories do you buy per year? | Number | `accessories_per_year` | Regression, clustering |
| 27 | How many trips do you take per year? | Number | `trips_per_year` | Classification |
| 28 | What is the highest price you would pay for PowerCase (AED)? | Number | `max_price_aed` | **Regression target** |
| 29 | Would you like launch pre-order updates? | Yes / No | `preorder_signup` | Descriptive |
| 30 | Given the description, would you buy PowerCase? | Yes / Maybe / No | `intent` | **Classification target** |

Extra columns captured by the survey tool: `respondent_id`, `responded_at`, `completion_min` (used to remove careless responses).

## Which algorithm answers which business question

| Algorithm | Business question | Columns used |
|-----------|-------------------|--------------|
| Classification | Will this person buy: Yes, Maybe or No? | Q1 to Q27 as inputs, Q30 as target. Q28 and Q29 are excluded so the model can score people who have not answered them. |
| Clustering | Which customer groups exist, and which are most likely to convert? | Habit, need and spending columns (Q8 to Q15, Q24, Q26) plus the problem count from Q16 |
| Association rule mining | Which problems and features go together, so which features launch first? | Only Q16 and Q17 (16 yes/no items) |
| Regression | How much will each person pay? | Q1 to Q27 as inputs, Q28 as target |
