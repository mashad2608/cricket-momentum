# Overnight results summary

Everything in the frozen plan has now been run, for men and women, with no errors. Below: what came out, what it means, and what to check.

- Code is explained in `CODE_GUIDE.md`.
- Every change to the plan is in the deviations log at the end of `ANALYSIS_PLAN.md`.

## How the confirmation was protected

- **Men were analysed first** (exploratory), and the code was fixed until it was right.
- **The code was then locked**, with checksums in `data/out/code_frozen_before_women.sha256`, and the women's data was run once.
- **The first women's attempt crashed** during feature selection, before any model or result. The cause was two innings with 101 legal balls, which are genuine umpire miscounts. Fixing it changed no men's results (verified). It is logged, and the women's run was restarted from scratch.
- **Nothing was tuned after seeing women's results.**

## Headline findings

| | Men | Women |
|---|---|---|
| Matches (full) | 189 | 181 |
| Final model, 2026 test log-loss (baseline) | 1.4915 (1.5394) | 1.4311 (1.4810) |
| 2026 test R² for runs | 0.028 | 0.034 |
| Chase win-probability AUC, 2026 | 0.895 | 0.873 |

### Thesis: trends v single balls ✅ for description, ❌ for forecasting

R² of expected v actual runs, first innings:

| Window | Men, describing | Men, forecasting | Women, describing | Women, forecasting |
|---|---|---|---|---|
| 1 ball | 0.035 | 0.023 | 0.032 | 0.024 |
| 10 balls | 0.249 | 0.043 | 0.219 | 0.015 |
| 50 balls | 0.397 | 0.047 | 0.363 | 0.039 |
| Whole innings | 0.404 | −0.090 | 0.361 | −0.071 |

**Meaning:** a trend *describes* a passage of play well, and much better than a single ball does. The same holds in both competitions, and chases behave similarly.

**Correction from the review below:** the near-zero forecast scores came from the boosted regressor *overfitting*. A simple ridge model forecasts 10–50 ball passages with R² of about 0.06–0.13. So forecasting is **modestly possible, but far weaker than describing**.

### RQ1: does momentum carry forward? ❌ not as streaks, in either competition

10-ball windows, both innings pooled:

| | Slope | 95% interval | Shuffled-data slope | Shuffle p | H1 supported? |
|---|---|---|---|---|---|
| Men | +0.082 | +0.032 to +0.131 | +0.087 | 0.60 | No |
| Women | +0.057 | +0.007 to +0.109 | +0.051 | 0.37 | No |

**Meaning:** a good 10 balls is followed by a slightly better next 10. But shuffling the ball order within each innings gives the *same* slope. So the persistence comes from **whole-innings form** (a team having a good or bad day), not from streaks or "momentum" in the usual sense. The 5-ball windows agree.

> In `data/out/compare/rq4_comparison.csv`, the "confirmed" label for RQ1 only checks the slope's sign and interval, not the shuffle test. Under the plan's full decision rule, **H1 is not supported in either competition**.

### RQ2: model v scoreboard momentum ✅ confirmed in both

| | Men | Women |
|---|---|---|
| Disagreement, first innings | 11% | 13% |
| Disagreement, chases | 23% | 23% |
| H2a: chase − first innings | +12.4 points (+10.6 to +14.1) | +10.6 points (+8.9 to +12.3) |
| H2b: model − scoreboard, predicting the next window | +0.051 (+0.029 to +0.073) | +0.038 (+0.013 to +0.062) |

**Meaning:** scoreboard momentum misleads mostly in chases. The model-based measure predicts the next window's over-performance better, though both correlations are small.

### RQ3: coaching decisions (PSM): no effect found (only one decision had enough cases to test)

| Decision | Men | Women |
|---|---|---|
| D1 keep v change bowler after an expensive set | Not estimable: only 20 cases of the same bowler continuing | Not estimable: 5 cases |
| D2 ten straight v split | Balance failed (max SMD 0.150); not estimable | Balance failed (0.244); not estimable |
| D3 fresh bowler after a hot spell | +0.19 (−0.27 to +0.64), p = 0.53, balanced | −0.21 (−1.03 to +0.15), p = 0.20, balanced |
| D4 promoted big hitter | Not estimable: 5 cases | Not estimable: 12 cases |

**Meaning:** captains almost never keep an expensive bowler on, and promotions of big hitters are rare, so there is too little data. For a fresh bowler after a hot spell, there's no detectable effect in either competition. This is honest and reportable, but it is the weakest part of the paper.

### SHAP: what drives the probabilities (similar in men and women)

- **Strongest drivers:** stage of innings (ball number, balls left, score), how long the batter and partnership have lasted, recent strike rate, and player history (boundary and dot rates).
- **Partial dependence (charts in `figures/<gender>/partial_dependence.png`):**
  - expected runs drop when the powerplay ends at ball 25
  - scoring and wicket risk rise at the death
  - batters settle after about 10 balls
- **Ablation:**
  - Men: match state alone brings most of the gain; history alone adds nothing; recent form adds more.
  - Women: history helps a little more.

### RQ4 and transfer

- Findings line up closely across men and women. No men v women difference is significant.
- The men's model works on women's data, but less well than the women's own model: log-loss 1.434 against 1.419. It over-predicts women's boundaries (16.5% against 15.5% actual).

## Review and extra analyses (exploratory, added after the frozen run)

A critical review of the method raised one conceptual issue. The model's *expected* value already uses recent form, so the team-level momentum test (RQ1) only looks for momentum *beyond* recent form, and may control the real thing away. Four extra checks followed (`src/robustness.py`, `src/compare_models.py`). They are exploratory, so **the women's data does not confirm them**.

| Check | Men | Women | What it means |
|---|---|---|---|
| **Batter-level streaks, 5 balls** (shuffle within each batter's innings) | slope +0.146 (+0.094 to +0.200), shuffle p = 0.001 | +0.067 (+0.003 to +0.136), p = 0.001 | **Individual batters run hot over about 5 balls** |
| Batter-level streaks, 10 balls | +0.118, p = 0.19 | +0.125, p = 0.03 | The effect fades over longer spells |
| Recent performance beyond "being set" (log-loss change) | −0.0007 (−0.0014 to −0.00005) | −0.0010 (−0.0019 to −0.00005) | Hot form helps prediction a little, beyond balls faced |
| Forecast with a simple ridge model (R²) | 0.10 / 0.11 / 0.13 for 10 / 20 / 50 balls | 0.09 / 0.11 / 0.06 | The boosted forecast in the primary test **overfitted**. Passages are forecastable, but only modestly |
| Leakage test on the women's build | passed (6 cut points) | passed | ✓ |

**Cautions on the batter-level result**

- It could be a genuine hot hand, or it could be batters settling in faster than the model's average settling curve. A follow-up should separate the two (for example, control for balls faced within the window).
- Because it was found after looking at the data, it must be confirmed on **fresh data** with a pre-written test. For example: the Vitality Blast, or other T20 leagues on Cricsheet.

### Model comparison: 7 models and 5 combinations, same features and folds

**Men, cross-validated log-loss (2022–2025)**

| Model | Log-loss | Δ v locked model (95% interval) |
|---|---|---|
| **LightGBM + random forest (locked model)** | **1.4949** | — |
| Random forest | 1.4954 | +0.0004 (−0.0002 to +0.0010) |
| LightGBM + XGBoost + random forest | 1.4955 | +0.0006 (+0.0003 to +0.0009) |
| All seven models | 1.4956 | +0.0007 (−0.0003 to +0.0017) |
| All tree models | 1.4959 | +0.0009 |
| LightGBM | 1.4968 | +0.0019 |
| Extra trees | 1.4969 | +0.0019 |
| LightGBM + XGBoost | 1.4970 | +0.0021 |
| XGBoost | 1.4980 | +0.0031 |
| Logistic regression | 1.5092 | +0.0143 |
| HistGradientBoosting | 1.5143 | +0.0194 |
| Neural network (MLP) | 1.5290 | +0.0340 |

**Women**

- Best: *all seven models*, at 1.4148 against 1.4151 for the locked model. The difference is −0.0004 (−0.0012 to +0.0004): not significant.
- The 2026 test (an exploratory second look) gives the same ranking in both competitions.

**Conclusion: keep the locked model.**

- No model or combination beats it significantly in either competition.
- Tree ensembles are best; the neural network and linear models are clearly worse.
- The ceiling comes from cricket's randomness, not from the algorithm. Gains between the top models are about 0.001 log-loss.
- CatBoost isn't installed, so it wasn't tested. Given these results, it would very likely tie as well.

## What the paper's story now is

> In The Hundred, team momentum is mostly an illusion. What looks like it is a team's form on the day, not a streak that carries forward. But **individual batters do run hot over about 5 balls**, in both the men's and women's competitions (exploratory, to be confirmed). Trends describe a match far better than single balls, while forecasting the next passage is only modestly possible. A situation-adjusted measure beats scoreboard momentum, especially in chases.

That's a stronger paper than the null result alone. The batter-level finding should be pre-registered and confirmed on a fresh competition before it becomes the headline.

## Things to decide or check together

1. **RQ3 is thin.** Three of the four decisions can't be estimated. Options: keep it as an honest null and limitation, or plan a follow-up with bowling-style data (a new, separate analysis, clearly labelled).
2. **Comparison verdicts.** In the RQ1 rows of `rq4_comparison.csv`, "confirmed" means slope only, as explained above.
3. **Forecast R² is noisy** for 50- and 100-ball windows, which have few examples.
4. **The earlier notebooks (01–05b) are men-only development work.** Final numbers come from `src/pipeline.py`. Small differences from notebook numbers (for example 2026 log-loss 1.4901 against 1.4915) are because the final pipeline adds the eight new features.
5. **Not yet built:** Power BI export tables, the GitHub README polish, and the paper itself.

## Where things are

| What | Where |
|---|---|
| Logs of every run | `data/out/male/log.txt`, `data/out/female/log.txt` |
| All results | `data/out/<gender>/results.json`, `rq3_psm.csv`, `shap_groups.csv`, `test_2026.csv` |
| Men v women | `data/out/compare/rq4_comparison.csv`, `transfer.json` |
| Plan and deviations | `ANALYSIS_PLAN.md` |
| Code explanation | `CODE_GUIDE.md` |
