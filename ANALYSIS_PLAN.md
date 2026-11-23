# Analysis plan

**Working title:** Is Momentum Real? Situation-Adjusted Evidence from Men's and Women's Hundred Cricket
**Status:** FROZEN on 2026-11-23. Any later change goes in the deviations log at the end.

## 0. What is already known (honesty statement)

- **Men's data (exploratory).** The men's results for RQ1 and RQ2 have already been seen during development (Steps 5 and 5b), including the 2026 season. Men's findings are therefore *exploratory*.
- **Women's data (confirmatory).** The Women's Hundred has not been analysed. Only match and delivery counts have been read (188 matches, about 36,800 deliveries, 2021–2026). Every women's result is *confirmatory*: it is produced once, by the frozen pipeline.

## 1. Research questions and hypotheses

### Central thesis
In cricket, the **trend** over a passage of play is more informative than the value at any single ball. Single-ball outcomes are mostly noise. Signal builds up over windows of balls.

- **Exploratory evidence (men, first innings, 2022–2026).** The R² of expected v actual runs rises with window size: 0.04 for 1 ball, 0.17 for 5, 0.25 for 10, 0.33 for 20, 0.39 for 50 and 0.40 for a whole innings.
- **Caution.** In that check, the expectation updates within the window, so it describes the passage rather than forecasting it.
- **HT. In a fair test, explanatory power rises with window size.** Report two versions:
  - **(a) Descriptive:** the expectation updates ball by ball.
  - **(b) Forecast:** the expectation is fixed at the start of the window, using the state at the window's first ball.
- **Windows:** 1, 5, 10, 20, 50 balls and the whole innings.
- **Data:** men (exploratory) and women (confirmatory).
- **Answers:** RQ1–RQ3 ask what those trends contain and what changes them.

### RQ1 (main): Does momentum exist?
After accounting for match situation and the players involved, does performing above expectation predict performing above expectation next?

- **H1.** The persistence slope is greater than 0.
- **Primary test:** 10-ball windows, both innings pooled.
- **Secondary tests:** 5-ball windows (for comparison with the ~3–5 ball persistence reported by Mishra, 2026, an arXiv preprint not yet read in full; check it before citing), and each innings separately.

### RQ2: What does a model-based measure change?

- **H2a.** Model momentum and scoreboard momentum disagree more often in chases than in first innings.
- **H2b.** Model momentum predicts the next window's over-performance better than scoreboard momentum.

### RQ3: Which coaching decisions are linked to momentum shifts?
In comparable situations, how does each of these four decisions relate to what happens next (two-sided; no direction assumed)?

- **D1. Keep or change the bowler** after an expensive set
- **D2. Ten straight or split:** bowling a bowler's second set straight after the first, or later
- **D3. Change after a hot spell:** a fresh bowler after the batting side has performed well above expected
- **D4. Promote a big hitter** after an early wicket

### RQ4: Do the answers hold in the Women's Hundred?

- **H4.** RQ1–RQ3 effects have the same sign in women's data. Men v women differences are reported with intervals.
- **Transfer check:** does the men's model stay calibrated on women's data?

## 2. Data

- **Source:** Cricsheet ball-by-ball JSON for the Hundred, men and women, 2021–2026 (ODC-By licence).
- **No Hampshire Cricket Club data is used.**
- **Analysis sample:** full matches only (as defined in `01_cleaning`). The same rules apply to the women's data.

## 3. Definitions (frozen)

### Ball value

- **First innings:** `runs + E(runs still to come after the ball) − E(runs still to come before the ball)`, where E comes from a monotone model of balls left × wickets in hand.
- **Chase:** change in the batting side's win probability, from a monotone model of runs needed, balls left and wickets in hand.

### Expected value and gap

- **Expected value:** the average ball value under the classifier's outcome probabilities (classes W, 0, 1, 2, 3, 4, 5+), with legal and illegal deliveries handled separately.
- **Gap:** actual value − expected value.
- **Standardised gap:** the gap divided by its standard deviation within each innings type (first innings, chase), so that both innings can be pooled.

### Momentum

- **Momentum (k balls):** the sum of the gap over the last k deliveries in the same innings.
- **Scoreboard momentum:** runs in the last k deliveries, with each wicket counting −6.

### Windows

- Non-overlapping k-ball windows within an innings, starting at delivery k.
- `past` = sum of the gap over the window; `future` = sum of the gap over the next k deliveries.

### Sets and roles

- **Set:** each block of 5 legal balls in an innings (legal balls 1–5, 6–10, and so on), as the format is structured. A bowler bowling 10 in a row bowls two sets.
- **Expensive set:** a set conceding at least 10 runs, including extras. That is the 75th percentile, and 29% of men's 2021–2025 sets.
- **Hot spell:** standardised 10-ball momentum in the top quarter (cut-off fixed from men's 2021–2025 data).
- **Fresh bowler:** has not bowled either of the last two sets.
- **Usual position:** a batter's median batting position over earlier matches (needs at least 3 prior innings).
- **Promoted:** the batter comes in at least 2 places earlier than their usual position.
- **Big hitter:** historical strike rate (from `bat_hist_runs`) above the league median at that date.

## 4. Models (frozen procedure)

- **Features:** the 42 features chosen in Step 4, plus the new features in Section 4a. The new features must be defined and their code frozen *before* any outcome analysis that uses them.
- **Classifier:** the average of a LightGBM and a random-forest classifier. Tuning, forward-season folds and evaluation follow the Step 4 procedure.
- **Training, men:** forward folds within 2021–2025. Out-of-sample predictions for 2022–2025 come from the fold models; 2026 comes from the final model.
- **Training, women:** the identical procedure re-run on women's data. The procedure is frozen, not the fitted numbers.
- **Transfer check:** the men's final model applied unchanged to women's 2022–2026 data. Report log-loss, Brier score and calibration.

### 4a. New cricket-defined features (to finalise before freezing)

| Feature | Definition |
|---|---|
| Set batter | Faced at least 15 balls this innings |
| New batter | Faced at most 5 balls this innings |
| Bowler's set number | Which set of their spell this is |
| Consecutive set | Bowler bowled the previous set too |
| Balls since bowler last bowled | Legal balls since this bowler's previous delivery |
| Pressure index | Dots in the last 10 balls, plus the required-rate gap in chases |
| Strike rotation | Share of singles and twos among the batting side's last 10 scoring balls |
| Batter death-phase record | Historical boundary and dot rates in balls 76–100, shrunk towards the league average |

Each new feature must pass the Step 2 leakage test. It is kept only if the random-forest permutation rule from Step 3 selects it.

## 5. Statistical methods

### RQ1

- **Slope:** OLS of `future` on `past` (standardised gaps). Also report Spearman's ρ.
- **Interval:** 95% from a match-level bootstrap (2,000 resamples).
- **Shuffle test:** shuffle the order of gaps within each innings 1,000 times. p = share of shuffled slopes at or above the observed slope.
- **Decision rule:** H1 is supported if the interval excludes 0 *and* the shuffle p < 0.05.

### RQ2

- **H2a:** share of balls where the two measures have opposite signs, first innings v chase. Interval for the difference.
- **H2b:** difference in Spearman ρ with future over-performance (model − scoreboard). Match-level bootstrap interval.

### RQ3 (propensity score matching)

**Units and treatments**

| | Unit | Treated | Control |
|---|---|---|---|
| D1 | End of an expensive set | Same bowler bowls the next set | Change of bowler |
| D2 | A bowler's 2nd set | Bowled straight after their 1st set | Bowled later |
| D3 | A set break during a hot spell | Fresh bowler | Bowler who bowled one of the last two sets |
| D4 | First wicket within 40 legal balls, at most 3 down | Promoted big hitter | Batter at or near their usual position |

**Outcome**

- D1–D3: standardised gap over the next set (5 balls).
- D4: standardised gap over the next 20 deliveries.
- Secondary outcomes: runs (first innings) or win-probability points (chase), and wickets.

**Covariates**

- Match state: innings, ball number, balls left, wickets down, score, runs needed, required-rate gap.
- Recent play: last-10 standardised gap.
- Players: batter and bowler history (runs, boundary, dot and wicket rates); bowler balls bowled.
- Context: venue and season scoring.
- D1–D3 add the outgoing bowler's figures; D4 adds the team's usual-position data.

**Matching**

- Propensity score from logistic regression.
- 1:1 nearest-neighbour matching on the logit, caliper 0.2 standard deviations.
- Matching with replacement, but each control can be used at most 5 times.

**Diagnostics**

- Every covariate must have an absolute standardised mean difference below 0.1 after matching. If balance fails, report the decision as "not estimable".
- Report: the effective sample size, the share of treated units matched, and the overlap plot.

**Effect, interval and sensitivity**

- **Effect:** ATT (average effect on the treated).
- **Interval:** 95% from a match-level bootstrap with re-matching in each resample.
- **Sensitivity:** an E-value-style analysis. How strong would a hidden factor have to be to explain the result away?

**Multiple testing and wording**

- Holm correction across D1–D4, separately for men and women.
- Results are reported as associations in comparable situations, not as proven causes.

### SHAP (descriptive, no hypothesis tests)

- TreeSHAP on the LightGBM part of the final model.
- Before attributing, group features whose |Spearman ρ| is above 0.8, and report SHAP by group.
- Also: partial-dependence plots for the top features, and a layered ablation (match state → + player history → + recent form).

### RQ4

- Men v women difference for each estimate, with an interval from independent bootstraps.
- **Sign agreement:** the sign must agree and the women's interval must exclude 0 for "confirmed". Otherwise report "not confirmed" (and say whether the result was a null or the opposite sign).

## 6. Reporting rules

- Report every test in this plan, including null and opposite-direction results.
- No extra outcomes, windows, thresholds or subgroups after freezing, except as clearly labelled exploratory analyses.
- Code, plan and results are published together (GitHub plus an arXiv preprint).

## 7. Deviations log

| Date | Change | Reason |
|---|---|---|
| 2026-11-23 | **Thesis forecast version.** The first implementation multiplied the first ball's expected runs by the window length. Replaced with a regressor that predicts the window's runs from the features at the window's first ball, trained on earlier seasons only (same forward scheme as the classifier). | The first version ignored phase changes and innings ending, so it was not a fair forecast. Fixed after seeing the men's results, before any women's analysis. |
| 2026-11-23 | **PSM covariates clipped** at their 1st and 99th percentiles before fitting the propensity model. | Heavy tails (for example the required-rate gap off the last few balls) destabilise logistic regression. The plan did not specify this. Decided before any PSM result was seen. |
| 2026-11-23 | **D4 unit read as any of the first three wickets** falling within the first 40 legal balls. | The plan's wording ("first wicket within 40 legal balls, at most 3 down") was ambiguous. Decided before any PSM result was seen. |
| 2026-11-23 | **D4 covariates leave out the incoming batter's own history.** | That history defines the treatment (big hitter), so including it would remove all overlap. |
| 2026-11-23 | **Hot-spell cut-off** taken from men's 2022–2025 sets, not 2021–2025. | Out-of-sample momentum does not exist for 2021 (there is no earlier season to train on). The same men's cut-off is used for women, as planned. |
| 2026-11-23 | **Covariate groups for SHAP** use complete-linkage clustering, so every pair within a group has \|Spearman ρ\| ≥ 0.8. | This is one concrete way to implement "group features with \|ρ\| > 0.8". |
| 2026-11-23 | **Infinite feature values set to missing** (for example the required rate when 0 balls remain). | Two women's innings contain 101 legal balls, from genuine umpire miscounts (one six-ball set each). This crashed the first women's run during feature selection, before any model or outcome result. Men's features contained no infinite values, so men's results are unaffected. The women's run was restarted from scratch. |
| 2026-11-23 | **Men v women difference intervals** combine the two independent bootstrap intervals with a normal approximation, instead of the cruder min/max bound first coded. | This implements the plan's "independent bootstraps" wording properly. Changed before the comparison was run. It doesn't affect any within-gender result. |
| 2026-11-23 | **Exploratory analyses added after the frozen run** (`src/robustness.py`, `src/compare_models.py`): hot-hand v settledness ablation; batter-level streak test with shuffle; ridge forecast check; leakage test on the women's build; a comparison of 7 models and 5 ensembles. | A post-run review found that the RQ1 expectation includes recent form and may absorb momentum. These analyses don't replace primary results. Women's data is not confirmatory for them. The batter-level finding must be confirmed on fresh data. |
| 2026-11-23 | **Exploratory: what moves momentum, 1st innings v chase** (`src/momentum_drivers.py`): 10-ball momentum split by ball outcome (wicket, dot, 1–3, four, six) as covariance shares; SHAP in runs grouped into cricket factors and every feature ranked, by innings. Also the RQ3 scorecard wording shortened to "No effect found" (the detail stays in the RQ3 section). | Asked for after all results were seen. Descriptive only, not a tested hypothesis, and labelled exploratory in the report. |
| 2026-11-23 | **Exploratory: richer player-history features** (`src/player_styles.py`, `src/history_pool.py`, `src/history_features.py`): batter v this bowler (head-to-head), batter at this venue, batter v pace/spin and v bowling type, bowler v left/right-handed batters, and career T20 records. History drawn from 21 Cricsheet T20 competitions (12,946 matches), strictly earlier dates only. Bowling type and batting hand from Wikipedia infoboxes, matched through the Cricsheet register and Wikidata; missing Hundred players listed in `data/ref/player_styles_manual.csv`. | Suggested after all results were seen. Tested against the locked model on the same forward folds; it does not change any primary result. |
