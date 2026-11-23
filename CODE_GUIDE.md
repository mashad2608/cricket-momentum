# Code guide

How the project's code works, in the order it runs. To rerun everything:

```bash
python src/pipeline.py male       # men: about 10 minutes
python src/pipeline.py female     # women: same code, same rules
python src/pipeline.py compare    # men v women, and the transfer check
```

## Folder map

| Path | What it is |
|---|---|
| `data/raw/hnd_male`, `data/raw/hnd_female` | Cricsheet match files (one JSON per match) |
| `src/clean.py` | Step 1: turns the JSON into one row per delivery and checks it |
| `src/features.py` | Step 2: builds the 80 pre-ball features |
| `src/pipeline.py` | Steps 3–6: selection, model, ball values, momentum and all analyses |
| `notebooks/01`–`05b` | The earlier step-by-step development notebooks (men only), kept for the record |
| `data/out/<gender>/` | Every output: models, predictions, results, logs |
| `figures/` | Charts |
| `ANALYSIS_PLAN.md` | The frozen plan, plus the deviations log |

---

## Step 1: `clean.py`

**Purpose:** a clean, trustworthy delivery table.

1. **Flatten.** Each delivery becomes a row with batter, bowler, runs, extras and any wicket. Super overs are dropped. *Retired hurt* is not counted as a wicket.
2. **Checks.** Every check stops the code if it fails:
   - no missing player IDs and no duplicate deliveries
   - runs add up (bat runs + extras = total)
   - our recomputed totals match the official result for every normal game (men 190 of 190, women 179 of 179)
3. **Franchises.** Teams renamed over time (for example Oval Invincibles → MI London) are linked by their home ground.
4. **Full matches.** A match is flagged, not deleted, when it is rain-affected, D/L, no result, or shortened. Analyses use full matches only.

## Step 2: `features.py`

**Purpose:** describe the match *before* each ball, without ever peeking at that ball or anything later.

**The golden rule.** Every feature for a ball uses only:
- **earlier balls in the same innings** (written as `cumsum − current ball`, or `shift(1)`), or
- **matches on earlier dates** (player and venue history).

**Feature families**

| Family | Examples | Cricket meaning |
|---|---|---|
| Match state | balls left, wickets down, required rate, powerplay | The situation |
| Recent play | runs, wickets, dots and boundaries in the last 5, 10 and 20 balls; balls since a boundary or wicket | Short-term flow |
| Partnership | runs and balls since the last wicket | Stability at the crease |
| Batter this innings | runs, balls, strike rate, last-10 strike rate | Is the batter set? |
| Bowler this innings | balls bowled, economy, wickets | How the spell is going |
| History, *shrunk* | batter and bowler runs, boundary, dot and wicket rates from earlier matches | Player quality. Shrinking pulls small samples towards the league average, so a batter with 10 career balls isn't treated as a star |
| Cricket features (plan 4a) | set batter, new batter, bowler's set number, consecutive set, pressure index, strike rotation, death-phase record | Coaches' language |

**Leakage test** (in `notebooks/02_features.ipynb`):
1. Scramble the results of a ball and of everything after it.
2. Rebuild all features.
3. Check that no feature at or before that ball changed.

It is run at 12 random cut points. A deliberately leaky control feature is caught, which proves the test can fail.

## Step 3: `stage_select` (random-forest selection)

**Purpose:** keep features that genuinely help, decided with past seasons only.

1. **Forward folds** (`forward_folds`): train on 2021 → check on 2022; train on 2021–22 → check 2023; and so on to 2025. **2026 is never used here.**
2. **Permutation importance.** In each fold, a random forest is fitted, then each feature is shuffled in turn. If shuffling makes predictions worse, the feature matters. This is done for a runs model and an outcome model.
3. **Keep a feature** if it helps in at least 3 of the 4 folds.

## Step 4: `stage_model` (the classifier)

**Purpose:** for each ball, the probability of each outcome: wicket, 0, 1, 2, 3, 4, 5 or more.

1. **Tuning.** LightGBM is tuned with Optuna (40 trials) on the forward folds. A random forest is tuned on a small grid.
2. **Choice.** Three candidates are compared by average log-loss: LightGBM, random forest, and the average of the two. The best is locked.
3. **Final model.** Trained on all of 2021–2025, then tested **once** on 2026 against two baselines:
   - "always the average"
   - a phase × wickets table
4. **Out-of-sample predictions.** Each season from 2022 to 2025 is predicted by a model trained only on earlier seasons; 2026 by the final model. Momentum is built from these, so the model never "remembers" the balls it scores.
5. **`exp_runs`** = the sum over outcomes of (probability × average runs for that outcome).

## Step 5: `stage_values` (ball values and momentum)

**Purpose:** put every ball on a common scale of "how much did it help the batting side".

1. **Two situation models**, trained on 2021–2025 with *monotone constraints* (more balls or wickets can never mean fewer runs, or a lower chance of winning):
   - **First innings:** expected runs still to come, from balls left and wickets in hand.
   - **Chase:** win probability, from runs needed, balls left and wickets in hand.
2. **Actual value of a ball:**
   - First innings: runs + (expected runs to come after) − (before). A wicket automatically costs what it removes: about 9 runs early, close to 0 off the last balls.
   - Chase: win probability after − before.
3. **Expected value:** the same, averaged over the classifier's outcome probabilities.
4. **Gap** = actual − expected. **Standardised gap** = gap ÷ its spread, so both innings can be pooled.
5. **Momentum** = the sum of the gap over the last 10 deliveries. **Scoreboard momentum** = runs in the last 10, minus 6 per wicket.

## Step 6: `stage_analysis` (the research questions)

| Question | Function or block | What it does |
|---|---|---|
| **Thesis** (trend v single ball) | HT block + `forecast_windows` (see also `robustness.py` E3) | R² of expected v actual runs over windows of 1–100 balls, in two ways: *descriptive* (expectation updates during the window) and *forecast* (predicted from the window's first ball only) |
| **RQ1** (does momentum persist?) | RQ1 block, `windows`, `boot_matches` | Splits each innings into 10- or 5-ball windows and regresses the next window on the last. Interval from resampling whole matches. **Shuffle test:** shuffles the ball order within each innings 1,000 times. If real data looks like the shuffled data, the "persistence" isn't about order |
| **RQ2** (model v scoreboard) | RQ2 block | H2a: how often the two measures disagree, first innings v chase. H2b: which better predicts the next window's over-performance |
| **RQ3** (coaching decisions) | `rq3`, `psm`, `d4_units`, `evalue` | Builds a unit table for each decision (sets, or wickets for D4), matches treated and control situations on their propensity score, checks balance, then estimates the effect with an interval and a sensitivity value. Holm correction across decisions |
| **SHAP** | `shap_and_ablation` | Groups strongly correlated features (Spearman ≥ 0.8), then measures what drives wicket, boundary and dot probabilities. Also partial-dependence plots and the ablation (situation → + history → + recent form) |
| **RQ4** (men v women) | `compare` | Puts each result side by side, with a verdict (confirmed or not), plus the *transfer check*: the men's model used unchanged on women's data |

### How PSM works here (`psm`)

1. **Propensity score:** a logistic regression estimates how likely each situation was to get the decision (for example a fresh bowler), given the covariates.
2. **Matching:** each treated situation is matched to the closest control within a caliper (0.2 standard deviations of the logit). A control can be reused at most 5 times. This fixes the dissertation's problem, where one control was used 6,717 times.
3. **Balance:** every covariate must differ by less than 0.1 standard deviations after matching. Otherwise the result is reported as **"not estimable"** and no effect is claimed.
4. **Effect:** the average difference in the outcome between matched pairs (the ATT).
5. **Interval:** match-level bootstrap, re-matching each time.
6. **E-value:** how strong a hidden factor would have to be to explain the effect away.

---

## Words you'll meet

| Term | Meaning |
|---|---|
| **Log-loss** | How well probabilities match what happened. Lower is better. It punishes confident wrong answers |
| **Brier score** | The average squared error of the probabilities. Lower is better |
| **AUC** | 0.5 = guessing, 1.0 = perfect ranking. Does the model give higher wicket chances to balls that become wickets? |
| **R²** | The share of variation explained, compared with always predicting the average |
| **Calibration** | When the model says 10%, does it happen about 10% of the time? |
| **Bootstrap** | Resampling whole matches many times to see how much a result could vary |
| **Shrinkage** | Pulling small-sample averages towards the league average |
| **Monotone constraint** | Forcing a model to respect common sense (for example, more wickets in hand never lowers the win chance) |

## Extra scripts (exploratory)

| Script | What it does |
|---|---|
| `src/robustness.py` | E1: hot-hand v settledness ablation. E2: batter-level streak test with shuffle. E3: ridge forecast check. E4: leakage test on the women's build. Results: `data/out/compare/robustness.json` |
| `src/compare_models.py` | 7 model families and 5 ensembles, on the same features and forward folds. Results: `data/out/<gender>/model_comparison_*.csv` |
