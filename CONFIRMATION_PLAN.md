# Confirmation plan: batter-level hot hand on fresh data

**Frozen 2026-11-23, before any fresh data was downloaded.** The checksum is in `CONFIRMATION_PLAN.sha256`, and the analysis code checksums are in `data/out/confirm/code_frozen.sha256`.

## Why

The Hundred review found, *after* looking at the data, that a batter's 5-ball spell predicts their next 5 balls (shuffle p = 0.001 for men and for women). Because the hypothesis came from the data, the Hundred can't confirm it. This plan tests it once on competitions the project has never used.

The following development checks were done on Hundred data before freezing, and they shaped the test:
- **Settling** (removing the average gap at each balls-faced position) changed nothing: the men's slope went from 0.146 to 0.144.
- **Situation miscalibration** (removing the average gap by innings × 10-ball stage × wickets) lowered the men's slope to 0.113, but the excess over the shuffle null stayed clear (p = 0.002 for men and women).

So the primary test below uses **both adjustments**.

## Hypothesis (H5)

Within a batter's innings, how far the batter beat or fell short of the model over their last 5 balls faced predicts how far they beat or fall short of it over their next 5. This holds beyond:
- that batter's overall form in that innings
- the average settling-in curve
- the average model error in each match situation.

## Data

| | Competition (Cricsheet code) | Format |
|---|---|---|
| **Men** | T20 Blast (`ntb_male`) | T20 |
| **Women** | English women's domestic T20: Charlotte Edwards Cup (`cec_female`) plus Women's T20 Blast (`wtb_female`), pooled as one competition | T20 |

- Format settings: 120 balls, 6-ball overs, powerplay = first 36 balls, death = from ball 91.
- Full matches only, defined as in the main plan but at 120 balls. Rain-affected, D/L, no-result and shortened games are excluded.
- **Test seasons: 2023–2026.** Each test season is predicted by models trained on all *earlier* seasons, with full matches only. Earlier seasons are used only for training.

## Model recipe (fixed in advance, no tuning on the new data)

- **Features:** the Hundred's selected feature list for the same gender (men 47, women 48), computed with the T20 format settings. In T20 a "set" means an over.
- **Classifier:** the average of LightGBM and a random forest. LightGBM uses the Hundred's tuned settings for that gender, with early stopping on a 20% group hold-out. The random forest has 500 trees, `min_samples_leaf` 60 and `max_features` 0.2, the Hundred's chosen values for both genders.
- **Class means and legal-ball shares:** from training seasons before 2023.
- **Situation models:** the main plan's monotone LightGBM settings, trained on full matches before 2023.
- **Ball values:**
  - Actual value, expected value and gap: as in the main plan.
  - Standardised gap: the gap ÷ its SD within each innings type, using the test seasons.

## Test

**Units**
- Balls a batter actually faced (wides excluded), in order within each batter-innings.
- Batter-innings with at least 10 balls faced.

**Adjustment.** Two steps, applied in this order to the standardised gap:
1. Subtract the mean for its cell: innings × 10-ball stage (`(ball_no − 1) // 10`) × wickets down (capped at 6).
2. Subtract the mean at each balls-faced position (1–30, then 31+).

**Statistic**
- Split each batter-innings into non-overlapping 5-ball windows.
- Compute the OLS slope of the next window's summed adjusted gap on the last window's.

**Null**
- Shuffle the balls within each batter-innings, 1,000 times, with seed 43.
- p = (number of shuffled slopes ≥ observed + 1) ÷ 1,001. The test is one-sided.

**Effect size**
- Excess slope = observed slope − mean shuffled slope.
- 95% interval from a match-level bootstrap, 2,000 resamples, seed 42.

## Decision rule

- **H5 confirmed in a competition:** the excess slope is > 0 and the Holm-adjusted p is < 0.05. The Holm adjustment is across the two competitions.
- **Not confirmed** otherwise. Report whether that was a null or the opposite sign.
- **Both competitions are reported whatever the result.**

## Secondary (reported, not used for the decision)

1. The same test **without adjustment** (the exploratory Hundred version).
2. **Settling-only** adjustment.
3. **10-ball windows**, fully adjusted.
4. **Model sanity check:** classifier log-loss on the test seasons, against the class-frequency baseline from training seasons.
5. **Effect in runs:** the next window's mean actual runs minus expected runs, for batters in the top quarter of last-window adjusted gap, against the bottom quarter.

## Rules

- The code is written and checksummed before download, and run once.
- If the code crashes before any test statistic is computed, a fix is allowed and logged, and the run starts again from scratch.
- No change to windows, adjustments, seasons or thresholds after the data is seen.
- Anything else is labelled exploratory.

## Deviations log

| Date | Change | Reason |
|---|---|---|
| 2026-11-23 | **Test seasons were 2023–2025, not 2023–2026.** Cricsheet's 2026 files leave out the chase target, so every 2026 match was flagged incomplete by the frozen cleaning code. | Found after the run. The primary result stands as run. Any 2026 analysis is a labelled second look, not confirmatory. |
| 2026-11-23 | **Women's T20 Blast contributed 45 of 112 matches**, for the same reason (its 2026 season has no targets). | Same cause. No change to the primary result. |
| 2026-11-23 | **`clean.py` changed after the run.** A missing chase target is now filled as first-innings runs + 1, only in games without a rain rule. | Needed for any 2026 re-run. The Hundred is unaffected (verified: targets and full-match counts are identical). The frozen checksums in `data/out/confirm/code_frozen.sha256` refer to the version before this change. |
