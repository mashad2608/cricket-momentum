"""EXPLORATORY sensitivity check: do the main findings change with a better ball model?

Version 2 model = locked features + player history (21 T20 competitions) + feature combinations, with weather and the
weather-based combinations left out (they did not help in src/feature_lab.py). Same recipe (LightGBM tuned settings +
random forest), same forward folds, same situation models and value formula as the main run. Then the thesis, RQ1, RQ2,
the batter hot hand, and the deep-analysis events are recomputed and set beside the original (version 1) results.
The frozen version 1 results stay the primary ones.

Usage:  python src/v2_sensitivity.py
Output: data/out/v2/<gender>/ (values.parquet, results.json, log.txt), data/out/compare/v2_vs_v1.csv
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import log_loss
ROOT = os.path.expanduser('~/hundred-momentum'); sys.path.insert(0, ROOT + '/src')
import pipeline as PL
import feature_lab as FL
from streak import streak_test
import deep_analysis as DA
K = ['W', '0', '1', '2', '3', '4', '5p']
WEATHER_COMBOS = ['combo_humid_x_pace', 'combo_cloud_x_pace', 'combo_dew_x_spin', 'combo_dampair_x_spin', 'combo_pitch_x_batter']
V2 = f'{ROOT}/data/out/v2'


def predictions(g, log):
    D, fam, art = FL.assemble(g)
    FE = fam['locked'] + fam['history'] + [c for c in fam['combinations'] if c not in WEATHER_COMBOS]
    P_LGB, (leaf, mf), cm, legal_k = art['lgb_params'], art['rf'], art['cm'], art['legal_k']
    V = PL.dev_frame(D); y = V.cls.values; gid = V.match_id.values; X = V[FE].astype(float)
    def fit_pair(tr):
        a, b = next(GroupShuffleSplit(1, test_size=.2, random_state=7).split(tr, groups=gid[tr])); a, b = tr[a], tr[b]
        m1 = lgb.LGBMClassifier(**P_LGB).fit(X.iloc[a], y[a], eval_set=[(X.iloc[b], y[b])], callbacks=[lgb.early_stopping(80, verbose=False)])
        med = X.iloc[tr].median(); m2 = RandomForestClassifier(n_estimators=500, min_samples_leaf=leaf, max_features=mf, n_jobs=-1, random_state=1).fit(X.iloc[tr].fillna(med), y[tr])
        return m1, m2, med
    def pred(m, Z):
        m1, m2, med = m; Zx = Z[FE].astype(float)
        return (PL.full(m1.predict_proba(Zx), m1.classes_) + PL.full(m2.predict_proba(Zx.fillna(med)), m2.classes_)) / 2
    rows = []
    for tr, va, s in PL.forward_folds(V.season.values):
        Q = pred(fit_pair(tr), V.iloc[va]); rows.append(V.iloc[va].assign(**{f'p_{k}': Q[:, i] for i, k in enumerate(K)})); log(f'  {g} fold {s} log-loss {log_loss(y[va], Q, labels=range(7)):.4f}')
    T = D[(D.season == 2026) & (D.full_match == 1)].reset_index(drop=True); Q = pred(fit_pair(np.arange(len(V))), T)
    log(f'  {g} 2026 log-loss {log_loss(T.cls, Q, labels=range(7)):.4f}')
    rows.append(T.assign(**{f'p_{k}': Q[:, i] for i, k in enumerate(K)})); R = pd.concat(rows, ignore_index=True); R['exp_runs'] = R[[f'p_{k}' for k in K]].values @ cm
    return D, R, cm, legal_k, len(FE)


def summarise(g, d, res, lab):
    ht = pd.DataFrame(res['HT']); rq1 = pd.DataFrame(res['RQ1']); r10 = rq1[(rq1.window == 10) & (rq1['sample'] == 'pooled')].iloc[0]
    st = streak_test(d, 5, 'full', n_shuffle=500)
    dd = DA.prep(d); ev = DA.events(dd, lab).set_index(['event', 'innings']); K_ = DA.bowler_phases(dd, lab, n_perm=1)[2].set_index(['kind', 'part'])
    gap = lambda part: float(K_.loc[('spin', part)].runs_saved_per_ball - K_.loc[('pace', part)].runs_saved_per_ball)
    return {'Thesis: R² one ball (1st inns)': ht.query('innings==1 and window==1').R2_descriptive.iloc[0],
            'Thesis: R² whole innings (1st inns)': ht.query('innings==1 and window==100').R2_descriptive.iloc[0],
            'RQ1: team momentum slope (10 balls)': r10.slope, 'RQ1: shuffle p (streaks real if < 0.05)': r10.shuffle_p,
            'RQ2: scoreboard disagrees, chase': res['RQ2']['H2a']['disagree_inn2'], 'RQ2: scoreboard disagrees, 1st inns': res['RQ2']['H2a']['disagree_inn1'],
            'RQ2: model beats scoreboard at forecasting (rho gap)': res['RQ2']['H2b']['diff'],
            'H5: batter hot hand, excess slope': st['excess_slope'], 'H5: hot hand shuffle p': st['shuffle_p'],
            'Deep: free hit, next ball (1st inns, SD)': float(ev.loc[('After a no-ball: the free hit (next ball)', '1st innings')].momentum_after),
            'Deep: after a six, next ball (chase, SD)': float(ev.loc[('After a six (next ball)', 'Chase')].momentum_after),
            'Deep: spin minus pace, balls 1–20 (runs saved per ball)': gap(0), 'Deep: spin minus pace, balls 41–60': gap(2)}


def main():
    os.makedirs(V2, exist_ok=True); fh = open(f'{V2}/log.txt', 'w')
    def log(*a): s = ' '.join(str(x) for x in a); print(s, flush=True); fh.write(s + '\n'); fh.flush()
    PL.paths = lambda g: (f'{V2}/{g}', f'{ROOT}/figures/v2/{g}')
    PL.rq3 = lambda g, D, d, log, hot_cut: ({'skipped': 'not rerun in the sensitivity check'}, hot_cut)
    PL.shap_and_ablation = lambda g, D, log: {'skipped': True}
    rows = []
    for g, lab in DA.G.items():
        os.makedirs(f'{V2}/{g}', exist_ok=True); os.makedirs(f'{ROOT}/figures/v2/{g}', exist_ok=True)
        D, R, cm, legal_k, nf = predictions(g, log); log(f'{g}: version 2 uses {nf} features')
        M = pd.read_parquet(f'{ROOT}/data/out/{g}/matches.parquet')
        d2 = PL.stage_values(g, D, M, R, cm, legal_k, log); d2.attrs['g'] = g
        import shutil; shutil.copy(f'{ROOT}/data/out/{g}/final_model.pkl', f'{V2}/{g}/final_model.pkl')   # used only by the secondary forecast-R² check, not in the comparison
        hot = json.load(open(f'{ROOT}/data/out/{g}/results.json'))['hot_cut']
        res2 = PL.stage_analysis(g, D, d2, log, hot_cut=hot); res2 = {k: (v.to_dict('records') if isinstance(v, pd.DataFrame) else v) for k, v in res2.items()}
        d1 = pd.read_parquet(f'{ROOT}/data/out/{g}/values.parquet'); res1 = json.load(open(f'{ROOT}/data/out/{g}/results.json'))
        s1, s2 = summarise(g, d1, res1, lab), summarise(g, d2, res2, lab)
        for k in s1: rows.append(dict(gender=lab, result=k, version1=s1[k], version2=s2[k]))
    T = pd.DataFrame(rows); T['change'] = T.version2 - T.version1; T.round(4).to_csv(f'{ROOT}/data/out/compare/v2_vs_v1.csv', index=False)
    log('\n' + T.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
