"""Exploratory analyses added AFTER the frozen plan was run (see ANALYSIS_PLAN deviations log).
They do not replace any primary result. For these, women's data is NOT a confirmation set.

E1  Hot hand v settledness: does recent PERFORMANCE predict the next ball beyond simply being SET?
E2  Batter-level streaks: the classic hot-hand test on one batter's own consecutive balls.
E3  Forecast sanity check: a simple regularised model in place of the boosted regressor.
E4  Leakage test re-run on the women's feature build.
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb, joblib
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import log_loss
ROOT = os.path.expanduser('~/hundred-momentum'); sys.path.insert(0, ROOT + '/src')
from pipeline import paths, dev_frame, forward_folds, full, boot_matches, slope
from features import build

SETTLED = {'bat_balls', 'part_balls', 'bowl_balls', 'bowler_set_no', 'consecutive_set', 'balls_since_bowler_last', 'set_batter', 'new_batter', 'since_wicket', 'bowl_wkts', 'ball_in_set'}
HOT = {'last5_runs', 'last5_bnd', 'last5_dot', 'last10_runs', 'last10_bnd', 'last10_dot', 'last20_runs', 'last20_bnd', 'last20_dot', 'since_boundary', 'prev_runs', 'bat_sr10', 'bat_dots10',
       'form_vs_hist', 'bowl_dots10', 'bowl_econ', 'strike_rotation', 'bat_runs', 'bat_sr', 'part_runs'}


def e1(g):
    o, _ = paths(g); D = pd.read_parquet(f'{o}/features.parquet'); art = joblib.load(f'{o}/final_model.pkl'); F = art['feats']; P = art['lgb_params']
    V = dev_frame(D); y = V.cls.values; gid = V.match_id.values; FOLDS = forward_folds(V.season.values)
    base = [f for f in F if f not in HOT]; withhot = F
    out = {}; per = {}
    for nm, fs in [('state + history + settledness', base), ('+ recent performance (hot hand)', withhot)]:
        lls, pb = [], []
        for tr, va, s in FOLDS:
            a, b = next(GroupShuffleSplit(1, test_size=.2, random_state=7).split(tr, groups=gid[tr])); a, b = tr[a], tr[b]
            m = lgb.LGBMClassifier(**P).fit(V.iloc[a][fs].astype(float), y[a], eval_set=[(V.iloc[b][fs].astype(float), y[b])], callbacks=[lgb.early_stopping(80, verbose=False)])
            Q = full(m.predict_proba(V.iloc[va][fs].astype(float)), m.classes_); lls.append(log_loss(y[va], Q, labels=range(7))); pb.append(-np.log(Q[np.arange(len(va)), y[va]]))
        out[nm] = dict(n_features=len(fs), cv_logloss=float(np.mean(lls))); per[nm] = np.concatenate(pb)
    va_all = np.concatenate([va for _, va, _ in FOLDS]); diff = per['+ recent performance (hot hand)'] - per['state + history + settledness']
    bs = [diff[i].mean() for i in boot_matches(gid[va_all], 1000, 41)]
    out['gain_from_hot_hand'] = dict(mean_logloss_change=float(diff.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), hot_features_used=sorted(set(F) & HOT))
    return out


def e2(g, k=5, n_shuffle=1000):
    o, _ = paths(g); d = pd.read_parquet(f'{o}/values.parquet')
    f = d[d.wides == 0].sort_values(['date', 'match_id', 'innings', 'seq'])                   # balls the batter actually faced
    seqs = [(mid, x.gap_std.values) for (mid, inn, b), x in f.groupby(['match_id', 'innings', 'batter_id'], sort=False) if len(x) >= 2 * k]
    def win(vs):
        P_, F_, M_ = [], [], []
        for mid, v in vs:
            for s in range(k, len(v) - k + 1, k): P_.append(v[s - k:s].sum()); F_.append(v[s:s + k].sum()); M_.append(mid)
        return np.array(P_), np.array(F_), np.array(M_)
    p, fu, mids = win(seqs); obs = slope(p, fu)
    bs = [slope(p[i], fu[i]) for i in boot_matches(mids, 2000, 42)]
    rng = np.random.default_rng(43); null = [slope(*win([(m, rng.permutation(v)) for m, v in seqs])[:2]) for _ in range(n_shuffle)]
    return dict(window=k, batter_innings=len(seqs), windows=len(p), slope=float(obs), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                shuffle_null_mean=float(np.mean(null)), shuffle_p=float((np.sum(np.array(null) >= obs) + 1) / (n_shuffle + 1)))


def e3(g):
    o, _ = paths(g); D = pd.read_parquet(f'{o}/features.parquet'); A = D[(D.full_match == 1) & (D.innings == 1)].sort_values(['date', 'match_id', 'seq'])
    X0 = ['ball_no', 'wkts_pre', 'score_pre', 'powerplay', 'bat_hist_runs', 'bowl_hist_runs', 'venue_hist_runs', 'season_hist_runs']
    res = []
    for k in (10, 20, 50):
        rows = []
        for mid, x in A.groupby('match_id', sort=False):
            r = x.runs_total.values
            for s in range(0, len(r) - k + 1, k): rows.append((x.index[s], r[s:s + k].sum()))
        W = A.loc[[i for i, _ in rows], X0 + ['season']].copy(); W['y'] = [v for _, v in rows]; W[X0] = W[X0].fillna(W[X0].median())
        pred = pd.Series(np.nan, index=W.index)
        for s in (2022, 2023, 2024, 2025, 2026):
            tr, te = W.season < s, W.season == s
            pred[te] = RidgeCV(alphas=np.logspace(-2, 4, 20)).fit(W.loc[tr, X0], W.y[tr]).predict(W.loc[te, X0])
        ok = pred.notna(); yv = W.y[ok]; base = np.array([W.y[W.season < s].mean() for s in W.season[ok]])
        res.append(dict(window=k, n=int(ok.sum()), R2_forecast_ridge=float(1 - ((yv - pred[ok]) ** 2).sum() / ((yv - base) ** 2).sum())))
    return res


def e4(g, cuts=6):
    o, _ = paths(g); D = pd.read_parquet(f'{o}/features.parquet'); feats = json.load(open(f'{o}/feature_list.json'))
    raw = D[[c for c in D.columns if c not in feats and c not in ('cls', 'legal', 'is_wicket')]].sort_values(['date', 'match_id', 'innings', 'seq']).reset_index(drop=True)
    base, _ = build(raw); base = base.reset_index(drop=True); rng = np.random.default_rng(0); OUT = ['runs_bat', 'runs_extra', 'runs_total', 'wides', 'noballs', 'byes', 'legbyes', 'n_wickets', 'bowler_wicket']
    bad = set(); picks = rng.choice(np.where(raw.date > raw.date.iloc[0])[0], cuts, replace=False)
    for p in picks:
        r = raw.iloc[p]; same = raw.match_id == r.match_id
        later = ((raw.date >= r.date) & ~same) | (same & ((raw.innings > r.innings) | ((raw.innings == r.innings) & (raw.seq >= r.seq))))
        s2 = raw.copy(); ix = np.where(later)[0]; s2.loc[ix, OUT] = raw.loc[rng.permutation(ix), OUT].values
        D2, _ = build(s2); D2 = D2.reset_index(drop=True)
        c = np.where((raw.date < r.date) | (same & ((raw.innings < r.innings) | ((raw.innings == r.innings) & (raw.seq <= r.seq)))))[0]
        bad |= {f for f in feats if not np.allclose(base[f].values[c].astype(float), D2[f].values[c].astype(float), equal_nan=True)}
    return dict(cut_points=cuts, features_changed=sorted(bad))


if __name__ == '__main__':
    R = {}
    for g in ('male', 'female'):
        R[g] = dict(E1=e1(g), E2_5=e2(g, 5), E2_10=e2(g, 10), E3=e3(g), E4=e4(g)); print(g, json.dumps(R[g], indent=1, default=float), flush=True)
    os.makedirs(f'{ROOT}/data/out/compare', exist_ok=True); json.dump(R, open(f'{ROOT}/data/out/compare/robustness.json', 'w'), indent=1, default=float)
