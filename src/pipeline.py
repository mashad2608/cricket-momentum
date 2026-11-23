"""Frozen pipeline (ANALYSIS_PLAN.md). Identical code for men and women.

Usage:  python src/pipeline.py male   |   python src/pipeline.py female   |   python src/pipeline.py compare
Outputs go to data/out/<gender>/ and figures/<gender>/.
"""
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb, optuna, joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import log_loss, roc_auc_score
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

ROOT = os.path.expanduser('~/hundred-momentum'); sys.path.insert(0, ROOT + '/src')
from clean import clean
from features import build
optuna.logging.set_verbosity(optuna.logging.WARNING)
K = ['W', '0', '1', '2', '3', '4', '5p']
EXPENSIVE_SET = 10          # plan 3: fixed for both competitions
SEASONS_CV = (2022, 2023, 2024, 2025)


def paths(g):
    o = f'{ROOT}/data/out/{g}'; f = f'{ROOT}/figures/{g}'; os.makedirs(o, exist_ok=True); os.makedirs(f, exist_ok=True); return o, f


def logger(o):
    fh = open(f'{o}/log.txt', 'a')
    def log(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); fh.write(s + '\n'); fh.flush()
    return log


def full(P, lab):
    Q = np.full((len(P), 7), 1e-9); Q[:, lab] = P; return Q / Q.sum(1, keepdims=True)


# ------------------------------------------------------------------ stages 1-2: clean + features
def stage_data(g, log):
    o, _ = paths(g); raw = f'{ROOT}/data/raw/hnd_{g}'
    B, M = clean(raw, log)
    D, feats = build(B[[c for c in B if c != 'legal']])
    D['cls'] = np.where(D.is_wicket == 1, 0, 1 + np.minimum(D.runs_total, 5))
    D.to_parquet(f'{o}/features.parquet', index=False); M.to_parquet(f'{o}/matches.parquet', index=False); json.dump(feats, open(f'{o}/feature_list.json', 'w'))
    log(f'features built: {len(feats)}; deliveries {len(D)}; full-match deliveries {int((D.full_match == 1).sum())}')
    return D, M, feats


def dev_frame(D):
    return D[(D.season <= 2025) & (D.full_match == 1)].reset_index(drop=True)


def forward_folds(S):
    return [(np.where(S < s)[0], np.where(S == s)[0], s) for s in SEASONS_CV]


# ------------------------------------------------------------------ stage 3: random-forest selection (Step 3 rule)
def stage_select(g, D, feats, log):
    o, _ = paths(g); V = dev_frame(D); X = V[feats].astype(float); yr = V.runs_total.values.astype(float); yc = V.cls.values; gid = V.match_id.values
    def perm_select(tr, seed=0, reps=3):
        rng = np.random.default_rng(seed); ir = np.zeros(len(feats)); ic = np.zeros(len(feats)); Xn = X.iloc[tr].fillna(X.iloc[tr].median()).values
        for a, b in GroupKFold(3).split(Xn, groups=gid[tr]):
            kw = dict(n_estimators=150, min_samples_leaf=40, max_features=0.33, n_jobs=-1, random_state=seed)
            R = RandomForestRegressor(**kw).fit(Xn[a], yr[tr][a]); C = RandomForestClassifier(**kw).fit(Xn[a], yc[tr][a])
            lab = C.classes_; y1, y2 = yr[tr][b], yc[tr][b]; ok = np.isin(y2, lab)
            r0 = ((R.predict(Xn[b]) - y1) ** 2).mean(); c0 = log_loss(y2[ok], C.predict_proba(Xn[b][ok]), labels=lab)
            for j in range(len(feats)):
                for _ in range(reps):
                    Z = Xn[b].copy(); Z[:, j] = rng.permutation(Z[:, j])
                    ir[j] += (((R.predict(Z) - y1) ** 2).mean() - r0) / (3 * reps) / r0
                    ic[j] += (log_loss(y2[ok], C.predict_proba(Z[ok]), labels=lab) - c0) / (3 * reps) / c0
        return ir, ic
    SEL = []
    for tr, va, s in forward_folds(V.season.values):
        r, c = perm_select(tr); SEL.append([f for f, x, y in zip(feats, r, c) if x > 0 or y > 0]); log(f'  selection fold {s}: kept {len(SEL[-1])} of {len(feats)}')
    cnt = pd.Series([f for s_ in SEL for f in s_]).value_counts(); FEATS = sorted(cnt[cnt >= 3].index)
    json.dump(FEATS, open(f'{o}/final_features.json', 'w')); log(f'features kept in >=3 folds: {len(FEATS)}')
    return FEATS


# ------------------------------------------------------------------ stage 4: tune, choose, lock, test 2026, out-of-sample predictions
def stage_model(g, D, FEATS, log):
    o, _ = paths(g); V = dev_frame(D); X = V[FEATS].astype(float); y = V.cls.values; gid = V.match_id.values; FOLDS = forward_folds(V.season.values)
    def inner(tr): a, b = next(GroupShuffleSplit(1, test_size=.2, random_state=7).split(tr, groups=gid[tr])); return tr[a], tr[b]
    def fit_lgb(p, tr, XX=X, yy=y):
        a, b = inner(tr); return lgb.LGBMClassifier(**p).fit(XX.iloc[a], yy[a], eval_set=[(XX.iloc[b], yy[b])], callbacks=[lgb.early_stopping(80, verbose=False)])
    def lgb_params(t): return dict(n_estimators=3000, learning_rate=t.suggest_float('learning_rate', .005, .08, log=True), num_leaves=t.suggest_int('num_leaves', 4, 48), max_depth=t.suggest_int('max_depth', 2, 8),
        min_child_samples=t.suggest_int('min_child_samples', 40, 600, log=True), subsample=t.suggest_float('subsample', .5, 1), subsample_freq=1, colsample_bytree=t.suggest_float('colsample_bytree', .2, 1),
        reg_lambda=t.suggest_float('reg_lambda', 1e-3, 50, log=True), min_split_gain=t.suggest_float('min_split_gain', 0, .2), random_state=42, n_jobs=-1, verbose=-1)
    def cv_ll(p): return np.mean([log_loss(y[va], full(m.predict_proba(X.iloc[va]), m.classes_), labels=range(7)) for tr, va, s in FOLDS for m in [fit_lgb(p, tr)]])
    st = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42)); st.optimize(lambda t: cv_ll(lgb_params(t)), n_trials=40)
    P_LGB = lgb_params(optuna.trial.FixedTrial(st.best_params)); log(f'LightGBM tuned CV log-loss {st.best_value:.5f}')
    def rf_fold(leaf, mf, tr, va):
        med = X.iloc[tr].median(); m = RandomForestClassifier(n_estimators=500, min_samples_leaf=leaf, max_features=mf, n_jobs=-1, random_state=1).fit(X.iloc[tr].fillna(med), y[tr])
        return full(m.predict_proba(X.iloc[va].fillna(med)), m.classes_)
    grid = {(l, f): [rf_fold(l, f, tr, va) for tr, va, s in FOLDS] for l in (25, 60, 120) for f in (.2, .4)}
    ll = lambda Ps: np.mean([log_loss(y[va], P, labels=range(7)) for P, (tr, va, s) in zip(Ps, FOLDS)]); best_rf = min(grid, key=lambda k: ll(grid[k]))
    PL = [full(m.predict_proba(X.iloc[va]), m.classes_) for tr, va, s in FOLDS for m in [fit_lgb(P_LGB, tr)]]
    cand = {'LightGBM (tuned)': PL, 'Random forest (tuned)': grid[best_rf], 'Average of both': [(a + b) / 2 for a, b in zip(PL, grid[best_rf])]}
    res = {k: ll(v) for k, v in cand.items()}; WIN = min(res, key=res.get); log('candidates CV log-loss:', {k: round(v, 5) for k, v in res.items()}, '-> chosen:', WIN)
    use_l, use_r = WIN != 'Random forest (tuned)', WIN != 'LightGBM (tuned)'
    def fit_pair(tr):
        m = {}
        if use_l: m['lgb'] = fit_lgb(P_LGB, tr)
        if use_r: med = X.iloc[tr].median(); m['rf'] = RandomForestClassifier(n_estimators=500, min_samples_leaf=best_rf[0], max_features=best_rf[1], n_jobs=-1, random_state=1).fit(X.iloc[tr].fillna(med), y[tr]); m['med'] = med
        return m
    def pred(m, Z):
        Ps = []
        if 'lgb' in m: Ps.append(full(m['lgb'].predict_proba(Z[FEATS].astype(float)), m['lgb'].classes_))
        if 'rf' in m: Ps.append(full(m['rf'].predict_proba(Z[FEATS].astype(float).fillna(m['med'])), m['rf'].classes_))
        return sum(Ps) / len(Ps)
    final = fit_pair(np.arange(len(V)))
    if 'lgb' not in final: final['lgb_shap'] = fit_lgb(P_LGB, np.arange(len(V)))       # LightGBM copy kept only for SHAP
    cm = np.array([V.runs_total[V.cls == c].mean() for c in range(7)]); legal_k = np.array([V.legal[V.cls == c].mean() for c in range(7)])
    joblib.dump(dict(model=final, feats=FEATS, cm=cm, legal_k=legal_k, choice=WIN, lgb_params=P_LGB, rf=best_rf), f'{o}/final_model.pkl')
    # 2026 test, once
    T = D[(D.season == 2026) & (D.full_match == 1)].reset_index(drop=True); P = pred(final, T); yt = T.cls.values; O = np.eye(7)[yt]
    pr = np.bincount(y, minlength=7) / len(y); key = lambda d: (np.minimum(d.ball_no - 1, 99) // 25 * 10 + np.minimum(d.wkts_pre, 6)).values
    tb = pd.DataFrame(np.eye(7)[y]).groupby(key(V)).sum().add(50 * pr, axis=1); tb = tb.div(tb.sum(1), axis=0); PB = np.vstack([tb.loc[k].values if k in tb.index else pr for k in key(T)])
    rt = T.runs_total.values
    row = lambda nm, Q: dict(model=nm, logloss=log_loss(yt, Q, labels=range(7)), brier=((Q - O) ** 2).sum(1).mean(), R2_runs=1 - ((rt - Q @ cm) ** 2).sum() / ((rt - V.runs_total.mean()) ** 2).sum(),
        AUC_wicket=roc_auc_score(yt == 0, Q[:, 0]), AUC_boundary=roc_auc_score(T.runs_bat.values >= 4, Q[:, 5:].sum(1)))
    TEST = pd.DataFrame([row('mean baseline', np.tile(pr, (len(T), 1))), row('phase x wickets', PB), row(WIN, P)]); TEST.to_csv(f'{o}/test_2026.csv', index=False); log('2026 test:\n' + TEST.round(4).to_string(index=False))
    # out-of-sample probabilities: 2022-25 from forward-fold models, 2026 from final
    rows = []
    for tr, va, s in FOLDS:
        Q = pred(fit_pair(tr), V.iloc[va]); rows.append(V.iloc[va].assign(**{f'p_{k}': Q[:, i] for i, k in enumerate(K)}))
    rows.append(T.assign(**{f'p_{k}': P[:, i] for i, k in enumerate(K)})); R = pd.concat(rows, ignore_index=True)
    R['exp_runs'] = R[[f'p_{k}' for k in K]].values @ cm
    R.to_parquet(f'{o}/predictions.parquet', index=False); json.dump(dict(choice=WIN, cv=res, best_rf=best_rf, lgb=P_LGB), open(f'{o}/model_choice.json', 'w'), indent=1, default=str)
    return R, cm, legal_k


# ------------------------------------------------------------------ stage 5: situation models, ball values, momentum
def stage_values(g, D, M, R, cm, legal_k, log):
    o, _ = paths(g); V = dev_frame(D)
    F1 = V[V.innings == 1].copy(); F1['rest'] = F1.match_id.map(F1.groupby('match_id').runs_total.sum()) - F1.score_pre; F1['wih'] = 10 - F1.wkts_pre
    RR = lgb.LGBMRegressor(n_estimators=400, learning_rate=.03, num_leaves=8, min_child_samples=200, monotone_constraints=[1, 1], random_state=0, verbose=-1).fit(F1[['balls_left', 'wih']], F1.rest)
    def chase(d):
        d = d[d.innings == 2].merge(M[['match_id', 'winner', 'result']], on='match_id'); d = d[d.result == 'win']
        return d.assign(won=(d.batting_team == d.winner).astype(int), wih=10 - d.wkts_pre)
    C = chase(V); WX = ['runs_needed', 'balls_left', 'wih']
    WP = lgb.LGBMClassifier(n_estimators=500, learning_rate=.03, num_leaves=8, min_child_samples=150, monotone_constraints=[-1, 1, 1], random_state=0, verbose=-1).fit(C[WX], C.won)
    T26 = chase(D[(D.season == 2026) & (D.full_match == 1)])
    p = WP.predict_proba(T26[WX])[:, 1]; log(f'win-prob model on 2026 chases: log-loss {log_loss(T26.won, p):.4f} (constant {log_loss(T26.won, np.full(len(T26), C.won.mean())):.4f}), AUC {roc_auc_score(T26.won, p):.3f}')
    joblib.dump(dict(RR=RR, WP=WP), f'{o}/situation_models.pkl')
    Er = lambda b, w: np.where((np.asarray(w) <= 0) | (np.asarray(b) <= 0), 0.0, RR.predict(pd.DataFrame({'balls_left': np.asarray(b, float), 'wih': np.asarray(w, float)})))
    def wp(need, balls, wih):
        need, balls, wih = (np.asarray(a, float) for a in (need, balls, wih))
        q = WP.predict_proba(pd.DataFrame({'runs_needed': need, 'balls_left': np.maximum(balls, 1), 'wih': np.maximum(wih, 1)}))[:, 1]
        return np.where(need <= 0, 1.0, np.where((balls <= 0) | (wih <= 0), 0.0, q))
    d = R.sort_values(['date', 'match_id', 'innings', 'seq']).reset_index(drop=True)
    w0 = 10 - d.wkts_pre.values; b0 = d.balls_left.values; P = d[[f'p_{k}' for k in K]].values; r = d.runs_total.values; inn1 = d.innings.values == 1; need0 = d.runs_needed.values
    b1 = b0 - d.legal.values; w1 = w0 - d.is_wicket.values
    d['value_actual'] = np.where(inn1, r + Er(b1, w1) - Er(b0, w0), wp(need0 - r, b1, w1) - wp(need0, b0, w0))
    ex = np.zeros(len(d))
    for k in range(7):
        dw = 1 if k == 0 else 0
        for lg_, pl in [(1, legal_k[k]), (0, 1 - legal_k[k])]:
            ex += P[:, k] * pl * np.where(inn1, cm[k] + Er(b0 - lg_, w0 - dw) - Er(b0, w0), wp(need0 - cm[k], b0 - lg_, w0 - dw) - wp(need0, b0, w0))
    d['value_expected'] = ex; d['value_gap'] = d.value_actual - ex
    d['gap_std'] = d.value_gap / d.groupby('innings').value_gap.transform('std')
    d['wp_pre'] = np.where(inn1, np.nan, wp(need0, b0, w0))
    g_ = d.groupby(['match_id', 'innings'], sort=False)
    d['momentum10'] = g_.value_gap.transform(lambda x: x.rolling(10, min_periods=1).sum())
    d['mom10_std'] = g_.gap_std.transform(lambda x: x.rolling(10, min_periods=1).sum())
    d['scoreboard10'] = g_.runs_total.transform(lambda x: x.rolling(10, min_periods=1).sum()) - 6 * g_.is_wicket.transform(lambda x: x.rolling(10, min_periods=1).sum())
    d['set_idx'] = (d.ball_no - 1) // 5
    d.to_parquet(f'{o}/values.parquet', index=False)
    log(f'ball values: {len(d)} deliveries, seasons {sorted(d.season.unique())}; mean gap by innings', d.groupby('innings').value_gap.mean().round(4).to_dict())
    return d


# ------------------------------------------------------------------ forecast version of the thesis test (deviation 1)
def forecast_windows(D, d, inn, k):
    """Runs in the next k deliveries, predicted ONLY from the state at the window's first ball.
    Regressor trained on the same forward seasons as the classifier: 2022-25 windows from models fitted on earlier seasons, 2026 from 2021-25."""
    art = joblib.load(f"{paths(d.attrs['g'])[0]}/final_model.pkl"); FEATS = art['feats']
    A = D[(D.full_match == 1) & (D.innings == inn)].sort_values(['date', 'match_id', 'innings', 'seq'])
    rows = []
    for mid, x in A.groupby('match_id', sort=False):
        r = x.runs_total.values
        for s in range(0, len(r) - k + 1, k): rows.append((x.index[s], r[s:s + k].sum()))
    if not rows: return dict(R2=np.nan, n=0)
    W = A.loc[[i for i, _ in rows], FEATS + ['season', 'match_id']].astype({f: float for f in FEATS}); W['y'] = [v for _, v in rows]
    pred = pd.Series(np.nan, index=W.index); prm = dict(n_estimators=300, learning_rate=.03, num_leaves=8, min_child_samples=max(20, len(W) // 200), subsample=.8, subsample_freq=1, colsample_bytree=.6, random_state=0, verbose=-1)
    for s in (2022, 2023, 2024, 2025, 2026):
        tr, te = W.season < s, W.season == s
        if tr.sum() < 30 or te.sum() == 0: continue
        pred[te] = lgb.LGBMRegressor(**prm).fit(W.loc[tr, FEATS], W.y[tr]).predict(W.loc[te, FEATS])
    ok = pred.notna(); y = W.y[ok]; base = np.array([W.y[W.season < s].mean() for s in W.season[ok]])
    return dict(R2=float(1 - ((y - pred[ok]) ** 2).sum() / ((y - base) ** 2).sum()), n=int(ok.sum()))


# ------------------------------------------------------------------ helpers for analysis
def boot_matches(ids, n, seed):
    rng = np.random.default_rng(seed); um = np.unique(ids); pos = {q: np.where(ids == q)[0] for q in um}
    for _ in range(n): yield np.concatenate([pos[q] for q in rng.choice(um, len(um))])


def windows(d, k, col):
    rows = []
    for (mid, inn), x in d.groupby(['match_id', 'innings'], sort=False):
        v = x[col].values
        for s in range(k, len(v) - k + 1, k): rows.append((mid, inn, v[s - k:s].sum(), v[s:s + k].sum()))
    return pd.DataFrame(rows, columns=['match_id', 'innings', 'past', 'future'])


def slope(p, f): return np.polyfit(p, f, 1)[0]


# ------------------------------------------------------------------ stage 6: analyses (thesis, RQ1, RQ2, RQ3, SHAP)
def stage_analysis(g, D, d, log, hot_cut=None):
    o, fdir = paths(g); out = {}
    # ---- HT: explanatory power by window size, descriptive and forecast
    ht = []
    for inn in (1, 2):
        z = d[d.innings == inn]
        for k in (1, 5, 10, 20, 50, 100):
            A, E1, idx = [], [], []
            for _, x in z.groupby('match_id', sort=False):
                r = x.runs_total.values; e = x.exp_runs.values
                for s in range(0, len(r) - k + 1, k): A.append(r[s:s + k].sum()); E1.append(e[s:s + k].sum()); idx.append(x.index[s])
            A, E1 = np.array(A), np.array(E1); r2 = lambda E: 1 - ((A - E) ** 2).sum() / ((A - A.mean()) ** 2).sum()
            E2 = forecast_windows(D, d, inn, k)
            ht.append(dict(innings=inn, window=k, n=len(A), R2_descriptive=r2(E1), R2_forecast=E2['R2'], n_forecast=E2['n']))
    out['HT'] = pd.DataFrame(ht); log('THESIS (R2 of expected v actual runs by window):\n' + out['HT'].round(3).to_string(index=False))
    # ---- RQ1: persistence of standardised gap
    rq1 = []
    for k in (10, 5):
        Wk = windows(d, k, 'gap_std')
        for lab, z in [('pooled', Wk), ('innings 1', Wk[Wk.innings == 1]), ('innings 2', Wk[Wk.innings == 2])]:
            ids = z.match_id.values; b = [slope(z.past.values[i], z.future.values[i]) for i in boot_matches(ids, 2000, 11)]
            row = dict(window=k, sample=lab, n=len(z), slope=slope(z.past, z.future), lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5), spearman=spearmanr(z.past, z.future).correlation)
            if lab == 'pooled':
                rng = np.random.default_rng(21); obs = row['slope']; inns = [x.gap_std.values for _, x in d.groupby(['match_id', 'innings'], sort=False)]; null = []
                for _ in range(1000):
                    P_, F_ = [], []
                    for v in inns:
                        v = rng.permutation(v)
                        for s in range(k, len(v) - k + 1, k): P_.append(v[s - k:s].sum()); F_.append(v[s:s + k].sum())
                    null.append(slope(np.array(P_), np.array(F_)))
                row['shuffle_p'] = (np.sum(np.array(null) >= obs) + 1) / 1001; row['shuffle_null_mean'] = float(np.mean(null))
                row['H1_supported'] = bool(row['lo'] > 0 and row['shuffle_p'] < .05)
            rq1.append(row)
    out['RQ1'] = pd.DataFrame(rq1); log('RQ1 (persistence):\n' + out['RQ1'].round(4).to_string(index=False))
    # ---- RQ2
    dd = d.copy(); gg = dd.groupby('innings')
    dd['m_z'] = (dd.momentum10 - gg.momentum10.transform('mean')) / gg.momentum10.transform('std'); dd['s_z'] = (dd.scoreboard10 - gg.scoreboard10.transform('mean')) / gg.scoreboard10.transform('std')
    dd = dd[dd.ball_no >= 11]
    def h2a(z): return np.mean(np.sign(z[z.innings == 2].m_z) != np.sign(z[z.innings == 2].s_z)) - np.mean(np.sign(z[z.innings == 1].m_z) != np.sign(z[z.innings == 1].s_z))
    b = [h2a(dd.iloc[i]) for i in boot_matches(dd.match_id.values, 1000, 12)]
    r2a = dict(disagree_inn1=np.mean(np.sign(dd[dd.innings == 1].m_z) != np.sign(dd[dd.innings == 1].s_z)), disagree_inn2=np.mean(np.sign(dd[dd.innings == 2].m_z) != np.sign(dd[dd.innings == 2].s_z)), diff=h2a(dd), lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5))
    W = windows(d, 10, 'gap_std').rename(columns={'past': 'gap_past', 'future': 'gap_future'})
    sb = d.assign(sbv=np.where(d.is_wicket == 1, -6, d.runs_total)); sb['sbv'] = sb.sbv / sb.groupby('innings').sbv.transform('std')
    W['sb_past'] = windows(sb, 10, 'sbv').past.values
    def h2b(z): return spearmanr(z.gap_past, z.gap_future).correlation - spearmanr(z.sb_past, z.gap_future).correlation
    b = [h2b(W.iloc[i]) for i in boot_matches(W.match_id.values, 1000, 13)]
    r2b = dict(rho_model=spearmanr(W.gap_past, W.gap_future).correlation, rho_scoreboard=spearmanr(W.sb_past, W.gap_future).correlation, diff=h2b(W), lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5))
    out['RQ2'] = dict(H2a=r2a, H2b=r2b); log('RQ2:', json.dumps(out['RQ2'], default=lambda v: round(float(v), 4)))
    # ---- RQ3
    out['RQ3'], hot_used = rq3(g, D, d, log, hot_cut)
    # ---- SHAP + ablation
    out['SHAP'] = shap_and_ablation(g, D, log)
    json.dump({k: (v.to_dict('records') if isinstance(v, pd.DataFrame) else v) for k, v in out.items()} | {'hot_cut': hot_used}, open(f'{o}/results.json', 'w'), indent=1, default=lambda v: float(v) if np.isscalar(v) else str(v))
    return out


# ------------------------------------------------------------------ RQ3: propensity score matching for four decisions
def set_table(d):
    agg = d.groupby(['match_id', 'innings', 'set_idx'], sort=False).agg(bowler=('bowler_id', 'first'), runs=('runs_total', 'sum'), legal=('legal', 'sum'), gap=('gap_std', 'sum'),
        value=('value_actual', 'sum'), wkts=('is_wicket', 'sum'), mom_end=('mom10_std', 'last'), date=('date', 'first'), season=('season', 'first'), first_row=('seq', 'idxmin')).reset_index()
    return agg


STATE = ['innings2', 'ball_no', 'balls_left', 'wkts_pre', 'score_pre', 'runs_needed', 'rate_gap', 'bat_hist_runs', 'bat_hist_bnd', 'bat_hist_dot', 'bat_hist_wkt', 'venue_hist_runs', 'season_hist_runs']


def psm(U, cov, seed=0, min_n=30):
    t = U.treat.values == 1
    if t.sum() < min_n or (~t).sum() < min_n: return None
    X = U[cov].astype(float); X = X.fillna(X.median()); lo, hi = X.quantile(.01), X.quantile(.99); X = X.clip(lo, hi, axis=1)
    Z = ((X - X.mean()) / X.std().replace(0, 1)).values
    p = LogisticRegression(C=1.0, max_iter=3000).fit(Z, t).predict_proba(Z)[:, 1]; lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    cal = 0.2 * lg.std(); ti, ci = np.where(t)[0], np.where(~t)[0]
    nn = NearestNeighbors(n_neighbors=min(60, len(ci))).fit(lg[ci, None]); dist, ind = nn.kneighbors(lg[ti, None])
    uses = np.zeros(len(ci), int); pairs = []; rng = np.random.default_rng(seed)
    for i in rng.permutation(len(ti)):
        for dd_, j in zip(dist[i], ind[i]):
            if dd_ > cal: break
            if uses[j] < 5: uses[j] += 1; pairs.append((ti[i], ci[j])); break
    if len(pairs) < min_n: return None
    T_, C_ = map(np.array, zip(*pairs)); y = U.y.values
    sd = np.sqrt((X.values[t].var(0) + X.values[~t].var(0)) / 2); sd[sd == 0] = 1
    smd = (X.values[T_].mean(0) - X.values[C_].mean(0)) / sd; w = uses[uses > 0]
    res = dict(att=float((y[T_] - y[C_]).mean()), n_treated=int(t.sum()), n_control=int((~t).sum()), matched=len(pairs), share_matched=len(pairs) / t.sum(),
               max_abs_smd=float(np.abs(smd).max()), balanced=bool(np.abs(smd).max() < .1), ess_controls=float(w.sum() ** 2 / (w ** 2).sum()), max_control_use=int(uses.max()))
    for c in [c for c in U.columns if c.startswith('y2_')]: res['att_' + c[3:]] = float((U[c].values[T_] - U[c].values[C_]).mean())
    res['_smd'] = dict(zip(cov, np.round(smd, 3)))
    return res


def evalue(est, sd):
    if not np.isfinite(est) or est == 0: return 1.0
    rr = np.exp(0.91 * abs(est) / sd); return float(rr + np.sqrt(rr * (rr - 1)))


def rq3(g, D, d, log, hot_cut):
    o, _ = paths(g); S = set_table(d); S = S.sort_values(['match_id', 'innings', 'set_idx']).reset_index(drop=True)
    nxt = S.groupby(['match_id', 'innings']).shift(-1); prv = S.groupby(['match_id', 'innings']).shift(1)
    first = d.loc[S.first_row.values].reset_index(drop=True)                      # first delivery of each set
    first_next = d.set_index(d.index).reindex(nxt.first_row.values).reset_index(drop=True)
    base_cov = lambda F: F[STATE].assign(mom_start=None)
    def bowl_cov(idx_rows):  # outgoing bowler's figures at their last ball of the set
        return pd.DataFrame({'out_bowl_hist_runs': idx_rows.bowl_hist_runs.values, 'out_bowl_hist_wkt': idx_rows.bowl_hist_wkt.values, 'out_bowl_balls': idx_rows.bowl_balls.values})
    last_rows = d.groupby(['match_id', 'innings', 'set_idx'], sort=False).tail(1).set_index(['match_id', 'innings', 'set_idx']).reindex(pd.MultiIndex.from_frame(S[['match_id', 'innings', 'set_idx']])).reset_index()
    results = {}
    def make(mask, treat, cov_frame, y, extra):
        U = cov_frame[mask].copy(); U['treat'] = treat[mask].astype(int).values; U['y'] = y[mask].values; U['match_id'] = S.match_id[mask].values
        for k_, v_ in extra.items(): U['y2_' + k_] = v_[mask].values
        return U.dropna(subset=['y']).reset_index(drop=True)
    # covariates measured at the start of the NEXT set (D1, D3)
    cov_next = pd.concat([first_next[STATE].reset_index(drop=True), pd.DataFrame({'mom_start': S.mom_end.values, 'set_runs': S.runs.values}), bowl_cov(last_rows)], axis=1)
    sec = lambda: {'value_inn1': nxt.value.where(S.innings == 1), 'value_inn2': nxt.value.where(S.innings == 2), 'wkts': nxt.wkts}
    # D1 keep or change after an expensive set
    m1 = (S.legal == 5) & (S.runs >= EXPENSIVE_SET) & nxt.bowler.notna()
    U1 = make(m1, (nxt.bowler == S.bowler), cov_next, nxt.gap, sec())
    # D2 ten straight or split: a bowler's 2nd set
    bsn = first.bowler_set_no.values
    cov_d2 = pd.concat([first[STATE].reset_index(drop=True), pd.DataFrame({'mom_start': prv.mom_end.values, 'bowl_hist_runs': first.bowl_hist_runs.values, 'bowl_hist_wkt': first.bowl_hist_wkt.values,
                        'first_set_runs': first.bowl_econ.values * first.bowl_balls.values, 'first_set_wkts': first.bowl_wkts.values})], axis=1)
    m2 = pd.Series(bsn == 2) & S.legal.ge(1)
    U2 = make(m2, pd.Series(first.consecutive_set.values == 1), cov_d2, S.gap, {'value_inn1': S.value.where(S.innings == 1), 'value_inn2': S.value.where(S.innings == 2), 'wkts': S.wkts})
    # D3 fresh bowler after a hot spell
    if hot_cut is None: hot_cut = float(S[(S.season <= 2025) & (S.legal == 5)].mom_end.quantile(.75))
    m3 = (S.legal == 5) & (S.mom_end >= hot_cut) & nxt.bowler.notna() & prv.bowler.notna()
    fresh = (nxt.bowler != S.bowler) & (nxt.bowler != prv.bowler)
    U3 = make(m3, fresh, cov_next, nxt.gap, sec())
    # D4 promoted big hitter after an early wicket
    U4 = d4_units(d)
    for nm, U, cov in [('D1 keep v change bowler', U1, list(cov_next.columns)), ('D2 ten straight v split', U2, list(cov_d2.columns)),
                       ('D3 fresh bowler after hot spell', U3, list(cov_next.columns)), ('D4 promoted big hitter', U4, [c for c in U4.columns if c not in ('treat', 'y', 'match_id') and not c.startswith('y2_')] if len(U4) else [])]:
        cov = [c for c in cov if U[c].notna().any()] if len(U) else cov
        r = psm(U, cov) if len(U) else None
        if r is None: results[nm] = dict(status='not estimable (too few units or matches)', n_units=len(U), n_treated=int(U.treat.sum()) if len(U) else 0); log(f'{nm}: not estimable', results[nm]); continue
        bs = [x['att'] for i in boot_matches(U.match_id.values, 500, 31) for x in [psm(U.iloc[i].reset_index(drop=True), cov, seed=1)] if x is not None]
        lo_, hi_ = np.percentile(bs, [2.5, 97.5]); p = max(2 * min(np.mean(np.array(bs) <= 0), np.mean(np.array(bs) >= 0)), 1 / len(bs))
        sdy = U.y.std(); bound = 0 if lo_ <= 0 <= hi_ else (lo_ if r['att'] > 0 else hi_)
        r.update(lo=lo_, hi=hi_, p=p, n_boot=len(bs), evalue=evalue(r['att'], sdy), evalue_ci=evalue(bound, sdy) if bound != 0 else 1.0,
                 status='estimable' if r['balanced'] else 'not estimable (balance failed)')
        results[nm] = r; log(f"{nm}: treated {r['n_treated']}, matched {r['matched']}, ATT {r['att']:+.3f} [{lo_:+.3f}, {hi_:+.3f}], p {p:.3f}, max|SMD| {r['max_abs_smd']:.3f}, status {r['status']}")
    est = [k for k, v in results.items() if v.get('status') == 'estimable']; ps = sorted((results[k]['p'], k) for k in est)
    for i, (p, k) in enumerate(ps): results[k]['p_holm'] = min(1.0, max(p * (len(ps) - j) for j, (p, _) in enumerate(ps[:i + 1])))
    pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk != '_smd'} for k, v in results.items()}).T.to_csv(f'{o}/rq3_psm.csv')
    return results, hot_cut


def d4_units(d):
    rows = []; X = d.sort_values(['date', 'match_id', 'innings', 'seq'])
    # batting position = order of first appearance (striker or non-striker) in the innings
    pos_rows = []
    for (mid, inn), x in X.groupby(['match_id', 'innings'], sort=False):
        seen = {}
        for s_, a, b in zip(x.seq.values, x.batter_id.values, x.non_striker_id.values):
            for p in (a, b):
                if p not in seen: seen[p] = (len(seen) + 1, s_)
        for p, (ps, s_) in seen.items(): pos_rows.append((mid, inn, x.date.iloc[0], p, ps, s_))
    Pz = pd.DataFrame(pos_rows, columns=['match_id', 'innings', 'date', 'player', 'pos', 'first_seq']).sort_values('date')
    gp = Pz.groupby('player', sort=False)
    # usual position from matches on strictly earlier dates
    daily = Pz.groupby(['player', 'date']).pos.apply(list).reset_index().sort_values('date')
    hist = {}; usual = []
    for pl, dt, ps in daily.itertuples(index=False):
        h = hist.get(pl, []); usual.append((pl, dt, np.median(h) if len(h) >= 3 else np.nan)); hist[pl] = h + ps
    Pz = Pz.merge(pd.DataFrame(usual, columns=['player', 'date', 'usual']), on=['player', 'date'], how='left')
    first_ball = X.drop_duplicates(['match_id', 'innings', 'batter_id']).set_index(['match_id', 'innings', 'batter_id'])
    for (mid, inn), x in X.groupby(['match_id', 'innings'], sort=False):
        x = x.reset_index(drop=True); wk = x.index[(x.is_wicket == 1) & (x.ball_no <= 40) & (x.wkts_pre + 1 <= 3)]
        pz = Pz[(Pz.match_id == mid) & (Pz.innings == inn)]
        for i in wk:
            s_ = x.seq[i]; inc = pz[pz.first_seq > s_].sort_values('first_seq').head(1)
            if inc.empty or i + 1 >= len(x): continue
            inc = inc.iloc[0]
            if not np.isfinite(inc.usual): continue
            nb = x.iloc[i + 1]
            try: hsr = first_ball.loc[(mid, inn, inc.player)].bat_hist_runs
            except KeyError: hsr = np.nan
            fut = x.gap_std.values[i + 1:i + 21]
            rows.append(dict(match_id=mid, date=x.date[i], slot=inc.pos, usual=inc.usual, hsr=hsr, y=fut.sum() if len(fut) == 20 else np.nan,
                             innings2=int(inn == 2), ball_no=nb.ball_no, balls_left=nb.balls_left, wkts_pre=nb.wkts_pre, score_pre=nb.score_pre, runs_needed=nb.runs_needed, rate_gap=nb.rate_gap,
                             mom_start=x.mom10_std[i], venue_hist_runs=nb.venue_hist_runs, season_hist_runs=nb.season_hist_runs))
    U = pd.DataFrame(rows)
    if U.empty: return U
    U = U.sort_values('date'); U['hsr_median_before'] = U.hsr.expanding().median().shift(1).values  # league median of incoming batters, earlier dates
    U['promoted'] = (U.slot <= U.usual - 2) & (U.hsr > U.hsr_median_before); U['near_usual'] = (U.slot - U.usual).abs() <= 1
    U = U[U.promoted | (U.near_usual & ~U.promoted)].copy(); U['treat'] = U.promoted.astype(int)
    return U.drop(columns=['date', 'usual', 'hsr', 'hsr_median_before', 'promoted', 'near_usual']).dropna(subset=['y']).reset_index(drop=True)


# ------------------------------------------------------------------ SHAP, grouping by Spearman, partial dependence, ablation
def shap_and_ablation(g, D, log):
    import shap, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    o, fdir = paths(g); art = joblib.load(f'{o}/final_model.pkl'); FEATS = art['feats']; m = art['model'].get('lgb') or art['model'].get('lgb_shap'); cm = art['cm']
    V = dev_frame(D); Xs = V[FEATS].astype(float).sample(4000, random_state=0)
    rho = Xs.corr(method='spearman').abs().fillna(0); Z = linkage(squareform(1 - rho.values, checks=False), 'complete'); lab = fcluster(Z, t=0.2, criterion='distance')
    groups = {}
    for f, l in zip(FEATS, lab): groups.setdefault(l, []).append(f)
    gname = {l: (fs[0] if len(fs) == 1 else f'{fs[0]} (+{len(fs) - 1})') for l, fs in groups.items()}
    sv = shap.TreeExplainer(m).shap_values(Xs)
    sv = np.stack(sv, -1) if isinstance(sv, list) else sv          # (n, features, classes)
    w_exp = cm - (cm * Xs.shape[0] / Xs.shape[0]).mean()           # weight per class for an expected-runs summary in log-odds space (descriptive)
    summ = {}
    for nm, cls in [('wicket', [0]), ('boundary', [5, 6]), ('dot', [1])]:
        s = sv[:, :, cls].sum(-1); summ[nm] = pd.Series({gname[l]: np.abs(s[:, [FEATS.index(f) for f in fs]].sum(1)).mean() for l, fs in groups.items()}).sort_values(ascending=False)
    T = pd.DataFrame(summ).fillna(0).sort_values('boundary', ascending=False); T.to_csv(f'{o}/shap_groups.csv')
    log('SHAP (mean |log-odds contribution| by correlated group):\n' + T.head(12).round(4).to_string())
    log('groups with >1 feature: ' + '; '.join(f'{gname[l]}: {fs}' for l, fs in groups.items() if len(fs) > 1))
    # partial dependence: expected runs and wicket probability as the top features vary
    top = [groups[l][0] for l in sorted(groups, key=lambda l: -T.loc[gname[l]].sum())[:4]]
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6)); pdp = {}
    for a, f in zip(ax, top):
        grid = np.unique(np.nanquantile(Xs[f], np.linspace(.02, .98, 15))); er, pw = [], []
        for v in grid:
            Zt = Xs.copy(); Zt[f] = v; P = full(m.predict_proba(Zt), m.classes_); er.append((P @ cm).mean()); pw.append(P[:, 0].mean())
        a.plot(grid, er, color='#1f77b4'); a.set_title(f); a.set_ylabel('expected runs'); a2 = a.twinx(); a2.plot(grid, pw, color='#d62728', ls='--'); a2.set_ylabel('P(wicket)', color='#d62728')
        pdp[f] = dict(grid=grid.tolist(), exp_runs=er, p_wicket=pw)
    fig.tight_layout(); fig.savefig(f'{fdir}/partial_dependence.png', dpi=140); plt.close(fig)
    # ablation: match state -> + player history -> + recent form, forward folds, tuned LightGBM settings
    state = {'innings2', 'ball_no', 'balls_left', 'wkts_pre', 'score_pre', 'crr', 'ball_in_set', 'powerplay', 'runs_needed', 'req_rate', 'rate_gap', 'chase_target', 'wkts_per_ball_left', 'free_hit', 'pressure_index'}
    hist = {f for f in FEATS if '_hist_' in f or f.startswith('matchup')}
    layers = [('match state', [f for f in FEATS if f in state])]; layers.append(('+ player history', layers[0][1] + sorted(hist)))
    layers.append(('+ recent form (all)', FEATS))
    X = V; y = V.cls.values; gid = V.match_id.values; FOLDS = forward_folds(V.season.values); P_LGB = art['lgb_params']; ab = []
    for nm, fs in layers:
        lls = []
        for tr, va, s in FOLDS:
            a_, b_ = next(GroupShuffleSplit(1, test_size=.2, random_state=7).split(tr, groups=gid[tr])); a_, b_ = tr[a_], tr[b_]
            mm = lgb.LGBMClassifier(**P_LGB).fit(X.iloc[a_][fs].astype(float), y[a_], eval_set=[(X.iloc[b_][fs].astype(float), y[b_])], callbacks=[lgb.early_stopping(80, verbose=False)])
            lls.append(log_loss(y[va], full(mm.predict_proba(X.iloc[va][fs].astype(float)), mm.classes_), labels=range(7)))
        ab.append(dict(layer=nm, n_features=len(fs), cv_logloss=np.mean(lls)))
    pr = np.bincount(y, minlength=7) / len(y); ab.insert(0, dict(layer='no features (base rates)', n_features=0, cv_logloss=np.mean([log_loss(y[va], np.tile(np.bincount(y[tr], minlength=7) / len(tr), (len(va), 1)), labels=range(7)) for tr, va, s in FOLDS])))
    AB = pd.DataFrame(ab); log('ABLATION:\n' + AB.round(5).to_string(index=False))
    return dict(groups={gname[l]: fs for l, fs in groups.items()}, shap=T.head(15).round(5).to_dict(), ablation=AB.to_dict('records'), pdp_features=top)


# ------------------------------------------------------------------ RQ4 / transfer
def compare(log):
    om, _ = paths('male'); of, _ = paths('female'); oc, fc = paths('compare')
    rm, rf = json.load(open(f'{om}/results.json')), json.load(open(f'{of}/results.json'))
    def dci(a, b, key):   # independent samples: combine the two bootstrap intervals (normal approximation)
        se = np.hypot((a['hi'] - a['lo']) / 3.92, (b['hi'] - b['lo']) / 3.92); dd = b[key] - a[key]; return (dd - 1.96 * se, dd + 1.96 * se)
    rows = []
    for a, b in zip(rm['RQ1'], rf['RQ1']):
        rows.append(dict(test=f"RQ1 slope, {a['window']}-ball, {a['sample']}", men=a['slope'], men_ci=(a['lo'], a['hi']), women=b['slope'], women_ci=(b['lo'], b['hi']),
                         diff=b['slope'] - a['slope'], diff_ci=dci(a, b, 'slope')))
    for k in ('H2a', 'H2b'):
        a, b = rm['RQ2'][k], rf['RQ2'][k]; rows.append(dict(test=f'RQ2 {k}', men=a['diff'], men_ci=(a['lo'], a['hi']), women=b['diff'], women_ci=(b['lo'], b['hi']), diff=b['diff'] - a['diff'], diff_ci=dci(a, b, 'diff')))
    for k in rm['RQ3']:
        a, b = rm['RQ3'][k], rf['RQ3'].get(k, {})
        rows.append(dict(test=f'RQ3 {k}', men=a.get('att'), men_ci=(a.get('lo'), a.get('hi')), men_status=a.get('status'), women=b.get('att'), women_ci=(b.get('lo'), b.get('hi')), women_status=b.get('status'),
                         diff=(b['att'] - a['att']) if 'att' in a and 'att' in b else None, diff_ci=dci(a, b, 'att') if 'lo' in a and 'lo' in b else None))
    C = pd.DataFrame(rows)
    def verdict(r):
        if r.test.startswith('RQ3') and (r.get('men_status') != 'estimable' or r.get('women_status') != 'estimable'): return 'not estimable in one or both'
        m, w, ci = r.men, r.women, r.women_ci
        if m is None or w is None or ci[0] is None: return 'n/a'
        if np.sign(m) == np.sign(w) and (ci[0] > 0 or ci[1] < 0): return 'confirmed'
        return 'not confirmed: opposite sign' if np.sign(m) != np.sign(w) and (ci[0] > 0 or ci[1] < 0) else 'not confirmed: women null'
    C['verdict'] = C.apply(verdict, axis=1); C.to_csv(f'{oc}/rq4_comparison.csv', index=False); log('RQ4:\n' + C.to_string(index=False))
    # transfer: men's final model applied unchanged to women's out-of-sample seasons
    art = joblib.load(f'{om}/final_model.pkl'); W = pd.read_parquet(f'{of}/predictions.parquet'); Fm = art['feats']; m = art['model']
    Ps = []
    if 'lgb' in m: Ps.append(full(m['lgb'].predict_proba(W[Fm].astype(float)), m['lgb'].classes_))
    if 'rf' in m: Ps.append(full(m['rf'].predict_proba(W[Fm].astype(float).fillna(m['med'])), m['rf'].classes_))
    Pm = sum(Ps) / len(Ps); Pw = W[[f'p_{k}' for k in K]].values; y = W.cls.values
    tr = dict(women_own_model_logloss=log_loss(y, Pw, labels=range(7)), men_model_on_women_logloss=log_loss(y, Pm, labels=range(7)),
              wicket_rate_actual=float((y == 0).mean()), wicket_rate_men_model=float(Pm[:, 0].mean()), wicket_rate_women_model=float(Pw[:, 0].mean()),
              boundary_rate_actual=float(W.runs_bat.ge(4).mean()), boundary_rate_men_model=float(Pm[:, 5:].sum(1).mean()), boundary_rate_women_model=float(Pw[:, 5:].sum(1).mean()))
    json.dump(tr, open(f'{oc}/transfer.json', 'w'), indent=1); log('TRANSFER:', json.dumps(tr, indent=1))


def run_analysis(g):
    o, _ = paths(g); log = logger(o); log(f'===== {g} analysis-only rerun {time.ctime()} =====')
    D = pd.read_parquet(f'{o}/features.parquet'); d = pd.read_parquet(f'{o}/values.parquet')
    hot = json.load(open(f'{ROOT}/data/out/male/results.json'))['hot_cut'] if g == 'female' else None
    d.attrs['g'] = g; stage_analysis(g, D, d, log, hot_cut=hot)


def run(g):
    o, _ = paths(g); log = logger(o); t0 = time.time(); log(f'===== {g} run started {time.ctime()} =====')
    D, M, feats = stage_data(g, log); log(f'[{time.time() - t0:.0f}s] data')
    FEATS = stage_select(g, D, feats, log); log(f'[{time.time() - t0:.0f}s] selection')
    R, cm, legal_k = stage_model(g, D, FEATS, log); log(f'[{time.time() - t0:.0f}s] model')
    d = stage_values(g, D, M, R, cm, legal_k, log); d.attrs['g'] = g; log(f'[{time.time() - t0:.0f}s] values')
    hot = json.load(open(f'{ROOT}/data/out/male/results.json'))['hot_cut'] if g == 'female' else None
    stage_analysis(g, D, d, log, hot_cut=hot); log(f'[{time.time() - t0:.0f}s] analysis done')


if __name__ == '__main__':
    a = sys.argv[1]
    if a == 'compare': compare(logger(paths('compare')[0]))
    elif a.endswith('-analysis'): run_analysis(a.split('-')[0])
    else: run(a)
