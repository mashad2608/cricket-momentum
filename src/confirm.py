"""Fresh-data confirmation of the batter-level hot hand (CONFIRMATION_PLAN.md). Run once:  python src/confirm.py
Men: T20 Blast. Women: Charlotte Edwards Cup + Women's T20 Blast. Model recipe copied from the Hundred, no tuning here.
"""
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import log_loss
ROOT = os.path.expanduser('~/hundred-momentum'); sys.path.insert(0, ROOT + '/src')
from clean import clean
from features import build, T20
from streak import streak_test, faced
K = ['W', '0', '1', '2', '3', '4', '5p']; TEST = tuple(int(x) for x in os.environ.get('CONFIRM_TEST', '2023,2024,2025,2026').split(','))
OUT = os.environ.get('CONFIRM_OUT', 'confirm'); N = T20['balls']
COMP = {'male': ['ntb_male'], 'female': ['cec_female', 'wtb_female']}


def full(P, lab):
    Q = np.full((len(P), 7), 1e-9); Q[:, lab] = P; return Q / Q.sum(1, keepdims=True)


def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (len(ps) - r) * ps[i])); adj[i] = run
    return adj


def run(g, log):
    o = f'{ROOT}/data/out/{OUT}/{g}'; os.makedirs(o, exist_ok=True); t0 = time.time()
    parts = [clean(f'{ROOT}/data/raw/{c}', log, n_balls=N, n_franchises=None) for c in COMP[g]]
    B = pd.concat([p[0] for p in parts], ignore_index=True); M = pd.concat([p[1] for p in parts], ignore_index=True)
    assert not M.match_id.duplicated().any()
    D, _ = build(B[[c for c in B if c != 'legal']], T20); D['cls'] = np.where(D.is_wicket == 1, 0, 1 + np.minimum(D.runs_total, 5))
    D = D[D.full_match == 1].reset_index(drop=True)
    log(f'{g}: full-match deliveries {len(D)}, matches by season', D.groupby('season').match_id.nunique().to_dict())
    art = joblib.load(f'{ROOT}/data/out/{g}/final_model.pkl'); F = art['feats']; P_LGB = art['lgb_params']; leaf, mf = art['rf']
    X = D[F].astype(float); y = D.cls.values; gid = D.match_id.values; S = D.season.values
    # ---- classifier: each test season from models trained on all earlier seasons
    rows, lls = [], {}
    for s in TEST:
        tr, te = np.where(S < s)[0], np.where(S == s)[0]
        if len(te) == 0: continue
        a, b = next(GroupShuffleSplit(1, test_size=.2, random_state=7).split(tr, groups=gid[tr])); a, b = tr[a], tr[b]
        m1 = lgb.LGBMClassifier(**P_LGB).fit(X.iloc[a], y[a], eval_set=[(X.iloc[b], y[b])], callbacks=[lgb.early_stopping(80, verbose=False)])
        med = X.iloc[tr].median(); m2 = RandomForestClassifier(n_estimators=500, min_samples_leaf=leaf, max_features=mf, n_jobs=-1, random_state=1).fit(X.iloc[tr].fillna(med), y[tr])
        Q = (full(m1.predict_proba(X.iloc[te]), m1.classes_) + full(m2.predict_proba(X.iloc[te].fillna(med)), m2.classes_)) / 2
        pr = np.bincount(y[tr], minlength=7) / len(tr)
        lls[s] = dict(model=log_loss(y[te], Q, labels=range(7)), baseline=log_loss(y[te], np.tile(pr, (len(te), 1)), labels=range(7)), n=len(te))
        rows.append(D.iloc[te].assign(**{f'p_{k}': Q[:, i] for i, k in enumerate(K)})); log(f'  {g} {s}: {lls[s]} [{time.time() - t0:.0f}s]')
    R = pd.concat(rows, ignore_index=True)
    TR = D[D.season < TEST[0]]
    cm = np.array([TR.runs_total[TR.cls == c].mean() for c in range(7)]); legal_k = np.array([TR.legal[TR.cls == c].mean() for c in range(7)])
    # ---- situation models (main-plan settings), trained before the first test season
    F1 = TR[TR.innings == 1].copy(); F1['rest'] = F1.match_id.map(F1.groupby('match_id').runs_total.sum()) - F1.score_pre; F1['wih'] = 10 - F1.wkts_pre
    RR = lgb.LGBMRegressor(n_estimators=400, learning_rate=.03, num_leaves=8, min_child_samples=200, monotone_constraints=[1, 1], random_state=0, verbose=-1).fit(F1[['balls_left', 'wih']], F1.rest)
    def chase(d):
        d = d[d.innings == 2].merge(M[['match_id', 'winner', 'result']], on='match_id'); d = d[d.result == 'win']
        return d.assign(won=(d.batting_team == d.winner).astype(int), wih=10 - d.wkts_pre)
    C = chase(TR); WX = ['runs_needed', 'balls_left', 'wih']
    WP = lgb.LGBMClassifier(n_estimators=500, learning_rate=.03, num_leaves=8, min_child_samples=150, monotone_constraints=[-1, 1, 1], random_state=0, verbose=-1).fit(C[WX], C.won)
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
    d['value_expected'] = ex; d['value_gap'] = d.value_actual - ex; d['gap_std'] = d.value_gap / d.groupby('innings').value_gap.transform('std')
    d['exp_runs'] = P @ cm
    d.to_parquet(f'{o}/values.parquet', index=False); log(f'  {g}: values for {len(d)} test deliveries [{time.time() - t0:.0f}s]')
    # ---- tests
    out = dict(classifier=lls, primary=streak_test(d, 5, 'full'), raw=streak_test(d, 5, False), settling_only=streak_test(d, 5, 'settling'), ten_ball=streak_test(d, 10, 'full'))
    out['runs_effect'] = runs_effect(d); log(f'  {g} results: ' + json.dumps(out, default=float)); return out


def runs_effect(d, k=5):
    f = faced(d, 'full'); P_, A_, E_ = [], [], []
    for _, x in f.groupby(['match_id', 'innings', 'batter_id'], sort=False):
        if len(x) < 2 * k: continue
        g_, rb, er = x.g.values, x.runs_total.values, x.exp_runs.values
        for s in range(k, len(g_) - k + 1, k): P_.append(g_[s - k:s].sum()); A_.append(rb[s:s + k].sum()); E_.append(er[s:s + k].sum())
    P_, A_, E_ = map(np.array, (P_, A_, E_)); lo, hi = np.quantile(P_, [.25, .75]); over = A_ - E_
    return dict(top_quarter_next5_runs_over_expected=float(over[P_ >= hi].mean()), bottom_quarter=float(over[P_ <= lo].mean()), difference=float(over[P_ >= hi].mean() - over[P_ <= lo].mean()))


if __name__ == '__main__':
    os.makedirs(f'{ROOT}/data/out/{OUT}', exist_ok=True); fh = open(f'{ROOT}/data/out/{OUT}/log.txt', 'a')
    def log(*a): s = ' '.join(str(x) for x in a); print(s, flush=True); fh.write(s + '\n'); fh.flush()
    res = {g: run(g, log) for g in ('male', 'female')}
    adj = holm(np.array([res[g]['primary']['shuffle_p'] for g in ('male', 'female')]))
    for g, a in zip(('male', 'female'), adj):
        pr = res[g]['primary']; pr['holm_p'] = float(a); pr['H5'] = 'confirmed' if (pr['excess_slope'] > 0 and a < .05) else ('not confirmed: opposite sign' if pr['excess_slope'] < 0 else 'not confirmed: null')
        log(f'{g.upper()} H5: {pr["H5"]} (excess slope {pr["excess_slope"]:.4f}, p {pr["shuffle_p"]:.4f}, Holm {a:.4f})')
    json.dump(res, open(f'{ROOT}/data/out/{OUT}/results.json', 'w'), indent=1, default=float)
