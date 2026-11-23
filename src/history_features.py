"""EXPLORATORY: do richer player-history features improve the ball model? (Added after the frozen plan.)

New pre-ball features, from every T20 / Hundred match in data/clean/history_pool.parquet on STRICTLY EARLIER dates,
same gender, shrunk towards a sensible prior so thin histories don't mislead:
  ext_bat_*      batter's career T20 record (all competitions)                      prior: average batter, 60 balls
  ext_bowl_*     bowler's career T20 record                                         prior: average bowler, 60 balls
  h2h_*          this batter against this bowler (head-to-head)                     prior: batter + bowler records combined, 30 balls
  batvenue_*     this batter at this venue                                          prior: batter's career record, 60 balls
  batkind_*      this batter against this kind of bowler (pace or spin)             prior: batter's career record, 60 balls
  battype_*      this batter against this bowling type (fast, medium, off spin, leg spin, left-arm orthodox, left-arm wrist)
                                                                                    prior: batter v pace/spin, 60 balls
  bowlhand_*     this bowler against left- or right-handed batters                  prior: bowler's career record, 60 balls
  flags          bowler is spin, bowler is left-arm, batter is left-handed
Bowling type and batting hand come from src/player_styles.py (Wikipedia infoboxes via the Cricsheet register).

Test: the locked recipe (LightGBM with the tuned settings + random forest, averaged), forward season folds 2022-2025,
locked features v locked features + new features. Paired match bootstrap of the log-loss difference. 2026 reported as a second look.

Usage:  python src/history_features.py
Output: data/out/compare/history_features.json, data/out/<gender>/history_feature_importance.csv
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import log_loss
ROOT = os.path.expanduser('~/hundred-momentum'); sys.path.insert(0, ROOT + '/src')
from pipeline import full, dev_frame, forward_folds
from history_pool import venue_key
ST = ['runs', 'bnd', 'dot', 'out']


def asof_hist(H, T, keys, k, prior):
    """Shrunk rates for `keys` from H rows on dates strictly before each T row's date. prior: DataFrame of ST rates aligned to T."""
    A = H.groupby(keys + ['date']).agg(n=('runs', 'size'), **{s: (s, 'sum') for s in ST}).reset_index()
    A = A.sort_values('date'); A[['n'] + ST] = A.groupby(keys)[['n'] + ST].cumsum()
    t = T[keys + ['date']].reset_index().rename(columns={'index': '_i'})
    ok = t[keys].notna().all(1); m = pd.merge_asof(t[ok].sort_values('date'), A, on='date', by=keys, allow_exact_matches=False).set_index('_i').reindex(T.index)
    n = m.n.fillna(0).values; out = pd.DataFrame(index=T.index)
    for s in ST: out[s] = (m[s].fillna(0).values + k * prior[s].values) / (n + k)
    out['n'] = n; return out


def build(D, g):
    H = pd.read_parquet(f'{ROOT}/data/clean/history_pool.parquet'); H = H[H.gender == g].copy(); H['bnd'] = H.four + H.six
    S = pd.read_csv(f'{ROOT}/data/ref/player_styles.csv', dtype=str).set_index('identifier')
    H['kind'] = H.bowler_id.map(S.bowl_kind); H['btype'] = H.bowler_id.map(S.bowl_type); H['hand'] = H.batter_id.map(S.bat_hand)
    T = D[['date', 'batter_id', 'bowler_id', 'venue']].copy(); T['date'] = pd.to_datetime(T.date); T['venue'] = T.venue.map(venue_key)
    T['kind'] = T.bowler_id.map(S.bowl_kind); T['btype'] = T.bowler_id.map(S.bowl_type); T['hand'] = T.batter_id.map(S.bat_hand)
    H['venue'] = H.venue.astype(str)
    base = pd.DataFrame({s: np.full(len(T), H[s].mean()) for s in ST}, index=T.index)
    F = pd.DataFrame(index=D.index)
    bat = asof_hist(H, T, ['batter_id'], 60, base); bowl = asof_hist(H, T, ['bowler_id'], 60, base)
    combo = (bat[ST] + bowl[ST] - base[ST]).clip(lower=1e-3)
    h2h = asof_hist(H, T, ['batter_id', 'bowler_id'], 30, combo)
    ven = asof_hist(H, T, ['batter_id', 'venue'], 60, bat)
    kind = asof_hist(H, T, ['batter_id', 'kind'], 60, bat)
    btyp = asof_hist(H, T, ['batter_id', 'btype'], 60, kind)
    hand = asof_hist(H, T, ['bowler_id', 'hand'], 60, bowl)
    for nm, X, cols in [('ext_bat', bat, ST + ['n']), ('ext_bowl', bowl, ST + ['n']), ('h2h', h2h, ['runs', 'out', 'n']), ('batvenue', ven, ['runs', 'bnd', 'n']),
                        ('batkind', kind, ST + ['n']), ('battype', btyp, ['runs', 'out']), ('bowlhand', hand, ['runs', 'out'])]:
        for c in cols: F[f'{nm}_{c}'] = X[c].values
    # edges: how much better or worse than the player's own record in this match-up
    F['h2h_edge_runs'] = h2h.runs.values - combo.runs.values; F['batvenue_edge_runs'] = ven.runs.values - bat.runs.values
    F['batkind_edge_runs'] = kind.runs.values - bat.runs.values; F['bowlhand_edge_runs'] = hand.runs.values - bowl.runs.values
    F['bowl_spin'] = (T.kind == 'spin').astype(float).where(T.kind.notna()).values
    F['bowl_left_arm'] = T.bowler_id.map(S.bowl_arm).eq('left').astype(float).where(T.bowler_id.map(S.bowl_arm).notna()).values
    F['bat_left'] = (T.hand == 'left').astype(float).where(T.hand.notna()).values
    cover = dict(batter_has_history=float((bat.n > 0).mean()), h2h_has_history=float((h2h.n > 0).mean()), h2h_median_balls=float(np.median(h2h.n[h2h.n > 0])) if (h2h.n > 0).any() else 0,
                 batvenue_has_history=float((ven.n > 0).mean()), bowler_type_known=float(T.kind.notna().mean()), batter_hand_known=float(T.hand.notna().mean()))
    return F, cover


def evaluate(g, log):
    D = pd.read_parquet(f'{ROOT}/data/out/{g}/features.parquet'); art = joblib.load(f'{ROOT}/data/out/{g}/final_model.pkl')
    FE, P_LGB, (leaf, mf) = art['feats'], art['lgb_params'], art['rf']
    NEW, cover = build(D, g); NEWF = list(NEW.columns); D = pd.concat([D, NEW], axis=1)
    log(f'{g}: {len(NEWF)} new features; coverage {json.dumps({k: round(v, 3) for k, v in cover.items()})}')
    V = dev_frame(D); y = V.cls.values; gid = V.match_id.values
    def fit_pred(feats, tr, te, Z=None):
        X = V[feats].astype(float); Zx = (Z if Z is not None else V.iloc[te])[feats].astype(float)
        a, b = next(GroupShuffleSplit(1, test_size=.2, random_state=7).split(tr, groups=gid[tr])); a, b = tr[a], tr[b]
        m1 = lgb.LGBMClassifier(**P_LGB).fit(X.iloc[a], y[a], eval_set=[(X.iloc[b], y[b])], callbacks=[lgb.early_stopping(80, verbose=False)])
        med = X.iloc[tr].median(); m2 = RandomForestClassifier(n_estimators=500, min_samples_leaf=leaf, max_features=mf, n_jobs=-1, random_state=1).fit(X.iloc[tr].fillna(med), y[tr])
        return (full(m1.predict_proba(Zx), m1.classes_) + full(m2.predict_proba(Zx.fillna(med)), m2.classes_)) / 2, m1
    ll_b, ll_n, ids, folds = [], [], [], {}
    for tr, te, s in forward_folds(V.season.values):
        Pb, _ = fit_pred(FE, tr, te); Pn, _ = fit_pred(FE + NEWF, tr, te); yt = y[te]
        lb = -np.log(Pb[np.arange(len(te)), yt]); ln = -np.log(Pn[np.arange(len(te)), yt]); ll_b.append(lb); ll_n.append(ln); ids.append(gid[te])
        folds[s] = dict(locked=float(lb.mean()), with_history=float(ln.mean()), diff=float(ln.mean() - lb.mean())); log(f'  {g} fold {s}: {folds[s]}')
    lb, ln, ids = map(np.concatenate, (ll_b, ll_n, ids)); d = pd.Series(ln - lb).groupby(ids).agg(['sum', 'size'])
    rng = np.random.default_rng(42); u = d.index.values; boot = []
    for _ in range(2000):
        k = rng.choice(len(u), len(u)); boot.append(d['sum'].values[k].sum() / d['size'].values[k].sum())
    cv = dict(locked=float(lb.mean()), with_history=float(ln.mean()), diff=float((ln - lb).mean()), lo=float(np.quantile(boot, .025)), hi=float(np.quantile(boot, .975)),
              p_worse_or_equal=float(np.mean(np.array(boot) >= 0)), folds=folds)
    # 2026: second look (the test season was already used once for the locked model)
    T = D[(D.season == 2026) & (D.full_match == 1)].reset_index(drop=True); allv = np.arange(len(V))
    Pb, _ = fit_pred(FE, allv, None, T); Pn, mN = fit_pred(FE + NEWF, allv, None, T)
    t26 = dict(locked=float(log_loss(T.cls, Pb, labels=range(7))), with_history=float(log_loss(T.cls, Pn, labels=range(7)))); t26['diff'] = t26['with_history'] - t26['locked']
    imp = pd.Series(mN.booster_.feature_importance('gain'), index=FE + NEWF); imp = (imp / imp.sum()).sort_values(ascending=False)
    I = pd.DataFrame({'gain_share': imp, 'rank': np.arange(1, len(imp) + 1), 'new': [f in NEWF for f in imp.index]}); I.index.name = 'feature'
    I.to_csv(f'{ROOT}/data/out/{g}/history_feature_importance.csv')
    res = dict(new_features=NEWF, coverage=cover, cv=cv, test_2026_second_look=t26, new_gain_share=float(I[I.new].gain_share.sum()),
               top_new=I[I.new].head(10).reset_index()[['feature', 'rank', 'gain_share']].to_dict('records'))
    log(f'  {g} CV: {json.dumps({k: v for k, v in cv.items() if k != "folds"})}\n  {g} 2026 (second look): {t26}\n  {g} new features carry {res["new_gain_share"]:.1%} of LightGBM gain; top new: {res["top_new"][:5]}')
    return res


if __name__ == '__main__':
    o = f'{ROOT}/data/out/compare'; fh = open(f'{o}/history_features_log.txt', 'w')
    def log(*a): s = ' '.join(str(x) for x in a); print(s, flush=True); fh.write(s + '\n'); fh.flush()
    out = {g: evaluate(g, log) for g in ('male', 'female')}
    json.dump(out, open(f'{o}/history_features.json', 'w'), indent=1, default=float)
