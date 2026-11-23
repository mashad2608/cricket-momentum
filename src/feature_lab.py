"""EXPLORATORY feature lab (added after the frozen plan): which new information helps predict the next ball, and which
features act together?

Families tested on top of the locked features, one at a time and all together, with the locked recipe
(LightGBM tuned settings + random forest, averaged) on forward season folds 2022-2025:
  history       player history from 21 T20 competitions (src/history_features.py)
  conditions    weather and pitch stand-ins (src/conditions.py)
  combinations  hand-made feature combinations (below)
Then SHAP interaction values on two LightGBM models (boundary or not, wicket or not) with every feature,
to rank the pairs of features whose combined effect is more than the sum of their parts.

Usage:  python src/feature_lab.py
Output: data/out/compare/feature_lab.json, data/out/<gender>/feature_interactions.csv
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb, joblib, shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
ROOT = os.path.expanduser('~/hundred-momentum'); sys.path.insert(0, ROOT + '/src')
from pipeline import full, dev_frame, forward_folds
import history_features as HF
from explain_match import group_of
from player_styles import classify_bowl  # noqa: F401  (documents where bowling type comes from)


def combinations(D):
    """Hand-made combinations, each a cricket idea."""
    C = pd.DataFrame(index=D.index); death = (D.ball_no >= 76).astype(float); spin = D.bowl_spin
    C['combo_bnd_matchup'] = D.ext_bat_bnd * D.ext_bowl_bnd / D.ext_bat_bnd.mean()           # boundary hitter v expensive bowler
    C['combo_out_matchup'] = D.ext_bat_out * D.ext_bowl_out / D.ext_bat_out.mean()           # fragile batter v wicket-taker
    C['combo_spin_x_lefthander'] = spin * D.bat_left                                         # spin to a left-hander
    C['combo_spin_x_death'] = spin * death; C['combo_pace_x_powerplay'] = (1 - spin) * D.powerplay
    C['combo_set_x_death'] = (D.bat_balls >= 10).astype(float) * death                       # set batter at the death
    C['combo_new_batter_x_spin'] = D.new_batter * spin
    C['combo_chase_pressure'] = np.where(D.innings == 2, D.req_rate * (D.wkts_pre + 1) / 10, np.nan)   # rate needed, scaled up by wickets lost
    C['combo_humid_x_pace'] = D.wx_humidity * (1 - spin); C['combo_cloud_x_pace'] = D.wx_cloud * (1 - spin)   # swing conditions
    C['combo_dew_x_spin'] = D.evening_game * spin; C['combo_dampair_x_spin'] = -D.wx_dew_spread * spin      # wet ball for spinners
    C['combo_form_x_quality'] = D.bat_sr10.fillna(D.ext_bat_runs) * D.ext_bat_runs            # in form and good
    C['combo_prev_boundary'] = (D.prev_runs >= 4).astype(float)                                # last ball went for four or six
    C['combo_prev_bnd_same_bowler'] = C.combo_prev_boundary * (D.ball_in_set > 1)             # ... and the same bowler continues
    C['combo_pitch_x_batter'] = D.inn1_rpb_vs_venue * D.ext_bat_runs                           # good pitch today and a good batter
    return C


def assemble(g):
    D = pd.read_parquet(f'{ROOT}/data/out/{g}/features.parquet')
    H, _ = HF.build(D, g); W = pd.read_parquet(f'{ROOT}/data/out/{g}/conditions.parquet')
    D = pd.concat([D, H, W], axis=1); C = combinations(D); D = pd.concat([D, C], axis=1)
    art = joblib.load(f'{ROOT}/data/out/{g}/final_model.pkl')
    fam = dict(locked=art['feats'], history=list(H.columns), conditions=list(W.columns), combinations=list(C.columns))
    return D, fam, art


def run(g, log):
    D, fam, art = assemble(g); P_LGB, (leaf, mf) = art['lgb_params'], art['rf']
    V = dev_frame(D); y = V.cls.values; gid = V.match_id.values; FOLDS = forward_folds(V.season.values)
    def ll(feats):
        X = V[feats].astype(float); per = []
        for tr, te, s in FOLDS:
            a, b = next(GroupShuffleSplit(1, test_size=.2, random_state=7).split(tr, groups=gid[tr])); a, b = tr[a], tr[b]
            m1 = lgb.LGBMClassifier(**P_LGB).fit(X.iloc[a], y[a], eval_set=[(X.iloc[b], y[b])], callbacks=[lgb.early_stopping(80, verbose=False)])
            med = X.iloc[tr].median(); m2 = RandomForestClassifier(n_estimators=500, min_samples_leaf=leaf, max_features=mf, n_jobs=-1, random_state=1).fit(X.iloc[tr].fillna(med), y[tr])
            P = (full(m1.predict_proba(X.iloc[te]), m1.classes_) + full(m2.predict_proba(X.iloc[te].fillna(med)), m2.classes_)) / 2
            per.append(-np.log(P[np.arange(len(te)), y[te]]))
        return np.concatenate(per)
    ids = np.concatenate([gid[te] for _, te, _ in FOLDS])
    def boot(d, n=2000):
        s = pd.Series(d).groupby(ids).agg(['sum', 'size']); rng = np.random.default_rng(42); u = len(s); out = []
        for _ in range(n): k = rng.choice(u, u); out.append(s['sum'].values[k].sum() / s['size'].values[k].sum())
        return float(np.quantile(out, .025)), float(np.quantile(out, .975))
    base = ll(fam['locked']); res = {'locked': dict(logloss=float(base.mean()))}
    configs = {'+ history': fam['history'], '+ conditions': fam['conditions'], '+ combinations': fam['combinations'], '+ history + conditions': fam['history'] + fam['conditions'],
               '+ all three': fam['history'] + fam['conditions'] + fam['combinations']}
    for name, extra in configs.items():
        l = ll(fam['locked'] + extra); d = l - base; lo, hi = boot(d)
        res[name] = dict(logloss=float(l.mean()), diff=float(d.mean()), lo=lo, hi=hi, n_features=len(fam['locked']) + len(extra)); log(f'  {g} {name}: {res[name]}')
    inter = interactions(g, V, fam, log)
    return dict(ablation=res, families={k: v for k, v in fam.items() if k != 'locked'}, top_interactions=inter)


def interactions(g, V, fam, log, n=2500):
    feats = fam['locked'] + fam['history'] + fam['conditions'] + fam['combinations']; feats = list(dict.fromkeys(feats))
    X = V[feats].astype(float); rows = []
    for target, yv in [('boundary', (V.runs_bat >= 4).astype(int).values), ('wicket', (V.is_wicket == 1).astype(int).values)]:
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=.03, num_leaves=15, min_child_samples=100, subsample=.8, subsample_freq=1, colsample_bytree=.7, random_state=0, verbose=-1).fit(X, yv)
        S = X.sample(min(n, len(X)), random_state=1); I = shap.TreeExplainer(m).shap_interaction_values(S)
        I = I[1] if isinstance(I, list) else I; A = np.abs(I).mean(0); np.fill_diagonal(A, 0)
        iu = np.triu_indices(len(feats), 1); vals = A[iu] * 2
        top = np.argsort(vals)[::-1][:20]
        for t in top:
            i, j = iu[0][t], iu[1][t]; a, b = S.iloc[:, i], S.iloc[:, j]
            hi_a, hi_b = a > a.median(), b > b.median(); v = I[:, i, j] * 2
            q = {f'{x}_{y}': float(np.nanmean(v[(hi_a == (x == 'hi')) & (hi_b == (y == 'hi'))])) if ((hi_a == (x == 'hi')) & (hi_b == (y == 'hi'))).any() else np.nan for x in ('hi', 'lo') for y in ('hi', 'lo')}
            rows.append(dict(target=target, a=feats[i], b=feats[j], group_a=group_of(feats[i]), group_b=group_of(feats[j]), strength=float(vals[t]), **{f'when_{k}': q[k] for k in q}))
    R = pd.DataFrame(rows); R.to_csv(f'{ROOT}/data/out/{g}/feature_interactions.csv', index=False)
    log(f'  {g} top interactions:\n' + R.groupby('target').head(8)[['target', 'a', 'b', 'strength']].to_string(index=False))
    return R.groupby('target').head(10).to_dict('records')


if __name__ == '__main__':
    o = f'{ROOT}/data/out/compare'; fh = open(f'{o}/feature_lab_log.txt', 'w')
    def log(*a): s = ' '.join(str(x) for x in a); print(s, flush=True); fh.write(s + '\n'); fh.flush()
    out = {g: run(g, log) for g in ('male', 'female')}
    json.dump(out, open(f'{o}/feature_lab.json', 'w'), indent=1, default=float)
