"""Exploratory: compare a wide range of model families on identical data, features and forward folds.
Choice is by mean cross-validated log-loss (2022-2025 folds). 2026 scores are a SECOND look at the test season and are labelled exploratory.
"""
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb, joblib
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import log_loss, roc_auc_score
ROOT = os.path.expanduser('~/hundred-momentum'); sys.path.insert(0, ROOT + '/src')
from pipeline import paths, dev_frame, forward_folds, full, boot_matches


def models(art):
    P = art['lgb_params']; rl, rf_ = art['rf']
    def es_split(tr, gid): a, b = next(GroupShuffleSplit(1, test_size=.2, random_state=7).split(tr, groups=gid[tr])); return tr[a], tr[b]
    def lgbm(X, y, tr, gid):
        a, b = es_split(tr, gid); return lgb.LGBMClassifier(**P).fit(X.iloc[a], y[a], eval_set=[(X.iloc[b], y[b])], callbacks=[lgb.early_stopping(80, verbose=False)])
    def xgbm(X, y, tr, gid):
        a, b = es_split(tr, gid)
        return xgb.XGBClassifier(n_estimators=3000, learning_rate=.02, max_depth=3, min_child_weight=20, subsample=.8, colsample_bytree=.6, reg_lambda=5, objective='multi:softprob',
                                 num_class=7, eval_metric='mlogloss', early_stopping_rounds=80, tree_method='hist', random_state=0, n_jobs=-1, verbosity=0).fit(X.iloc[a], y[a], eval_set=[(X.iloc[b], y[b])], verbose=False)
    def hgb(X, y, tr, gid): return HistGradientBoostingClassifier(learning_rate=.03, max_leaf_nodes=15, min_samples_leaf=80, l2_regularization=1.0, max_iter=2000, early_stopping=True, validation_fraction=.2, n_iter_no_change=60, random_state=0).fit(X.iloc[tr], y[tr])
    def rf(X, y, tr, gid): return make_pipeline(SimpleImputer(strategy='median'), RandomForestClassifier(n_estimators=500, min_samples_leaf=rl, max_features=rf_, n_jobs=-1, random_state=1)).fit(X.iloc[tr], y[tr])
    def et(X, y, tr, gid): return make_pipeline(SimpleImputer(strategy='median'), ExtraTreesClassifier(n_estimators=500, min_samples_leaf=60, max_features=.4, n_jobs=-1, random_state=2)).fit(X.iloc[tr], y[tr])
    def logit(X, y, tr, gid): return make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(), LogisticRegression(C=.05, max_iter=3000)).fit(X.iloc[tr], y[tr])
    def mlp(X, y, tr, gid): return make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-2, learning_rate_init=1e-3, early_stopping=True, validation_fraction=.2, n_iter_no_change=15, max_iter=300, random_state=0)).fit(X.iloc[tr], y[tr])
    return {'Logistic regression': logit, 'Neural network (MLP)': mlp, 'Random forest': rf, 'Extra trees': et, 'HistGradientBoosting': hgb, 'LightGBM': lgbm, 'XGBoost': xgbm}


ENSEMBLES = {'LightGBM + random forest (locked model)': ['LightGBM', 'Random forest'], 'LightGBM + XGBoost': ['LightGBM', 'XGBoost'],
             'LightGBM + XGBoost + random forest': ['LightGBM', 'XGBoost', 'Random forest'], 'All tree models': ['LightGBM', 'XGBoost', 'Random forest', 'Extra trees', 'HistGradientBoosting'],
             'All seven models': ['Logistic regression', 'Neural network (MLP)', 'Random forest', 'Extra trees', 'HistGradientBoosting', 'LightGBM', 'XGBoost']}


def metrics(y, Q, runs, bnd, cm, base_mean):
    O = np.eye(7)[y]
    return dict(logloss=log_loss(y, Q, labels=range(7)), brier=((Q - O) ** 2).sum(1).mean(), R2_runs=1 - ((runs - Q @ cm) ** 2).sum() / ((runs - base_mean) ** 2).sum(),
                AUC_wicket=roc_auc_score(y == 0, Q[:, 0]), AUC_boundary=roc_auc_score(bnd, Q[:, 5:].sum(1)))


def run(g, log):
    o, _ = paths(g); D = pd.read_parquet(f'{o}/features.parquet'); art = joblib.load(f'{o}/final_model.pkl'); F = art['feats']; cm = art['cm']
    V = dev_frame(D); X = V[F].astype(float); y = V.cls.values; gid = V.match_id.values; FOLDS = forward_folds(V.season.values); M = models(art)
    preds = {k: [] for k in M}; t0 = time.time()
    for tr, va, s in FOLDS:
        for k, f in M.items():
            m = f(X, y, tr, gid); preds[k].append(full(m.predict_proba(X.iloc[va]), m.classes_))
        log(f'  {g} fold {s} done [{time.time() - t0:.0f}s]')
    for e, parts in ENSEMBLES.items(): preds[e] = [sum(preds[p][i] for p in parts) / len(parts) for i in range(len(FOLDS))]
    va_all = np.concatenate([va for _, va, _ in FOLDS]); rows = []; perball = {}
    for k, Ps in preds.items():
        Q = np.vstack(Ps); perball[k] = -np.log(Q[np.arange(len(va_all)), y[va_all]])
        r = metrics(y[va_all], Q, V.runs_total.values[va_all], V.runs_bat.values[va_all] >= 4, cm, V.runs_total.mean()); r['model'] = k
        r['cv_logloss_by_season'] = [round(log_loss(y[va], P, labels=range(7)), 4) for P, (_, va, _) in zip(Ps, FOLDS)]; rows.append(r)
    T = pd.DataFrame(rows).set_index('model').sort_values('logloss'); ref = 'LightGBM + random forest (locked model)'
    for k in T.index:
        d_ = perball[k] - perball[ref]; bs = [d_[i].mean() for i in boot_matches(gid[va_all], 1000, 51)]
        T.loc[k, 'vs_locked'] = d_.mean(); T.loc[k, 'vs_locked_lo'] = np.percentile(bs, 2.5); T.loc[k, 'vs_locked_hi'] = np.percentile(bs, 97.5)
    best = T.index[0]; log(f'{g.upper()} cross-validated (2022-25):\n' + T.drop(columns='cv_logloss_by_season').round(5).to_string())
    # exploratory second look at 2026: refit each base model on all of 2021-25
    Te = D[(D.season == 2026) & (D.full_match == 1)].reset_index(drop=True); allidx = np.arange(len(V)); P26 = {}
    for k, f in M.items(): m = f(X, y, allidx, gid); P26[k] = full(m.predict_proba(Te[F].astype(float)), m.classes_)
    for e, parts in ENSEMBLES.items(): P26[e] = sum(P26[p] for p in parts) / len(parts)
    T26 = pd.DataFrame({k: metrics(Te.cls.values, Q, Te.runs_total.values, Te.runs_bat.values >= 4, cm, V.runs_total.mean()) for k, Q in P26.items()}).T.loc[T.index]
    log(f'{g.upper()} 2026 (exploratory second look):\n' + T26.round(5).to_string())
    T.to_csv(f'{o}/model_comparison_cv.csv'); T26.to_csv(f'{o}/model_comparison_2026.csv')
    return dict(best_cv=best, cv=T.drop(columns='cv_logloss_by_season').round(6).to_dict('index'), test2026=T26.round(6).to_dict('index'))


if __name__ == '__main__':
    out = {}
    for g in ('male', 'female'): out[g] = run(g, lambda *a: print(*a, flush=True))
    json.dump(out, open(f'{ROOT}/data/out/compare/model_comparison.json', 'w'), indent=1, default=float)
