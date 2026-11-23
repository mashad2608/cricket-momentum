"""Pre-ball features. Every feature for a ball uses earlier balls of the same innings, or matches on EARLIER DATES."""
import numpy as np, pandas as pd

BASE = dict(runs=1.4, bnd=0.18, dot=0.37, wkt=0.06)   # fixed fallbacks, used only before any history exists
HUNDRED = dict(balls=100, unit=5, pp=25, death=75)     # balls per innings, balls per set/over, powerplay balls, death phase starts
T20 = dict(balls=120, unit=6, pp=36, death=90)

def _hist(B, key, name, prior_balls, stats=('runs', 'bnd', 'dot', 'wkt')):
    """Shrunk historical rates for `key`, from matches on strictly earlier dates."""
    cols = [f'_{s}' for s in stats]
    lg = B.groupby('date')[cols].sum().assign(_n=B.groupby('date').size()).sort_index()
    lgp = lg.cumsum().shift(1).fillna(0)
    prior = {s: ((lgp[f'_{s}'] + BASE[s] * 200) / (lgp['_n'] + 200)) for s in stats}
    k = list(key) if isinstance(key, (list, tuple)) else [key]
    d = B.groupby(k + ['date'])[cols].sum().assign(_n=B.groupby(k + ['date']).size()).reset_index().sort_values('date')
    g = d.groupby(k, sort=False)
    cum = g[cols + ['_n']].cumsum() - d[cols + ['_n']]
    out = d[k + ['date']].copy(); out[f'{name}_hist_balls'] = cum['_n']
    for s in stats:
        out[f'{name}_hist_{s}'] = (cum[f'_{s}'] + prior_balls * out['date'].map(prior[s])) / (cum['_n'] + prior_balls)
    return B[k + ['date']].merge(out, on=k + ['date'], how='left').drop(columns=k + ['date']).set_index(B.index)

def _hist_sub(B, mask, key, name, prior_balls, stats=('bnd', 'dot')):
    """Like _hist, but only balls where `mask` is true count towards the history (e.g. death-phase balls)."""
    Z = B[mask]; cols = [f'_{s}' for s in stats]
    lg = B.groupby('date')[cols].sum().assign(_n=B.groupby('date').size()).sort_index(); lgp = lg.cumsum().shift(1).fillna(0)
    lz = Z.groupby('date')[cols].sum().assign(_n=Z.groupby('date').size()).reindex(lg.index, fill_value=0).cumsum().shift(1).fillna(0)
    prior = {s: (lz[f'_{s}'] + BASE[s] * 200) / (lz['_n'] + 200) for s in stats}
    d = Z.groupby([key, 'date'])[cols].sum().assign(_n=Z.groupby([key, 'date']).size()).reset_index().sort_values('date')
    cum = d.groupby(key, sort=False)[cols + ['_n']].cumsum() - d[cols + ['_n']]
    # carry each player's death-phase history forward to every later date they play
    allk = B[[key, 'date']].drop_duplicates().sort_values('date')
    d2 = d[[key, 'date']].assign(**{c: cum[c].values + d[c].values for c in cols + ['_n']})   # totals INCLUDING that date
    m = pd.merge_asof(allk, d2.sort_values('date'), on='date', by=key, allow_exact_matches=False)   # strictly earlier dates
    out = m[[key, 'date']].copy(); n_ = m['_n'].fillna(0)
    for s_ in stats: out[f'{name}_{s_}'] = (m[f'_{s_}'].fillna(0) + prior_balls * out['date'].map(prior[s_])) / (n_ + prior_balls)
    return B[[key, 'date']].merge(out, on=[key, 'date'], how='left').drop(columns=[key, 'date']).set_index(B.index)

def build(B, fmt=HUNDRED):
    N, U, PP, DTH = fmt['balls'], fmt['unit'], fmt['pp'], fmt['death']
    B = B.sort_values(['date', 'match_id', 'innings', 'seq']).reset_index(drop=True).copy()
    B['legal'] = ((B.wides == 0) & (B.noballs == 0)).astype(int)
    B['is_wicket'] = (B.n_wickets > 0).astype(int)
    B['_runs'] = B.runs_total; B['_bnd'] = (B.runs_bat >= 4).astype(int); B['_dot'] = (B.runs_total == 0).astype(int); B['_wkt'] = B.is_wicket
    inn = ['match_id', 'innings']; g = B.groupby(inn, sort=False); F = pd.DataFrame(index=B.index)
    prev = lambda c, gr: gr[c].cumsum() - B[c]
    # match state
    score, wk, lp = prev('runs_total', g), prev('n_wickets', g), prev('legal', g)
    F['innings2'] = (B.innings == 2).astype(int); F['ball_no'] = lp + 1; F['balls_left'] = N - lp; F['wkts_pre'] = wk; F['score_pre'] = score
    F['crr'] = score / lp.where(lp > 0); F['ball_in_set'] = lp % U; F['powerplay'] = (lp < PP).astype(int)
    need = (B.target - score).where(B.innings == 2)
    F['runs_needed'] = need; F['req_rate'] = need / (N - lp); F['rate_gap'] = F.req_rate - F.crr; F['chase_target'] = B.target.where(B.innings == 2)
    F['wkts_per_ball_left'] = (10 - wk) / (N - lp)
    nb = (B.noballs > 0).astype(int); F['free_hit'] = nb.groupby([B.match_id, B.innings]).shift(1).fillna(0)
    # recent team play
    for n in (5, 10, 20):
        for c in ['_runs', '_wkt', '_dot', '_bnd']:
            F[f'last{n}{c}'] = g[c].transform(lambda x: x.shift(1).rolling(n, min_periods=1).sum())
    for c, nm in [('_bnd', 'since_boundary'), ('_wkt', 'since_wicket')]:
        grp = g[c].cumsum() - B[c]; F[nm] = B.groupby([B.match_id, B.innings, grp]).cumcount()
    F['prev_runs'] = g['_runs'].shift(1)
    # partnership (resets after a wicket)
    ps = B.groupby([B.match_id, B.innings, wk], sort=False)
    F['part_runs'] = ps['runs_total'].cumsum() - B.runs_total; F['part_balls'] = ps.cumcount()
    # batter this innings
    bk = ['match_id', 'innings', 'batter_id']; bg = B.groupby(bk, sort=False); B['_bf'] = (B.wides == 0).astype(int)
    F['bat_runs'] = bg['runs_bat'].cumsum() - B.runs_bat; F['bat_balls'] = bg['_bf'].cumsum() - B._bf
    F['bat_sr'] = F.bat_runs / F.bat_balls.where(F.bat_balls > 0)
    F['bat_sr10'] = bg['runs_bat'].transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    F['bat_dots10'] = bg['_dot'].transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    # bowler this innings
    wg = B.groupby(['match_id', 'innings', 'bowler_id'], sort=False)
    F['bowl_balls'] = wg['legal'].cumsum() - B.legal; r = wg['runs_total'].cumsum() - B.runs_total
    F['bowl_econ'] = r / F.bowl_balls.where(F.bowl_balls > 0); F['bowl_wkts'] = wg['bowler_wicket'].cumsum() - B.bowler_wicket
    F['bowl_dots10'] = wg['_dot'].transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    # history: strictly earlier dates
    B['_phase'] = np.select([lp < PP, lp < DTH], [0, 1], 2)
    for key, nm, pb in [('batter_id', 'bat', 60), ('bowler_id', 'bowl', 60), (['batter_id', '_phase'], 'batph', 60), (['bowler_id', '_phase'], 'bowlph', 60), ('venue', 'venue', 400), ('season', 'season', 400)]:
        F = F.join(_hist(B, key, nm, pb))
    F = F.drop(columns=[c for c in F if c.startswith(('venue_hist_balls', 'season_hist_balls'))])
    F['form_vs_hist'] = F.bat_sr10 - F.bat_hist_runs; F['matchup_runs'] = F.bat_hist_runs + F.bowl_hist_runs; F['matchup_wkt'] = F.bat_hist_wkt + F.bowl_hist_wkt
    # ---- Step 6 additions: cricket-defined features (ANALYSIS_PLAN section 4a) ----
    F['set_batter'] = (F.bat_balls >= 15).astype(int); F['new_batter'] = (F.bat_balls <= 5).astype(int)
    B['_set'] = lp // U
    first_ball = B.groupby(['match_id', 'innings', 'bowler_id', '_set'], sort=False).cumcount() == 0
    F['bowler_set_no'] = (first_ball.astype(int).groupby([B.match_id, B.innings, B.bowler_id]).cumsum())
    setbowler = B.groupby(['match_id', 'innings', '_set'], sort=False).bowler_id.first()
    prev_b = pd.Series(list(zip(B.match_id, B.innings, B._set - 1))).map(setbowler).values
    F['consecutive_set'] = (prev_b == B.bowler_id.values).astype(int)
    B['_lp'] = lp; last = B.groupby(['match_id', 'innings', 'bowler_id'], sort=False)._lp.shift(1)
    F['balls_since_bowler_last'] = np.where(last.isna(), np.nan, lp - last)
    F['pressure_index'] = F.last10_dot / 10 + np.where(B.innings == 2, np.clip(F.rate_gap.fillna(0), 0, None), 0)
    sc = B.runs_bat.where(B.runs_bat > 0); rot = sc.isin([1, 2]).astype(float).where(sc.notna())
    F['strike_rotation'] = rot.groupby([B.match_id, B.innings]).transform(lambda x: x.shift(1).dropna().rolling(10, min_periods=3).mean().reindex(x.index).ffill())
    F = F.join(_hist_sub(B, B._phase == 2, 'batter_id', 'bat_death_hist', 60))
    F = F.replace([np.inf, -np.inf], np.nan)   # e.g. required rate when an umpire miscount leaves 0 balls left
    feats = list(F.columns)
    return pd.concat([B.drop(columns=[c for c in B if c.startswith('_')]), F], axis=1), feats
