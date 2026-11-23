"""EXPLORATORY deep analysis (added after the frozen plan): the small things that move momentum, measured against the model.

Everything uses the out-of-fold ball values (2022-2026): each ball's expected outcome comes from a model that never saw
that season, so "beat expectation" means better than the situation, players and conditions suggested.

  1. Bowlers by phase    each bowler's runs conceded v expected in balls 1-20 v balls 81-100 (and every 20-ball part);
                         a shuffle test asks whether phase specialists are real or mostly noise
  2. Game in parts       which 20-ball part of each innings is most linked to winning
  3. Small events        what happens to momentum after a boundary, two boundaries, a wicket, a wide or no-ball,
                         three dots, a new bowler, a bowler's back-to-back set, a new batter, the end of the powerplay
  4. Boundary clusters   is a boundary more likely straight after a boundary than the model expects, and when

Usage:  python src/deep_analysis.py
Output: data/out/compare/deep_analysis.json and dashboard tables in dashboard/deep_*.csv
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
ROOT = os.path.expanduser('~/hundred-momentum'); OUT = f'{ROOT}/data/out'; DB = f'{ROOT}/dashboard'
G = {'male': 'Men', 'female': 'Women'}
PARTS = ['Balls 1–20', 'Balls 21–40', 'Balls 41–60', 'Balls 61–80', 'Balls 81–100']


def load(g):
    return prep(pd.read_parquet(f'{OUT}/{g}/values.parquet'))


def prep(d):
    d = d.sort_values(['match_id', 'innings', 'seq']).reset_index(drop=True)
    S = pd.read_csv(f'{ROOT}/data/ref/player_styles.csv', dtype=str).set_index('identifier')
    d['kind'] = d.bowler_id.map(S.bowl_kind); d['part'] = np.minimum((d.ball_no - 1) // 20, 4)
    d['bnd'] = (d.runs_bat >= 4).astype(int); d['e_bnd'] = d.p_4 + d.p_5p; d['dot'] = ((d.runs_total == 0) & (d.is_wicket == 0)).astype(int)
    d['resid_runs'] = d.runs_total - d.exp_runs; d['resid_wkt'] = d.is_wicket - d.p_W
    return d


def boot_mean(v, ids, n=2000, seed=42):
    s = pd.DataFrame({'v': v, 'i': ids}).dropna().groupby('i').v.agg(['sum', 'size'])
    if len(s) < 5: return np.nan, np.nan
    rng = np.random.default_rng(seed); u = len(s); out = [s['sum'].values[k].sum() / s['size'].values[k].sum() for k in (rng.choice(u, u) for _ in range(n))]
    return float(np.quantile(out, .025)), float(np.quantile(out, .975))


# ---------------------------------------------------------------- 1. bowlers by phase
def bowler_phases(d, lab, k=60, min_balls=24, n_perm=500):
    legal = d[d.legal == 1]
    A = legal.groupby(['bowler_id', 'bowler', 'part']).agg(n=('resid_runs', 'size'), resid=('resid_runs', 'sum'), w=('resid_wkt', 'sum')).reset_index()
    A['runs_saved_per_ball'] = -A.resid / (A.n + k)                       # shrunk; positive = cheaper than expected
    W = A.pivot_table(index=['bowler_id', 'bowler'], columns='part', values=['n', 'runs_saved_per_ball'])
    early, late = 0, 4
    ok = (W[('n', early)] >= min_balls) & (W[('n', late)] >= min_balls)
    T = pd.DataFrame({'bowler': W.index.get_level_values('bowler'), 'balls_first20': W[('n', early)].values, 'balls_last20': W[('n', late)].values,
                      'saved_first20': W[('runs_saved_per_ball', early)].values, 'saved_last20': W[('runs_saved_per_ball', late)].values}, index=W.index.get_level_values('bowler_id'))[ok.values]
    T['late_minus_early'] = T.saved_last20 - T.saved_first20; T['gender'] = lab
    T['kind'] = T.index.map(d.drop_duplicates('bowler_id').set_index('bowler_id').kind)
    # is phase specialism real? spread of (late - early) across bowlers v shuffling which of a bowler's balls fall early or late
    Z = legal[legal.part.isin([early, late]) & legal.bowler_id.isin(T.index)][['bowler_id', 'part', 'resid_runs']].copy()
    def spread(z):
        g = z.groupby(['bowler_id', 'part']).resid_runs.agg(['sum', 'size']).unstack()
        s = -g['sum'] / (g['size'] + k); return float((s[late] - s[early]).std())
    obs = spread(Z); rng = np.random.default_rng(43); null = []
    for _ in range(n_perm):
        z = Z.copy(); z['part'] = z.groupby('bowler_id').part.transform(lambda p: rng.permutation(p.values)); null.append(spread(z))
    test = dict(bowlers=int(len(T)), spread_observed=obs, spread_shuffled=float(np.mean(null)), p=float((np.sum(np.array(null) >= obs) + 1) / (n_perm + 1)),
                corr_first_last=float(np.corrcoef(T.saved_first20, T.saved_last20)[0, 1]) if len(T) > 3 else np.nan)
    # by kind of bowler, every part
    K = legal.dropna(subset=['kind']).groupby(['kind', 'part']).agg(n=('resid_runs', 'size'), resid=('resid_runs', 'mean'), wkt=('resid_wkt', 'mean')).reset_index()
    K['runs_saved_per_ball'] = -K.resid; K['part_label'] = K.part.map(dict(enumerate(PARTS))); K['gender'] = lab
    return T.reset_index(), test, K


# ---------------------------------------------------------------- 2. which part of the innings decides matches
def game_parts(d, g, lab):
    M = pd.read_parquet(f'{OUT}/{g}/matches.parquet'); M['match_id'] = M.match_id.astype(str); d = d.assign(match_id=d.match_id.astype(str))
    d = d.merge(M[['match_id', 'winner', 'result']], on='match_id'); d = d[d.result == 'win']
    P = d.groupby(['match_id', 'innings', 'part']).agg(gap=('gap_std', 'sum'), team=('batting_team', 'first'), winner=('winner', 'first')).reset_index()
    P['won'] = (P.team == P.winner).astype(int); rows = []
    for (inn, part), z in P.groupby(['innings', 'part']):
        r = np.corrcoef(z.gap, z.won)[0, 1]; rng = np.random.default_rng(42); bs = []
        for _ in range(1000):
            k = rng.choice(len(z), len(z)); bs.append(np.corrcoef(z.gap.values[k], z.won.values[k])[0, 1])
        rows.append(dict(gender=lab, innings='1st innings' if inn == 1 else 'Chase', part=PARTS[part], corr_with_winning=float(r), lo=float(np.nanquantile(bs, .025)), hi=float(np.nanquantile(bs, .975)), matches=int(len(z))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 3. small events
def events(d, lab):
    grp = d.groupby(['match_id', 'innings'], sort=False)
    def fwd(col, k, include_now=False):                      # mean of `col` over the next k balls (or this + next k-1)
        s = grp[col].transform(lambda x: x[::-1].rolling(k, min_periods=k).mean()[::-1])
        return s if include_now else grp[col].shift(-1).groupby([d.match_id, d.innings]).transform(lambda x: x[::-1].rolling(k, min_periods=k).mean()[::-1])
    prev = lambda c, n=1: grp[c].shift(n)
    nxt1, nxt5 = fwd('gap_std', 1), fwd('gap_std', 5); now5 = fwd('gap_std', 5, True)
    E = {
        'After a four or six (next ball)': (d.bnd == 1, nxt1),
        'After two boundaries in a row (next ball)': ((d.bnd == 1) & (prev('bnd') == 1), nxt1),
        'After a six (next ball)': (d.runs_bat >= 6, nxt1),
        'After a wicket (next 5 balls)': (d.is_wicket == 1, nxt5),
        'After a wide (next ball)': (d.wides > 0, nxt1),
        'After a no-ball: the free hit (next ball)': (d.noballs > 0, nxt1),
        'After three dots in a row (next 5 balls)': ((d.dot == 1) & (prev('dot') == 1) & (prev('dot', 2) == 1), nxt5),
        "A bowler's first 5 balls of the match": ((d.ball_in_set == 1) & (d.bowler_set_no == 1), now5),
        "A bowler's back-to-back set (first 5 balls)": ((d.ball_in_set == 1) & (d.consecutive_set == 1), now5),
        "A new batter's first 5 balls": ((d.new_batter == 1) & (d.bat_balls == 0), now5),
        'First 5 balls after the powerplay': (d.ball_no == 26, now5),
        'Last 5 balls of the innings': (d.ball_no == 96, now5),
    }
    rows = []
    for name, (mask, val) in E.items():
        for inn, il in ((1, '1st innings'), (2, 'Chase')):
            m = mask.fillna(False).values & (d.innings.values == inn) & val.notna().values
            if m.sum() < 30: continue
            lo, hi = boot_mean(val.values[m], d.match_id.values[m])
            rows.append(dict(gender=lab, event=name, innings=il, n=int(m.sum()), momentum_after=float(val.values[m].mean()), lo=lo, hi=hi))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 4. boundary clusters
def boundary_clusters(d, lab):
    grp = d.groupby(['match_id', 'innings'], sort=False); prev_b = grp.bnd.shift(1); prev2 = grp.bnd.shift(2); same_set = d.ball_in_set > 1
    conds = {'After a boundary': prev_b == 1, 'After no boundary': prev_b == 0, 'After two boundaries': (prev_b == 1) & (prev2 == 1),
             'After a boundary, same bowler': (prev_b == 1) & same_set, 'After a boundary, new bowler': (prev_b == 1) & ~same_set,
             'After a boundary, v spin': (prev_b == 1) & (d.kind == 'spin'), 'After a boundary, v pace': (prev_b == 1) & (d.kind == 'pace'),
             'After a boundary, set batter (10+ balls)': (prev_b == 1) & (d.bat_balls >= 10), 'After a boundary, batter under 10 balls': (prev_b == 1) & (d.bat_balls < 10),
             'After a boundary, powerplay': (prev_b == 1) & (d.ball_no <= 25), 'After a boundary, death (76+)': (prev_b == 1) & (d.ball_no >= 76)}
    rows = []
    for name, m in conds.items():
        m = m.fillna(False).values
        lo, hi = boot_mean((d.bnd - d.e_bnd).values[m], d.match_id.values[m])
        rows.append(dict(gender=lab, condition=name, n=int(m.sum()), boundary_rate=float(d.bnd.values[m].mean()), expected=float(d.e_bnd.values[m].mean()),
                         excess=float((d.bnd - d.e_bnd).values[m].mean()), lo=lo, hi=hi))
    return pd.DataFrame(rows)


def main():
    os.makedirs(DB, exist_ok=True); res, BP, BK, GP, EV, BC = {}, [], [], [], [], []
    for g, lab in G.items():
        d = load(g)
        T, test, K = bowler_phases(d, lab); BP.append(T); BK.append(K)
        GP.append(game_parts(d, g, lab)); EV.append(events(d, lab)); BC.append(boundary_clusters(d, lab))
        res[g] = dict(specialist_test=test); print(lab, 'phase-specialist test:', {k: round(v, 4) if isinstance(v, float) else v for k, v in test.items()})
    BP, BK, GP, EV, BC = map(lambda x: pd.concat(x, ignore_index=True), (BP, BK, GP, EV, BC))
    BP.round(4).to_csv(f'{DB}/deep_bowler_phase.csv', index=False); BK.round(4).to_csv(f'{DB}/deep_bowler_kind_parts.csv', index=False)
    GP.round(4).to_csv(f'{DB}/deep_game_parts.csv', index=False); EV.round(4).to_csv(f'{DB}/deep_events.csv', index=False); BC.round(4).to_csv(f'{DB}/deep_boundary_clusters.csv', index=False)
    pd.DataFrame([dict(gender=G[g], **res[g]['specialist_test']) for g in G]).round(4).to_csv(f'{DB}/deep_specialist_test.csv', index=False)
    json.dump(res, open(f'{OUT}/compare/deep_analysis.json', 'w'), indent=1, default=float)
    pd.set_option('display.width', 200)
    print(GP.to_string()); print(EV.to_string()); print(BC.to_string())
    print(BK.pivot_table(index=['gender', 'kind'], columns='part_label', values='runs_saved_per_ball').round(3).to_string())
    for lab in G.values():
        z = BP[BP.gender == lab].sort_values('late_minus_early')
        print(lab, 'better early than late:\n', z.head(5)[['bowler', 'kind', 'balls_first20', 'balls_last20', 'saved_first20', 'saved_last20']].round(3).to_string(index=False))
        print(lab, 'better late than early:\n', z.tail(5)[['bowler', 'kind', 'balls_first20', 'balls_last20', 'saved_first20', 'saved_last20']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
