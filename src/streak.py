"""Batter-level streak (hot-hand) test, shared by the exploratory Hundred check and the fresh-data confirmation.

Unit: one batter's innings, balls actually faced (wides excluded), in order.
Windows: non-overlapping blocks of k balls; regress the next block's summed gap on the last block's.
Null: shuffle the balls WITHIN each batter-innings (keeps the batter's form on the day, destroys order).
detrend=True first subtracts the average gap at each balls-faced position (1..30, then 31+),
so a batter simply settling in faster than the model expects cannot create a positive slope.
"""
import numpy as np, pandas as pd


def slope(p, f): return np.polyfit(p, f, 1)[0]


def boot_matches(ids, n, seed):
    rng = np.random.default_rng(seed); u, inv = np.unique(ids, return_inverse=True); rows = [np.where(inv == j)[0] for j in range(len(u))]
    return [np.concatenate([rows[j] for j in rng.integers(0, len(u), len(u))]) for _ in range(n)]


def faced(d, detrend=False, cap=30):
    """detrend: False (raw), True or 'settling' (balls-faced position), 'full' (situation cell, then position)."""
    f = d[d.wides == 0].sort_values(['date', 'match_id', 'innings', 'seq']).copy()
    f['pos'] = np.minimum(f.groupby(['match_id', 'innings', 'batter_id']).cumcount(), cap)
    f['g'] = f.gap_std
    if detrend == 'full': f['g'] = f.g - f.groupby([f.innings, (f.ball_no - 1) // 10, np.minimum(f.wkts_pre, 6)]).g.transform('mean')
    if detrend: f['g'] = f.g - f.groupby('pos').g.transform('mean')
    return f


def streak_test(d, k=5, detrend=False, n_shuffle=1000, n_boot=2000, seed=42):
    f = faced(d, detrend)
    seqs = [(mid, x.g.values) for (mid, inn, b), x in f.groupby(['match_id', 'innings', 'batter_id'], sort=False) if len(x) >= 2 * k]
    def win(vs):
        P_, F_, M_ = [], [], []
        for mid, v in vs:
            for s in range(k, len(v) - k + 1, k): P_.append(v[s - k:s].sum()); F_.append(v[s:s + k].sum()); M_.append(mid)
        return np.array(P_), np.array(F_), np.array(M_)
    p, fu, mids = win(seqs); obs = slope(p, fu)
    bs = [slope(p[i], fu[i]) for i in boot_matches(mids, n_boot, seed)]
    rng = np.random.default_rng(seed + 1); null = np.array([slope(*win([(m, rng.permutation(v)) for m, v in seqs])[:2]) for _ in range(n_shuffle)])
    return dict(window=k, detrended=detrend, batter_innings=len(seqs), windows=len(p), slope=float(obs), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                shuffle_null_mean=float(null.mean()), excess_slope=float(obs - null.mean()), shuffle_p=float((np.sum(null >= obs) + 1) / (n_shuffle + 1)))


def settling_curve(d):
    f = faced(d); return f.groupby('pos').gap_std.agg(['mean', 'size']).reset_index()
