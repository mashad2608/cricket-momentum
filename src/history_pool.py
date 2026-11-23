"""One table of every ball faced in the T20 (and Hundred) matches we hold, for building player history features.

Competitions: The Hundred, T20 Blast, Charlotte Edwards Cup, Women's T20 Blast, IPL, BBL, PSL, CPL, SA20, ILT20, MLC, Super Smash,
BPL, LPL, men's and women's T20 internationals, WBBL, WPL, Women's CPL, Kia Super League (all Cricsheet, ODC-BY).
Player ids are Cricsheet's register ids, which are the same across competitions.

Usage:  python src/history_pool.py
Output: data/clean/history_pool.parquet (one row per ball faced: wides excluded)
"""
import os, json, glob
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
ROOT = os.path.expanduser('~/hundred-momentum')
COMPS = ['hnd_male', 'hnd_female', 'ntb_male', 'cec_female', 'wtb_female', 'ipl', 'bbl', 'psl', 'cpl', 'sat', 'ilt', 'mlc', 'ssm', 'bpl', 'lpl', 't20s_male',
         'wbb', 'wpl', 'wcl', 'wsl', 't20s_female']
NOT_BOWLER = {'run out', 'retired hurt', 'retired out', 'obstructing the field', 'retired not out', 'timed out', 'handled the ball'}


def venue_key(v):
    return str(v).split(',')[0].strip().lower()


def parse(path):
    try:
        d = json.load(open(path)); info = d['info']
    except Exception:
        return []
    if info.get('match_type') not in ('T20', 'IT20') and 'hnd_' not in path: return []
    reg = info.get('registry', {}).get('people', {}); date = info['dates'][0]; venue = venue_key(info.get('venue', '')); comp = path.split('/')[-2]
    mid = os.path.basename(path)[:-5]; rows = []
    for inn in d.get('innings', []):
        if inn.get('super_over'): continue
        for ov in inn.get('overs', []):
            for b in ov['deliveries']:
                ex = b.get('extras', {})
                if 'wides' in ex: continue
                rb = b['runs']['batter']; w = b.get('wickets', [])
                out = any(x.get('player_out') == b['batter'] and x.get('kind') not in NOT_BOWLER for x in w)
                rows.append((mid, comp, info.get('gender'), date, venue, reg.get(b['batter']), reg.get(b['bowler']), rb, int(rb == 4), int(rb == 6), int(rb == 0 and not out), int(out)))
    return rows


def main():
    files = [f for c in COMPS for f in glob.glob(f'{ROOT}/data/raw/{c}/*.json')]
    with ProcessPoolExecutor() as ex: parts = list(ex.map(parse, files, chunksize=64))
    H = pd.DataFrame([r for p in parts for r in p], columns=['match_id', 'comp', 'gender', 'date', 'venue', 'batter_id', 'bowler_id', 'runs', 'four', 'six', 'dot', 'out'])
    H = H.drop_duplicates()                       # the same match can sit in two folders (e.g. an international listed twice)
    H['date'] = pd.to_datetime(H.date); H = H.dropna(subset=['batter_id', 'bowler_id'])
    H.to_parquet(f'{ROOT}/data/clean/history_pool.parquet', index=False)
    print(f'{len(H):,} balls faced from {H.match_id.nunique():,} matches;', H.groupby('comp').match_id.nunique().to_dict())


if __name__ == '__main__':
    main()
