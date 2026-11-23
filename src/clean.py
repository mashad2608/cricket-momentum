"""Step 1 as a function: flatten Cricsheet JSON for one competition, check it, flag full matches, add pre-ball state."""
import json, glob, os
import numpy as np, pandas as pd

NOT_A_WICKET = {'retired hurt', 'retired not out'}
NOT_BOWLERS = {'run out', 'retired out', 'retired hurt', 'retired not out', 'obstructing the field', 'timed out', 'handled the ball'}

def clean(raw_dir, log=print, n_balls=100, n_franchises=8):
    files = sorted(glob.glob(f'{raw_dir}/*.json')); balls, matches, unavailable = [], [], []
    for f in files:
        d = json.load(open(f)); i = d['info']; mid = os.path.basename(f)[:-5]; o = i['outcome']; reg = i['registry']['people']
        matches.append(dict(match_id=mid, date=i['dates'][0], season=int(str(i['season'])[:4]), venue=i.get('venue'), team1=i['teams'][0], team2=i['teams'][1],
            winner=o.get('winner') or o.get('eliminator'), result=o.get('result', 'win'), method=o.get('method'), by_runs=o.get('by', {}).get('runs'), by_wickets=o.get('by', {}).get('wickets'),
            n_innings=sum(not x.get('super_over') for x in d['innings'])))
        for k, inn in enumerate([x for x in d['innings'] if not x.get('super_over')], 1):
            bat = inn['team']; bowl = [t for t in i['teams'] if t != bat][0]; tg = inn.get('target', {})
            for pl in inn.get('absent_hurt', []): unavailable.append((mid, k, pl))
            seq = 0
            for ov in inn['overs']:
                for b in ov['deliveries']:
                    seq += 1; e = b.get('extras', {}); w = [x for x in b.get('wickets', []) if x['kind'] not in NOT_A_WICKET]
                    for x in b.get('wickets', []):
                        if x['kind'] in NOT_A_WICKET: unavailable.append((mid, k, x['player_out']))
                    balls.append(dict(match_id=mid, innings=k, seq=seq, over=ov['over'], batting_team=bat, bowling_team=bowl, batter=b['batter'], batter_id=reg.get(b['batter']),
                        non_striker=b['non_striker'], non_striker_id=reg.get(b['non_striker']), bowler=b['bowler'], bowler_id=reg.get(b['bowler']),
                        runs_bat=b['runs']['batter'], runs_extra=b['runs']['extras'], runs_total=b['runs']['total'], wides=e.get('wides', 0), noballs=e.get('noballs', 0),
                        byes=e.get('byes', 0), legbyes=e.get('legbyes', 0), penalty=e.get('penalty', 0), n_wickets=len(w), wicket_kind=w[0]['kind'] if w else None,
                        player_out=w[0]['player_out'] if w else None, bowler_wicket=int(any(x['kind'] not in NOT_BOWLERS for x in w)), target=tg.get('runs'), target_overs=tg.get('overs')))
    M = pd.DataFrame(matches); M['date'] = pd.to_datetime(M['date']); B = pd.DataFrame(balls)
    assert B.batter_id.notna().all() and B.bowler_id.notna().all() and not B.duplicated(['match_id', 'innings', 'seq']).any()
    assert (B.runs_bat + B.runs_extra == B.runs_total).all() and (B[['wides', 'noballs', 'byes', 'legbyes', 'penalty']].sum(axis=1) == B.runs_extra).all()
    B['legal'] = ((B.wides == 0) & (B.noballs == 0)).astype(int)
    # Cricsheet's 2026 files omit the chase target: fill it as first-innings runs + 1, only for games without a rain rule
    miss = (B.innings == 2) & B.target.isna() & B.match_id.map(M.set_index('match_id').method).isna()
    if miss.any():
        r1 = B[B.innings == 1].groupby('match_id').runs_total.sum(); B.loc[miss, 'target'] = B.loc[miss, 'match_id'].map(r1) + 1; B.loc[miss, 'target_overs'] = 20
        log(f'target filled for {B.loc[miss].match_id.nunique()} matches with no target recorded')
    # franchises: link renamed teams by main home ground
    home = M.groupby('team1').venue.agg(lambda s: s.value_counts().index[0]); first = {t: M[(M.team1 == t) | (M.team2 == t)].season.min() for t in set(M.team1) | set(M.team2)}
    norm = lambda v: None if v is None else v.split(',')[0].replace('The ', '').replace('Kennington ', '').strip(); F = {}
    for t in sorted(first, key=lambda x: (first[x], x)):
        same = [u for u in F if norm(home.get(u)) == norm(home.get(t))]; F[t] = F[same[0]] if same else t
    if n_franchises: assert len(set(F.values())) == n_franchises, F
    for c in ['batting_team', 'bowling_team']: B[c.replace('team', 'franchise')] = B[c].map(F)
    # check against official results
    tot = B.groupby(['match_id', 'innings']).agg(team=('batting_team', 'first'), runs=('runs_total', 'sum'), wkts=('n_wickets', 'sum'), legal=('legal', 'sum'), target=('target', 'first')).reset_index()
    a = tot[tot.innings == 1].set_index('match_id'); b = tot[tot.innings == 2].set_index('match_id')
    x = M.set_index('match_id').join(a[['team', 'runs', 'wkts', 'legal']].add_suffix('_1')).join(b[['team', 'runs', 'wkts', 'legal', 'target']].add_suffix('_2'))
    std = x[(x.result == 'win') & x.method.isna()]
    ok = (std.by_runs.isna() | ((std.runs_1 - std.runs_2 == std.by_runs) & (std.winner == std.team_1))) & (std.by_wickets.isna() | ((10 - std.wkts_2 == std.by_wickets) & (std.winner == std.team_2))) & (std.target_2 == std.runs_1 + 1)
    log(f'standard games {len(std)}, matching official result {int(ok.sum())}; innings over {n_balls} legal balls {int((tot.legal > n_balls).sum())}')
    U = pd.DataFrame(unavailable, columns=['match_id', 'innings', 'player']).drop_duplicates()
    def resumed(r):
        z = B[(B.match_id == r.match_id) & (B.innings == r.innings)]; return (z.tail(1)[['batter', 'non_striker']] == r.player).any().any() or (z.player_out == r.player).any()
    U = U[~U.apply(resumed, axis=1)] if len(U) else U; nu = U.groupby(['match_id', 'innings']).size()
    lim = lambda k: 10 - x.index.map(lambda q: nu.get((q, k), 0))
    tov = B[B.innings == 2].groupby('match_id').target_overs.first()
    full = (x.n_innings == 2) & x.result.isin(['win', 'tie']) & x.method.isna() & ((x.legal_1 >= n_balls) | (x.wkts_1 >= lim(1))) & ((x.legal_2 >= n_balls) | (x.wkts_2 >= lim(2)) | (x.runs_2 >= x.target_2)) & (tov.reindex(x.index) == 20)
    M = M.merge(full.astype(int).rename('full_match').reset_index(), on='match_id'); log(f'matches {len(M)}, full {int(M.full_match.sum())}')
    B = B.merge(M[['match_id', 'date', 'season', 'venue', 'full_match']], on='match_id').sort_values(['date', 'match_id', 'innings', 'seq']).reset_index(drop=True)
    return B, M
