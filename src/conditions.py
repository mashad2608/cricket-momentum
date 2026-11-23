"""Match conditions for The Hundred: weather (Open-Meteo archive) and pitch stand-ins, all known before each ball.

Weather: Open-Meteo historical archive (ERA5-based, CC BY 4.0, open-meteo.com), hourly, at each ground.
  Cricsheet has no start times, so each match gets a time window: in a double-header (men and women at the same
  ground on the same day) the women's game is the afternoon game (14:00-17:00) and the men's the evening game
  (18:00-21:00); a stand-alone game gets 14:00-21:00. Weather is averaged over that window.
Pitch stand-ins (no open source has pitch reports):
  venue_season_rpb   runs per ball at this ground earlier this season (same competition), shrunk to the ground's all-time rate
  venue_last5_rpb    runs per ball over this ground's last 5 matches (same competition)
  sameday_rpb        runs per ball in the earlier game at this ground today (evening game of a double-header only)
  inn1_rpb_vs_venue  in a chase: the first innings' runs per ball minus the ground's usual rate (how this pitch played today)
  inn1_bnd_vs_venue  in a chase: the first innings' boundary rate minus the ground's usual rate
  evening_game       the later game of a double-header (lights, dew)
  toss_bat_first / batting_won_toss
Usage:  python src/conditions.py      Output: data/clean/weather_hourly.parquet, data/out/<gender>/conditions.parquet
"""
import os, sys, json, glob, time
import numpy as np, pandas as pd, requests
ROOT = os.path.expanduser('~/hundred-momentum'); sys.path.insert(0, ROOT + '/src')
UA = {'User-Agent': 'hundred-momentum-research/0.1 (cricket analytics script)'}
GROUNDS = {'Edgbaston': (52.4559, -1.9025), 'Headingley': (53.8176, -1.5822), "Lord's": (51.5294, -0.1727), 'Old Trafford': (53.4566, -2.2869),
           'Sophia Gardens, Cardiff': (51.4875, -3.1906), 'The Oval': (51.4837, -0.1150), 'The Rose Bowl, Southampton': (50.9241, -1.3220), 'Trent Bridge': (52.9370, -1.1322)}
HOURLY = ['temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'cloud_cover', 'wind_speed_10m', 'precipitation']


def weather():
    path = f'{ROOT}/data/clean/weather_hourly.parquet'
    if os.path.exists(path): return pd.read_parquet(path)
    out = []
    for v, (la, lo) in GROUNDS.items():
        for yr in range(2021, 2027):
            r = requests.get('https://archive-api.open-meteo.com/v1/archive', params=dict(latitude=la, longitude=lo, start_date=f'{yr}-07-01', end_date=f'{yr}-09-30',
                             hourly=','.join(HOURLY), timezone='Europe/London'), headers=UA, timeout=120)
            r.raise_for_status(); h = r.json()['hourly']
            out.append(pd.DataFrame(h).assign(venue=v)); time.sleep(1)
    W = pd.concat(out, ignore_index=True); W['time'] = pd.to_datetime(W.time); W.to_parquet(path, index=False); return W


def match_info():
    rows = []
    for g in ('male', 'female'):
        for f in glob.glob(f'{ROOT}/data/raw/hnd_{g}/*.json'):
            i = json.load(open(f))['info']
            rows.append(dict(match_id=os.path.basename(f)[:-5], gender=g, date=pd.Timestamp(i['dates'][0]), venue=i['venue'], toss_winner=i['toss']['winner'], toss_decision=i['toss']['decision']))
    M = pd.DataFrame(rows)
    both = M.groupby(['venue', 'date']).gender.nunique()
    M['double_header'] = M.set_index(['venue', 'date']).index.map(both).values > 1
    M['evening_game'] = (M.double_header & (M.gender == 'male')).astype(int)
    M['h0'] = np.where(~M.double_header, 14, np.where(M.gender == 'female', 14, 18)); M['h1'] = np.where(~M.double_header, 21, np.where(M.gender == 'female', 17, 21))
    return M


def build(g, W, M):
    D = pd.read_parquet(f'{ROOT}/data/out/{g}/features.parquet'); D['match_id'] = D.match_id.astype(str)
    Mg = M.copy(); Mg['match_id'] = Mg.match_id.astype(str)
    # weather over each match's window
    W = W.assign(date=W.time.dt.normalize(), hour=W.time.dt.hour)
    wx = []
    for r in Mg.itertuples():
        z = W[(W.venue == r.venue) & (W.date == r.date) & (W.hour >= r.h0) & (W.hour <= r.h1)]
        wx.append(dict(match_id=r.match_id, **{f'wx_{c.split("_")[0] if c != "relative_humidity_2m" else "humidity"}': z[c].mean() for c in HOURLY}))
    WX = pd.DataFrame(wx).rename(columns={'wx_dew': 'wx_dewpoint', 'wx_cloud': 'wx_cloud', 'wx_wind': 'wx_wind', 'wx_temperature': 'wx_temp', 'wx_precipitation': 'wx_rain'})
    WX['wx_dew_spread'] = WX.wx_temp - WX.wx_dewpoint                     # small spread = damp air, dew likely
    # per-match scoring, for ground form and same-day pitch
    MS = D.groupby('match_id').agg(date=('date', 'first'), venue=('venue', 'first'), rpb=('runs_total', 'mean'), bnd=('runs_bat', lambda s: (s >= 4).mean()), season=('season', 'first')).reset_index()
    MS['date'] = pd.to_datetime(MS.date); MS = MS.sort_values('date')
    allv = MS.groupby('venue').rpb.mean(); prior_n = 3
    def prior_mean(df, col, n):
        return df.groupby('venue')[col].transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())
    MS['venue_last5_rpb'] = prior_mean(MS, 'rpb', 5)
    cs = MS.groupby(['venue', 'season']).rpb.transform(lambda s: s.shift(1).expanding().sum()).fillna(0); cn = MS.groupby(['venue', 'season']).cumcount()
    base = MS.groupby('venue').rpb.transform(lambda s: s.shift(1).expanding().mean()).fillna(MS.rpb.mean())
    MS['venue_season_rpb'] = (cs + prior_n * base) / (cn + prior_n)
    MS['venue_hist_rpb'] = base; MS['venue_hist_bnd'] = MS.groupby('venue').bnd.transform(lambda s: s.shift(1).expanding().mean()).fillna(MS.bnd.mean())
    # same-day earlier game (the women's game before a men's evening game)
    other = f'{ROOT}/data/out/{"female" if g == "male" else "male"}/features.parquet'
    O = pd.read_parquet(other, columns=['match_id', 'date', 'venue', 'runs_total']); O['date'] = pd.to_datetime(O.date)
    od = O.groupby(['venue', 'date']).runs_total.mean().rename('sameday_rpb').reset_index()
    C = MS.merge(Mg[['match_id', 'toss_winner', 'toss_decision', 'evening_game']], on='match_id', how='left').merge(WX, on='match_id', how='left')
    C = C.merge(od, on=['venue', 'date'], how='left'); C.loc[C.evening_game == 0, 'sameday_rpb'] = np.nan
    # first-innings read of the pitch, for chase balls
    i1 = D[D.innings == 1].groupby('match_id').agg(i1_rpb=('runs_total', 'mean'), i1_bnd=('runs_bat', lambda s: (s >= 4).mean())).reset_index()
    C = C.merge(i1, on='match_id', how='left')
    X = D[['match_id', 'innings', 'batting_team']].merge(C, on='match_id', how='left')
    out = pd.DataFrame(index=D.index)
    for c in ['wx_temp', 'wx_humidity', 'wx_dewpoint', 'wx_cloud', 'wx_wind', 'wx_rain', 'wx_dew_spread', 'venue_season_rpb', 'venue_last5_rpb', 'sameday_rpb', 'evening_game']: out[c] = X[c].values
    chase = X.innings.values == 2
    out['inn1_rpb_vs_venue'] = np.where(chase, X.i1_rpb - X.venue_hist_rpb, np.nan); out['inn1_bnd_vs_venue'] = np.where(chase, X.i1_bnd - X.venue_hist_bnd, np.nan)
    bat_first_team = np.where(X.toss_decision == 'bat', X.toss_winner, None)
    out['toss_bat_first'] = (X.toss_decision == 'bat').astype(int).values; out['batting_won_toss'] = (X.toss_winner == X.batting_team).astype(int).values
    out.to_parquet(f'{ROOT}/data/out/{g}/conditions.parquet')
    cover = {c: float(out[c].notna().mean()) for c in out}; print(g, 'conditions built; coverage', {k: round(v, 2) for k, v in cover.items()})
    return out


if __name__ == '__main__':
    W = weather(); M = match_info(); print('weather hours', len(W), '| matches', len(M), '| double-headers', int(M.double_header.sum() / 2))
    for g in ('male', 'female'): build(g, W, M)
