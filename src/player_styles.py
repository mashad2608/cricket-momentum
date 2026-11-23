"""Batting hand and bowling type for every player in The Hundred (men and women).

Open sources only (no ESPNcricinfo scraping, which its terms forbid):
  1. Cricsheet player register (cricsheet.org/register/people.csv): each player's ESPNcricinfo id
  2. Wikidata (ESPNcricinfo player id, property P2697): the player's English Wikipedia article
  3. Wikipedia infobox (MediaWiki API): the 'batting' and 'bowling' fields (CC BY-SA)
Players still missing after that go to data/ref/player_styles_manual.csv to be filled in by hand; manual entries win.

Usage:  python src/player_styles.py            (uses cached downloads in data/ref/ if present)
        python src/player_styles.py --refresh  (downloads again)
Output: data/ref/player_styles.csv  (identifier, name, gender, bat_hand, bowl_arm, bowl_type, bowl_kind, source)
"""
import os, sys, re, json, glob, time
import pandas as pd, requests
ROOT = os.path.expanduser('~/hundred-momentum'); REF = f'{ROOT}/data/ref'; os.makedirs(REF, exist_ok=True)
UA = {'User-Agent': 'hundred-momentum-research/0.1 (cricket analytics script)'}
REFRESH = '--refresh' in sys.argv


def get(url, params, tries=8):
    """Polite GET: waits as long as the server asks on 429/503, with growing pauses."""
    for k in range(tries):
        r = requests.get(url, params=params, headers=UA, timeout=120)
        if r.status_code not in (429, 503): r.raise_for_status(); return r
        time.sleep(float(r.headers.get('Retry-After', 0) or 0) or 5 * 2 ** k)
    r.raise_for_status()


def players():
    ids = {}
    for g in ('hnd_male', 'hnd_female'):
        for f in glob.glob(f'{ROOT}/data/raw/{g}/*.json'):
            d = json.load(open(f)); reg = d['info']['registry']['people']
            for inn in d['innings']:
                for ov in inn['overs']:
                    for b in ov['deliveries']:
                        for role in ('batter', 'bowler'):
                            n = b[role]; r = ids.setdefault(reg[n], {'name': n, 'gender': g.split('_')[1], 'balls_bat': 0, 'balls_bowl': 0})
                            r['balls_bat' if role == 'batter' else 'balls_bowl'] += 1
    T = pd.DataFrame.from_dict(ids, orient='index'); T.index.name = 'identifier'; T['in_hundred'] = 1
    # everyone else in the history pool with at least 60 balls faced or bowled (their style is needed for history features)
    pool = f'{ROOT}/data/clean/history_pool.parquet'
    if os.path.exists(pool):
        H = pd.read_parquet(pool, columns=['batter_id', 'bowler_id', 'gender']); nb = H.groupby('batter_id').size(); nw = H.groupby('bowler_id').size()
        gmap = pd.concat([H.groupby('batter_id').gender.first(), H.groupby('bowler_id').gender.first()]).groupby(level=0).first()
        extra = sorted((set(nb[nb >= 60].index) | set(nw[nw >= 60].index)) - set(T.index))
        X = pd.DataFrame({'name': None, 'gender': gmap.reindex(extra).values, 'balls_bat': 0, 'balls_bowl': 0, 'in_hundred': 0}, index=pd.Index(extra, name='identifier'))
        T = pd.concat([T, X])
    if REFRESH or not os.path.exists(f'{REF}/people.csv'):
        open(f'{REF}/people.csv', 'wb').write(requests.get('https://cricsheet.org/register/people.csv', headers=UA, timeout=120).content)
    P = pd.read_csv(f'{REF}/people.csv', dtype=str).set_index('identifier')
    T = T.join(P[['name', 'unique_name', 'key_cricinfo', 'key_cricinfo_2']].rename(columns={'name': 'reg_name'}))
    T['name'] = T.name.fillna(T.reg_name); return T.drop(columns='reg_name').reset_index()


def wiki_titles(T):
    cache = f'{REF}/wiki_titles.json'
    if os.path.exists(cache) and not REFRESH: return json.load(open(cache))
    ids = pd.concat([T.key_cricinfo, T.key_cricinfo_2]).dropna().unique().tolist(); m = {}
    for i in range(0, len(ids), 150):
        q = 'SELECT ?cid ?article WHERE { VALUES ?cid {%s} ?item wdt:P2697 ?cid . ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> . }' % ' '.join(f'"{c}"' for c in ids[i:i + 150])
        r = get('https://query.wikidata.org/sparql', {'query': q, 'format': 'json'})
        for b in r.json()['results']['bindings']: m[b['cid']['value']] = requests.utils.unquote(b['article']['value'].split('/wiki/')[1]).replace('_', ' ')
        time.sleep(1)
    json.dump(m, open(cache, 'w'), indent=0); return m


def infoboxes(titles):
    cache = f'{REF}/wiki_infobox.json'
    out = json.load(open(cache)) if os.path.exists(cache) and not REFRESH else {}
    todo = [t for t in titles if t not in out]
    for i in range(0, len(todo), 50):
        batch = todo[i:i + 50]
        r = get('https://en.wikipedia.org/w/api.php', {'action': 'query', 'prop': 'revisions', 'rvprop': 'content', 'rvslots': 'main', 'titles': '|'.join(batch),
                                                       'redirects': 1, 'format': 'json', 'formatversion': 2, 'maxlag': 5})
        j = r.json().get('query', {})
        back = {x['to']: x['from'] for x in j.get('redirects', [])} | {x['to']: x['from'] for x in j.get('normalized', [])}
        for p in j.get('pages', []):
            t = p['revisions'][0]['slots']['main']['content'] if 'revisions' in p else ''
            out[back.get(p['title'], p['title'])] = {k: field(t, k) for k in ('batting', 'bowling')}
        for t in batch: out.setdefault(t, {'batting': None, 'bowling': None})
        json.dump(out, open(cache, 'w')); time.sleep(3)
    return out


def field(t, name):
    m = re.search(r'\|\s*' + name + r'\s*=\s*(.*)', t or '', re.I)
    if not m: return None
    v = m.group(1)
    v = re.sub(r'<ref[^>]*/>|<ref.*?</ref>|<ref.*$', '', v); v = re.sub(r'\{\{[^}]*\}\}?', '', v)
    v = re.sub(r'\[\[(?:[^\]|]*\|)?([^\]]*)\]\]', r'\1', v); v = re.sub(r'<[^>]*>', '', v)
    v = v.split('|')[0].strip(); return v or None


def classify_bat(s):
    s = (s or '').lower()
    return 'left' if 'left' in s else 'right' if 'right' in s else None


def classify_bowl(s):
    """-> (arm, type, kind). kind is pace or spin; type is the finer style."""
    s = (s or '').lower().replace('-', ' ')
    if not s.strip(): return None, None, None
    arm = 'left' if 'left' in s else 'right' if 'right' in s else None
    if any(w in s for w in ('off break', 'offbreak', 'off spin', 'offspin')): return arm or 'right', 'off spin', 'spin'
    if any(w in s for w in ('leg break', 'legbreak', 'leg spin', 'legspin', 'googly')): return arm or 'right', 'leg spin', 'spin'
    if 'orthodox' in s and 'unorthodox' not in s: return 'left', 'left-arm orthodox', 'spin'
    if any(w in s for w in ('unorthodox', 'chinaman', 'wrist')): return 'left', 'left-arm wrist spin', 'spin'
    if 'slow' in s and 'medium' not in s: return arm, 'slow (other)', 'spin'
    if 'fast' in s and 'medium' not in s: return arm, 'fast', 'pace'
    if 'fast' in s or 'medium' in s: return arm, 'medium / fast-medium', 'pace'
    return arm, None, None


def main():
    T = players(); m = wiki_titles(T)
    T['wiki_title'] = T.key_cricinfo.map(m).fillna(T.key_cricinfo_2.map(m))
    box = infoboxes(T.wiki_title.dropna().unique().tolist())
    T['batting_raw'] = T.wiki_title.map(lambda t: (box.get(t) or {}).get('batting'))
    T['bowling_raw'] = T.wiki_title.map(lambda t: (box.get(t) or {}).get('bowling'))
    T['bat_hand'] = T.batting_raw.map(classify_bat)
    T[['bowl_arm', 'bowl_type', 'bowl_kind']] = T.bowling_raw.apply(lambda s: pd.Series(classify_bowl(s)))
    T['source'] = 'wikipedia'
    # manual sheet: anyone who bowled without a type, or batted without a hand
    man = f'{REF}/player_styles_manual.csv'
    need = T[(T.in_hundred == 1) & (((T.balls_bowl > 0) & T.bowl_kind.isna()) | ((T.balls_bat > 0) & T.bat_hand.isna()))]
    if os.path.exists(man):
        M = pd.read_csv(man, dtype=str).set_index('identifier')
        for c in ('bat_hand', 'bowl_arm', 'bowl_type', 'bowl_kind'):
            v = T.identifier.map(M[c].dropna()) if c in M else None
            if v is not None: T[c] = v.fillna(T[c]); T.loc[v.notna(), 'source'] = 'manual'
        need = need[~need.identifier.isin(M.dropna(how='all', subset=[c for c in ('bat_hand', 'bowl_kind') if c in M]).index)]
    else:
        need[['identifier', 'name', 'gender', 'balls_bat', 'balls_bowl', 'wiki_title', 'batting_raw', 'bowling_raw', 'bat_hand', 'bowl_arm', 'bowl_type', 'bowl_kind']] \
            .sort_values('balls_bowl', ascending=False).to_csv(man, index=False)
    cols = ['identifier', 'name', 'gender', 'in_hundred', 'balls_bat', 'balls_bowl', 'wiki_title', 'batting_raw', 'bowling_raw', 'bat_hand', 'bowl_arm', 'bowl_type', 'bowl_kind', 'source']
    T[cols].to_csv(f'{REF}/player_styles.csv', index=False)
    print(f'all players looked up: {len(T)}, with a Wikipedia page {T.wiki_title.notna().sum()}, bowling type {T.bowl_kind.notna().sum()}, batting hand {T.bat_hand.notna().sum()}')
    T = T[T.in_hundred == 1]; bw = T[T.balls_bowl > 0]
    print(f'players {len(T)} | Wikipedia page {T.wiki_title.notna().sum()} | batting hand {T.bat_hand.notna().sum()} | bowlers {len(bw)}, typed {bw.bowl_kind.notna().sum()} '
          f'({bw.balls_bowl[bw.bowl_kind.notna()].sum() / bw.balls_bowl.sum():.1%} of balls bowled)')
    print(bw.groupby(['gender', 'bowl_kind', 'bowl_type'], dropna=False).balls_bowl.agg(['count', 'sum']).to_string())
    print('still to fill in:', len(need)); print(need[['name', 'gender', 'balls_bat', 'balls_bowl', 'batting_raw', 'bowling_raw']].to_string())


if __name__ == '__main__':
    main()
