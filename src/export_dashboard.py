"""Export the project's findings as small CSV tables for the dashboard (and Power BI).

Usage:  python src/export_dashboard.py
Output: dashboard/*.csv  (every number is read from the saved result files; nothing is retyped)
"""
import os, sys, json
import numpy as np, pandas as pd
ROOT = os.path.expanduser('~/hundred-momentum'); sys.path.insert(0, ROOT + '/src')
OUT = f'{ROOT}/data/out'; D = f'{ROOT}/dashboard'; os.makedirs(D, exist_ok=True)
J = lambda p: json.load(open(p))
G = {'male': 'Men', 'female': 'Women'}


def main():
    R = {g: J(f'{OUT}/{g}/results.json') for g in G}; C = J(f'{OUT}/confirm/results.json'); HH = J(f'{OUT}/compare/hot_hand_summary.json')
    MD = J(f'{OUT}/compare/momentum_drivers.json'); MC = J(f'{OUT}/compare/model_comparison.json')
    PSM = {g: pd.read_csv(f'{OUT}/{g}/rq3_psm.csv', index_col=0) for g in G}

    # headline figures (one row per number)
    h = []
    for g, lab in G.items():
        ht = pd.DataFrame(R[g]['HT']); q = R[g]['RQ2']['H2a']; d3 = PSM[g].loc['D3 fresh bowler after hot spell']
        h += [dict(metric='r2_one_ball', gender=lab, value=ht.query('innings==1 and window==1').R2_descriptive.iloc[0]),
              dict(metric='r2_innings', gender=lab, value=ht.query('innings==1 and window==100').R2_descriptive.iloc[0]),
              dict(metric='disagree_first_innings', gender=lab, value=q['disagree_inn1']),
              dict(metric='disagree_chase', gender=lab, value=q['disagree_inn2']),
              dict(metric='hot_hand_excess_slope_fresh', gender=lab, value=C[g]['primary']['excess_slope']),
              dict(metric='hot_hand_holm_p_fresh', gender=lab, value=C[g]['primary']['holm_p']),
              dict(metric='hot_hand_r2_fresh', gender=lab, value=HH[f'fresh_{g}']['R2']),
              dict(metric='psm_fresh_bowler_effect_sd', gender=lab, value=d3.att),
              dict(metric='psm_fresh_bowler_p', gender=lab, value=d3.p)]
    pd.DataFrame(h).to_csv(f'{D}/headline.csv', index=False)

    # research-question scorecard (text, as in the report)
    pd.DataFrame([
        ['Frozen', 'Thesis', 'Does a trend describe play better than a single ball?', 'Yes', 'Yes'],
        ['Frozen', 'RQ1', 'Does team momentum carry forward as streaks?', 'No (form on the day)', 'No (form on the day)'],
        ['Frozen', 'RQ2', 'Is model momentum better than scoreboard momentum, especially in chases?', 'Yes', 'Yes'],
        ['Frozen', 'RQ3', 'Do coaching decisions change momentum (PSM)?', 'No effect found', 'No effect found'],
        ['Frozen', 'RQ4', 'Do men and women differ?', 'No significant difference', 'No significant difference'],
        ['Fresh data', 'H5', 'Do batters run hot over 5 balls?', 'Confirmed (T20 Blast)', 'Confirmed (English T20)'],
        ['Exploratory', 'Drivers', 'What moves momentum, 1st innings v chase?', 'Wickets in 1st innings; dots and singles in chases', 'Same pattern; fours lead'],
    ], columns=['status', 'id', 'question', 'men', 'women']).to_csv(f'{D}/scorecard.csv', index=False)

    # thesis: R² by window length
    pd.concat([pd.DataFrame(R[g]['HT']).assign(gender=lab)[['gender', 'innings', 'window', 'R2_descriptive']] for g, lab in G.items()]) \
        .assign(innings=lambda t: t.innings.map({1: '1st innings', 2: 'Chase'})).rename(columns={'R2_descriptive': 'r2'}).round(4).to_csv(f'{D}/trend_r2.csv', index=False)

    # hot hand before (The Hundred) and after (fresh competitions)
    hh = []
    for g, lab in G.items():
        hh.append(dict(gender=lab, stage='Found: The Hundred', competition='The Hundred', r=HH[f'hundred_{g}']['r'], r2=HH[f'hundred_{g}']['R2'],
                       excess_slope=HH[f'hundred_{g}']['full5']['excess_slope'], p=HH[f'hundred_{g}']['full5']['shuffle_p']))
        hh.append(dict(gender=lab, stage='Confirmed: fresh data', competition='T20 Blast' if g == 'male' else 'English women’s T20', r=HH[f'fresh_{g}']['r'], r2=HH[f'fresh_{g}']['R2'],
                       excess_slope=C[g]['primary']['excess_slope'], p=C[g]['primary']['holm_p']))
    pd.DataFrame(hh).round(5).to_csv(f'{D}/hot_hand.csv', index=False)

    # what moves momentum: share of swings by kind of ball
    pd.DataFrame([dict(gender=lab, innings=inn, outcome=c, share=v['share'][c], mean_gap_sd=v['mean_gap'][c], frequency=v['freq'][c])
                  for g, lab in G.items() for inn, v in MD[g]['outcomes'].items() for c in v['share']]).round(4).to_csv(f'{D}/outcome_shares.csv', index=False)

    # every feature ranked, 1st innings v chase
    pd.concat([pd.read_csv(f'{OUT}/{g}/feature_ranking_by_innings.csv').assign(gender=lab) for g, lab in G.items()]) \
        .rename(columns={'1st innings': 'shap_first_innings', 'Chase': 'shap_chase', 'rank_1st': 'rank_first_innings'}) \
        [['gender', 'feature', 'label', 'factor', 'shap_first_innings', 'shap_chase', 'rank_first_innings', 'rank_chase', 'rank_change']].round(5).to_csv(f'{D}/feature_ranking.csv', index=False)

    # one match: model v scoreboard momentum, chase win probability
    from plot_match import load
    x = load('male', '1521261')
    x[['innings', 'ball_no', 'batting_team', 'm_z', 's_z', 'wp_pre', 'is_wicket', 'runs_total']].rename(columns={'m_z': 'model_momentum', 's_z': 'scoreboard_momentum', 'wp_pre': 'chase_win_prob'}) \
        .assign(model_momentum=lambda t: t.model_momentum.where(t.ball_no >= 10), scoreboard_momentum=lambda t: t.scoreboard_momentum.where(t.ball_no >= 10),
                chase_win_prob=lambda t: t.chase_win_prob.where(t.innings == 2)).round(3).to_csv(f'{D}/match_1521261.csv', index=False)

    # model comparison (cross-validated log-loss, lower is better)
    pd.DataFrame([dict(model=m, men=MC['male']['cv'][m]['logloss'], women=MC['female']['cv'].get(m, {}).get('logloss')) for m in MC['male']['cv']]) \
        .sort_values('men').round(5).to_csv(f'{D}/models.csv', index=False)

    # men v women: difference (women − men) with 95% interval, for every estimable test
    import re
    Q = pd.read_csv(f'{OUT}/compare/rq4_comparison.csv'); keep = {'RQ1 slope, 10-ball, pooled': 'Team momentum carries forward (10 balls)', 'RQ1 slope, 5-ball, pooled': 'Team momentum carries forward (5 balls)',
        'RQ2 H2a': 'Scoreboard misleads more in chases', 'RQ2 H2b': 'Model momentum predicts the next spell better', 'RQ3 D3 fresh bowler after hot spell': 'Fresh bowler after a hot spell (SD)'}
    rows = []
    for _, r in Q[Q.test.isin(keep)].iterrows():
        lo, hi = map(float, re.findall(r'-?\d+\.\d+(?:e-?\d+)?', r.diff_ci)[:2])
        rows.append(dict(test=keep[r.test], men=r.men, women=r.women, difference=r['diff'], lo=lo, hi=hi, significant='Yes' if lo > 0 or hi < 0 else 'No'))
    T = J(f'{OUT}/compare/transfer.json')
    rows.append(dict(test="Men's model used on women (log-loss gap)", men=T['men_model_on_women_logloss'], women=T['women_own_model_logloss'], difference=T['women_own_model_logloss'] - T['men_model_on_women_logloss'], lo=None, hi=None, significant=''))
    pd.DataFrame(rows).round(5).to_csv(f'{D}/gender_diff.csv', index=False)

    # deep analysis: what new information adds, and which features work together
    from momentum_drivers import LABEL
    NEWL = {'ext_bat': 'Batter career T20', 'ext_bowl': 'Bowler career T20', 'h2h': 'Batter v this bowler', 'batvenue': 'Batter at this ground', 'batkind': 'Batter v pace/spin',
            'battype': 'Batter v bowling type', 'bowlhand': 'Bowler v left/right-handers'}
    STAT = {'runs': 'runs per ball', 'bnd': 'boundary rate', 'dot': 'dot rate', 'out': 'dismissal rate', 'n': 'balls of history', 'edge_runs': 'edge in runs'}
    def lab(f):
        if f in LABEL: return LABEL[f]
        for k, v in NEWL.items():
            if f.startswith(k + '_'): return f"{v}: {STAT.get(f[len(k) + 1:], f[len(k) + 1:])}"
        extra = {'wx_temp': 'Temperature', 'wx_humidity': 'Humidity', 'wx_dewpoint': 'Dew point', 'wx_cloud': 'Cloud cover', 'wx_wind': 'Wind speed', 'wx_rain': 'Rain',
                 'wx_dew_spread': 'Air dampness (temp minus dew point)', 'venue_season_rpb': 'Ground: runs per ball this season', 'venue_last5_rpb': 'Ground: last 5 matches',
                 'sameday_rpb': 'Earlier game today: runs per ball', 'evening_game': 'Evening game', 'inn1_rpb_vs_venue': 'Pitch today: 1st-innings scoring v usual',
                 'inn1_bnd_vs_venue': 'Pitch today: 1st-innings boundaries v usual', 'toss_bat_first': 'Toss winner chose to bat', 'batting_won_toss': 'Batting side won the toss',
                 'bowl_spin': 'Bowler is a spinner', 'bowl_left_arm': 'Bowler is left-arm', 'bat_left': 'Batter is left-handed',
                 'combo_bnd_matchup': 'Boundary hitter v expensive bowler', 'combo_out_matchup': 'Fragile batter v wicket-taker', 'combo_spin_x_lefthander': 'Spin to a left-hander',
                 'combo_spin_x_death': 'Spin at the death', 'combo_pace_x_powerplay': 'Pace in the powerplay', 'combo_set_x_death': 'Set batter at the death', 'combo_new_batter_x_spin': 'New batter v spin',
                 'combo_chase_pressure': 'Chase pressure (rate x wickets lost)', 'combo_humid_x_pace': 'Humidity x pace', 'combo_cloud_x_pace': 'Cloud x pace', 'combo_dew_x_spin': 'Evening game x spin',
                 'combo_dampair_x_spin': 'Damp air x spin', 'combo_form_x_quality': 'In form x good batter', 'combo_prev_boundary': 'Last ball was a boundary',
                 'combo_prev_bnd_same_bowler': 'Last ball a boundary, same bowler', 'combo_pitch_x_batter': 'Good pitch today x good batter'}
        return extra.get(f, f)
    L = J(f'{OUT}/compare/feature_lab.json'); ab = []
    for g, labg in G.items():
        for name, v in L[g]['ablation'].items():
            if name == 'locked': continue
            ab.append(dict(gender=labg, added=('Add ' + name[2:]).replace(' + ', ' + ').replace('history', 'player history').replace('conditions', 'weather and pitch').replace('combinations', 'feature combinations').replace('all three', 'all three'),
                           diff=v['diff'], lo=v['lo'], hi=v['hi'], helps='Yes' if v['hi'] < 0 else ('Worse' if v['lo'] > 0 else 'No clear change')))
    pd.DataFrame(ab).round(5).to_csv(f'{D}/lab_ablation.csv', index=False)
    IR = []
    for g, labg in G.items():
        R = pd.read_csv(f'{OUT}/{g}/feature_interactions.csv').groupby('target').head(8)
        for r in R.itertuples():
            weather = r.a.startswith('wx_') or r.b.startswith('wx_')
            IR.append(dict(id=f'{labg[0]}-{r.target[0]}-{len(IR) + 1}', gender=labg, outcome='Boundary' if r.target == 'boundary' else 'Wicket', feature_a=lab(r.a), feature_b=lab(r.b), strength=r.strength,
                           note='Weather: likely noise (weather did not improve predictions)' if weather else ''))
    pd.DataFrame(IR).round(5).to_csv(f'{D}/interactions.csv', index=False)

    # robustness: every main finding with the original model (v1) and the model with all extra features (v2)
    if os.path.exists(f'{OUT}/compare/v2_vs_v1.csv'):
        pd.read_csv(f'{OUT}/compare/v2_vs_v1.csv').round(4).to_csv(f'{D}/robustness.csv', index=False)
    for f in sorted(os.listdir(D)): print(f, os.path.getsize(f'{D}/{f}'), 'bytes')


if __name__ == '__main__':
    main()
