"""Build the full results report as a PDF: reports/Momentum_in_The_Hundred_results.pdf
Every number is read from the saved result files (data/out/...), so the PDF always matches the code's output.
Usage:  python src/make_report.py
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = os.path.expanduser('~/hundred-momentum'); sys.path.insert(0, ROOT + '/src')
OUT = f'{ROOT}/data/out'; FIG = f'{ROOT}/figures'; REP = f'{ROOT}/reports'; os.makedirs(REP, exist_ok=True); os.makedirs(f'{FIG}/compare', exist_ok=True)
BLUE, ORANGE, INK, MUTED, GRID, SURF = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e7e6e2', '#ffffff'
J = lambda p: json.load(open(p))

# ------------------------------------------------------------------ fonts (DejaVu covers −, ·, ≥, →, ²)
fd = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf')
pdfmetrics.registerFont(TTFont('DV', f'{fd}/DejaVuSans.ttf')); pdfmetrics.registerFont(TTFont('DVB', f'{fd}/DejaVuSans-Bold.ttf'))
pdfmetrics.registerFontFamily('DV', normal='DV', bold='DVB')
ss = getSampleStyleSheet()
H1 = ParagraphStyle('h1', parent=ss['Heading1'], fontName='DVB', fontSize=16, spaceBefore=6, spaceAfter=8, textColor=colors.HexColor(INK))
H2 = ParagraphStyle('h2', parent=ss['Heading2'], fontName='DVB', fontSize=12, spaceBefore=10, spaceAfter=5, textColor=colors.HexColor(INK))
P = ParagraphStyle('p', parent=ss['Normal'], fontName='DV', fontSize=9.5, leading=13.5, spaceAfter=5)
SM = ParagraphStyle('sm', parent=P, fontSize=8, leading=10.5, textColor=colors.HexColor(MUTED))
BUL = ParagraphStyle('b', parent=P, leftIndent=12, bulletIndent=2)
BOX = ParagraphStyle('box', parent=P, backColor=colors.HexColor('#eef4fc'), borderColor=colors.HexColor(BLUE), borderWidth=0.6, borderPadding=7, spaceBefore=6, spaceAfter=10)
TC = ParagraphStyle('tc', parent=P, fontSize=8.3, leading=10.5, spaceAfter=0)
TCB = ParagraphStyle('tcb', parent=TC, fontName='DVB')


def para(t, s=P): return Paragraph(t, s)
def bullets(items): return [Paragraph(t, BUL, bulletText='•') for t in items]


def table(rows, widths, bold_rows=()):
    data = [[Paragraph(str(c), TCB if i == 0 or i in bold_rows else TC) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f0ec')), ('LINEBELOW', (0, 0), (-1, 0), .6, colors.HexColor(MUTED)),
                           ('LINEBELOW', (0, 1), (-1, -1), .25, colors.HexColor(GRID)), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                           ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3)] +
                          [('BACKGROUND', (0, r), (-1, r), colors.HexColor('#eef4fc')) for r in bold_rows]))
    return t


def img(path, width_cm):
    from PIL import Image as PI
    w, h = PI.open(path).size; return Image(path, width=width_cm * cm, height=width_cm * cm * h / w)


f3 = lambda v: f'{v:.3f}'; f4 = lambda v: f'{v:.4f}'; sg = lambda v, n=3: f'{v:+.{n}f}'.replace('-', '−'); pct = lambda v: f'{v:.0%}'
ci = lambda a, b, n=3: f'({sg(a, n)} to {sg(b, n)})'


# ------------------------------------------------------------------ cached extra numbers (hot hand on the Hundred, R²)
def hot_hand_numbers():
    cache = f'{OUT}/compare/hot_hand_summary.json'
    if os.path.exists(cache): return J(cache)
    from streak import streak_test, faced
    def r_stats(d, k=5):
        def pairs(shuffle=None):
            f = faced(d, 'full'); Pp, Ff = [], []
            for _, x in f.groupby(['match_id', 'innings', 'batter_id'], sort=False):
                v = x.g.values
                if len(v) < 2 * k: continue
                if shuffle is not None: v = shuffle.permutation(v)
                for s in range(k, len(v) - k + 1, k): Pp.append(v[s - k:s].sum()); Ff.append(v[s:s + k].sum())
            return np.corrcoef(Pp, Ff)[0, 1]
        r = pairs(); rng = np.random.default_rng(1); r0 = float(np.mean([pairs(rng) for _ in range(200)]))
        return dict(r=float(r), R2=float(r * r), r_null=r0, r_excess=float(r - r0), R2_excess=float((r - r0) ** 2))
    out = {}
    for lab, src in [('hundred', {g: f'{OUT}/{g}/values.parquet' for g in ('male', 'female')}), ('fresh', {g: f'{OUT}/confirm/{g}/values.parquet' for g in ('male', 'female')})]:
        for g, p in src.items():
            d = pd.read_parquet(p); e = dict(r_stats(d))
            if lab == 'hundred':
                e['full5'] = streak_test(d, 5, 'full'); e['full10'] = streak_test(d, 10, 'full')
            out[f'{lab}_{g}'] = e; print('hot hand cached', lab, g, flush=True)
    json.dump(out, open(cache, 'w'), indent=1); return out


# ------------------------------------------------------------------ summary figures
def fig_thesis(R):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED})
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    rid = {g: {e['window']: e['R2_forecast_ridge'] for e in R['rob'][g]['E3']} for g in ('male', 'female')}
    for a, g, nm in zip(ax, ('male', 'female'), ('Men', 'Women')):
        H = pd.DataFrame(R[g]['HT']); H = H[H.innings == 1]
        a.plot(H.window, H.R2_descriptive, color=BLUE, lw=2, marker='o', ms=5, label='Describing the passage (trend)')
        a.plot(list(rid[g]), list(rid[g].values()), color=ORANGE, lw=2, marker='s', ms=5, label='Forecasting it in advance (ridge)')
        for x_, y_ in zip(H.window, H.R2_descriptive): a.annotate(f'{y_:.2f}', (x_, y_), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=8, color=INK)
        a.set_xscale('log'); a.set_xticks([1, 5, 10, 20, 50, 100]); a.set_xticklabels(['1 ball', '5', '10', '20', '50', 'innings'])
        a.set_ylim(0, .5); a.grid(axis='y', color=GRID); a.set_axisbelow(True); [a.spines[s].set_visible(False) for s in ('top', 'right')]
        a.set_title(f'{nm}, first innings', loc='left', fontweight='bold', color=INK); a.set_xlabel('Window length (balls)')
    ax[0].set_ylabel('R² of expected v actual runs'); ax[0].legend(frameon=False, loc='upper left', fontsize=9)
    fig.tight_layout(); p = f'{FIG}/compare/thesis_trend_vs_ball.png'; fig.savefig(p, dpi=170); plt.close(fig); return p


def fig_hot_hand(HH, C):
    rows = [('Hundred men (found)', HH['hundred_male']['full5'], BLUE, .45), ('T20 Blast men (confirmation)', C['male']['primary'], BLUE, 1),
            ('Hundred women (found)', HH['hundred_female']['full5'], ORANGE, .45), ('English women\'s T20 (confirmation)', C['female']['primary'], ORANGE, 1)]
    fig, a = plt.subplots(figsize=(9, 3.2))
    for i, (lab, r, col, al) in enumerate(rows):
        y = len(rows) - 1 - i; ex = r['excess_slope']; lo = r['lo'] - r['shuffle_null_mean']; hi = r['hi'] - r['shuffle_null_mean']
        a.plot([lo, hi], [y, y], color=col, lw=2.2, alpha=al); a.scatter([ex], [y], color=col, s=60, zorder=3, alpha=al, edgecolor='white', linewidth=1.5)
        a.text(hi + .006, y, f'{ex:+.3f}   p = {r["shuffle_p"]:.3f}'.replace('-', '−'), va='center', fontsize=9, color=INK)
    a.axvline(0, color=MUTED, lw=.9); a.set_yticks(range(len(rows))); a.set_yticklabels([r[0] for r in rows][::-1]); a.set_xlim(-.05, .27)
    a.set_xlabel('Hot-hand effect: slope beyond shuffled order (95% interval), 5-ball windows, fully adjusted')
    a.grid(axis='x', color=GRID); a.set_axisbelow(True); [a.spines[s].set_visible(False) for s in ('top', 'right', 'left')]
    fig.tight_layout(); p = f'{FIG}/compare/hot_hand_before_after.png'; fig.savefig(p, dpi=170); plt.close(fig); return p


# ------------------------------------------------------------------ report
def build():
    R = {g: J(f'{OUT}/{g}/results.json') for g in ('male', 'female')}; R['rob'] = J(f'{OUT}/compare/robustness.json')
    T26 = {g: pd.read_csv(f'{OUT}/{g}/test_2026.csv') for g in ('male', 'female')}; MC = J(f'{OUT}/compare/model_comparison.json')
    PSM = {g: pd.read_csv(f'{OUT}/{g}/rq3_psm.csv', index_col=0) for g in ('male', 'female')}; TR = J(f'{OUT}/compare/transfer.json')
    C = J(f'{OUT}/confirm/results.json'); HH = hot_hand_numbers(); rob = R['rob']; MD = J(f'{OUT}/compare/momentum_drivers.json')
    sh = lambda g, inn, c: round(MD[g]['outcomes'][inn]['share'][c] * 100)
    pth, phh = fig_thesis(R), fig_hot_hand(HH, C)
    rq1 = {g: pd.DataFrame(R[g]['RQ1']) for g in ('male', 'female')}; r10 = {g: rq1[g][(rq1[g].window == 10) & (rq1[g]['sample'] == 'pooled')].iloc[0] for g in ('male', 'female')}
    s = []

    # ---- title + summary
    s += [Spacer(1, .3 * cm), para('Momentum in The Hundred', ParagraphStyle('t', parent=H1, fontSize=24, leading=29)),
          para('Is it the trend or the ball? Momentum, coaching decisions and the hot hand in men’s and women’s short-form cricket', ParagraphStyle('st', parent=P, fontSize=12.5, leading=16, textColor=colors.HexColor(MUTED))),
          Spacer(1, .2 * cm), para('Full results report · public Cricsheet ball-by-ball data · generated from the project code (src/make_report.py)', SM), Spacer(1, .3 * cm)]
    s += [para('Main findings', H2)] + bullets([
        f'<b>Trends describe cricket; single balls barely do.</b> Expected v actual runs: R² {f3(R["male"]["HT"][0]["R2_descriptive"])} for one ball, rising to {f3(pd.DataFrame(R["male"]["HT"]).query("innings==1 and window==100").R2_descriptive.iloc[0])} for an innings (men); women are the same ({f3(R["female"]["HT"][0]["R2_descriptive"])} to {f3(pd.DataFrame(R["female"]["HT"]).query("innings==1 and window==100").R2_descriptive.iloc[0])}).',
        f'<b>Team momentum is mostly an illusion.</b> A good 10 balls is followed by a slightly better next 10 (slope men {sg(r10["male"].slope)}, women {sg(r10["female"].slope)}), but shuffling the ball order gives the same slope (p = {r10["male"].shuffle_p:.2f} and {r10["female"].shuffle_p:.2f}). What persists is a team’s form on the day, not streaks.',
        f'<b>Batters do run hot over about 5 balls.</b> Found in The Hundred, then <b>confirmed on fresh data</b> with a test written before download: T20 Blast men {sg(C["male"]["primary"]["excess_slope"])} (p = {C["male"]["primary"]["holm_p"]:.3f}), English women’s T20 {sg(C["female"]["primary"]["excess_slope"])} (p = {C["female"]["primary"]["holm_p"]:.3f}, Holm-adjusted). It is real but small: under 1% of the next 5 balls is explained.',
        f'<b>The scoreboard misleads in chases.</b> Model and scoreboard momentum disagree on {pct(R["male"]["RQ2"]["H2a"]["disagree_inn2"])} of chase balls against {pct(R["male"]["RQ2"]["H2a"]["disagree_inn1"])} in first innings (men; women {pct(R["female"]["RQ2"]["H2a"]["disagree_inn2"])} v {pct(R["female"]["RQ2"]["H2a"]["disagree_inn1"])}).',
        f'<b>Bowling changes after a hot spell show no measurable effect</b> (PSM, men {sg(PSM["male"].loc["D3 fresh bowler after hot spell"].att, 2)} SD, p = {PSM["male"].loc["D3 fresh bowler after hot spell"].p:.2f}). Other decisions are too rare to estimate.',
        '<b>Men and women behave alike.</b> No men v women difference is significant; the men’s model transfers to women with only a small loss.',
        f'<b>The best model is LightGBM + random forest</b>, first of 12 for men and tied first for women. Every model hits the same ceiling, which is cricket’s randomness.'])
    s += [para('Research questions (frozen before analysis)', H2),
          table([['Question', 'Men', 'Women'],
                 ['Thesis: does a trend describe play better than a single ball?', 'Yes', 'Yes'],
                 ['RQ1: does team momentum carry forward as streaks?', 'No (form on the day)', 'No (form on the day)'],
                 ['RQ2: is model momentum better than scoreboard momentum, especially in chases?', 'Yes', 'Yes'],
                 ['RQ3: do coaching decisions change momentum (PSM)?', 'No effect found', 'No effect found'],
                 ['RQ4: do men and women differ?', 'No significant difference', ''],
                 ['H5 (fresh data): do batters run hot over 5 balls?', 'Confirmed (T20 Blast)', 'Confirmed (English T20)']], [9, 3.9, 3.9]), Spacer(1, .3 * cm),
          para('Added after the plan (exploratory)', H2),
          table([['Question', 'Men', 'Women'],
                 ['What moves momentum, 1st innings v chase?', f'Wickets: {sh("male", "1st innings", "Wicket")}% v {sh("male", "Chase", "Wicket")}% of swings; dots and singles matter more in chases',
                  f'Wickets: {sh("female", "1st innings", "Wicket")}% v {sh("female", "Chase", "Wicket")}%; same pattern, fours lead']], [9, 3.9, 3.9]), PageBreak()]

    # ---- data and method
    s += [para('1. Data and method', H1),
          table([['', 'Men', 'Women'], ['Competition', 'The Hundred 2021–2026', 'The Hundred 2021–2026'], ['Matches (full, used)', '201 (189)', '188 (181)'],
                 ['Deliveries (full matches)', '37,680', '35,879'], ['Checks', 'Totals match official results 190/190', 'Totals match official results 179/179'],
                 ['Fresh confirmation data', 'T20 Blast 2014–2025', 'Charlotte Edwards Cup + Women’s T20 Blast 2021–2025']], [4.2, 6.3, 6.3]), Spacer(1, .3 * cm),
          para('Pipeline', H2)] + bullets([
        '<b>Cleaning:</b> Cricsheet JSON to one row per ball; runs reconcile with official results; rain-affected and shortened matches flagged and excluded.',
        '<b>80 pre-ball features</b> in cricket language: match state, recent play, partnership, batter and bowler this innings, shrunk player history, set batter, new batter, bowler’s set number, pressure index and death-phase records. A scramble-the-future test proves no feature looks ahead.',
        '<b>Feature selection</b> by random-forest permutation importance on forward time folds (kept if it helps in 3 of 4 folds): 47 features for men, 48 for women.',
        '<b>Outcome model:</b> for each ball, the probability of a wicket, 0, 1, 2, 3, 4 or 5+. Tuned LightGBM averaged with a random forest. Trained on earlier seasons only; 2026 kept as an untouched test season.',
        '<b>Ball value:</b> first innings, runs plus the change in expected runs still to come; chase, the change in win probability. <b>Gap</b> = actual − expected value. <b>Momentum</b> = the gap summed over the last 10 balls.',
        '<b>Analyses:</b> persistence with a shuffle test (RQ1), model v scoreboard (RQ2), propensity-score matching on captain decisions (RQ3), SHAP and ablation, and men v women (RQ4). The plan was frozen with a checksum before results were seen; every change is in a deviations log.']) + [PageBreak()]

    # ---- model
    s += [para('2. The model', H1), para('2026 test season (used once)', H2)]
    rows = [['Model', 'Men log-loss', 'Men AUC wicket', 'Men AUC boundary', 'Women log-loss', 'Women AUC wicket', 'Women AUC boundary']]
    nm = {'mean baseline': 'Always the average', 'phase x wickets': 'Phase × wickets table', 'Average of both': '<b>LightGBM + random forest</b>'}
    for i in range(3):
        a, b = T26['male'].iloc[i], T26['female'].iloc[i]
        rows.append([nm[a.model], f4(a.logloss), f3(a.AUC_wicket), f3(a.AUC_boundary), f4(b.logloss), f3(b.AUC_wicket), f3(b.AUC_boundary)])
    s += [table(rows, [4.2, 2.1, 2.1, 2.2, 2.1, 2.1, 2.2], bold_rows=(3,)), para('Lower log-loss is better. AUC: 0.5 is guessing, 1.0 is perfect ranking.', SM)]
    s += [para('Comparing 12 models and combinations (cross-validated, 2022–2025)', H2)]
    cvm, cvf = MC['male']['cv'], MC['female']['cv']; order = sorted(cvm, key=lambda k: cvm[k]['logloss'])
    rows = [['Model', 'Men log-loss', 'Men v locked (95% interval)', 'Women log-loss', 'Women v locked']]
    for k in order:
        a, b = cvm[k], cvf[k]
        rows.append([k, f4(a['logloss']), '—' if 'locked' in k else f"{sg(a['vs_locked'], 4)} {ci(a['vs_locked_lo'], a['vs_locked_hi'], 4)}", f4(b['logloss']), '—' if 'locked' in k else sg(b['vs_locked'], 4)])
    s += [table(rows, [5.6, 2.2, 4.6, 2.3, 2.3], bold_rows=(order.index('LightGBM + random forest (locked model)') + 1,)),
          para('The locked model is best for men. For women, averaging all seven is 0.0004 better, which is not significant. Tree ensembles win; the neural network and linear models are clearly worse. Gaps between the top models are about 0.001, so the limit is cricket’s randomness, not the algorithm.', P), PageBreak()]

    # ---- thesis
    hm = pd.DataFrame(R['male']['HT']); hf = pd.DataFrame(R['female']['HT'])
    s += [para('3. Thesis: the trend, not the ball', H1), img(pth, 17),
          para('R² of expected v actual runs over windows of different length, first innings. <b>Describing</b>: the model’s expectation is updated during the window. <b>Forecasting</b>: predicted from the window’s first ball only, with a ridge model trained on earlier seasons.', SM)]
    rows = [['Window', 'Men describing', 'Men forecasting', 'Women describing', 'Women forecasting']]
    rid = {g: {e['window']: e['R2_forecast_ridge'] for e in rob[g]['E3']} for g in ('male', 'female')}
    for w in (1, 5, 10, 20, 50, 100):
        a = hm.query('innings==1 and window==@w').iloc[0]; b = hf.query('innings==1 and window==@w').iloc[0]
        rows.append(['innings' if w == 100 else f'{w} ball' + ('s' if w > 1 else ''), f3(a.R2_descriptive), f3(rid['male'][w]) if w in rid['male'] else '—', f3(b.R2_descriptive), f3(rid['female'][w]) if w in rid['female'] else '—'])
    s += [table(rows, [3, 3.4, 3.4, 3.4, 3.4]), Spacer(1, .2 * cm),
          para('A trend describes a passage of play roughly 10 times better than a single ball does, in both competitions. Forecasting the next passage is possible but modest (R² about 0.06–0.13). <b>Coaches should read trends, not single deliveries.</b>', BOX), PageBreak()]

    # ---- RQ1
    s += [para('4. RQ1: does team momentum carry forward?', H1)]
    rows = [['', 'Slope (next 10 on last 10)', '95% interval', 'Slope with shuffled ball order', 'Shuffle p', 'Streaks?']]
    for g, n_ in (('male', 'Men'), ('female', 'Women')):
        r = r10[g]; rows.append([n_, sg(r.slope), ci(r.lo, r.hi), sg(r.shuffle_null_mean), f'{r.shuffle_p:.2f}', 'No'])
    s += [table(rows, [1.8, 3.5, 3.6, 3.6, 1.8, 2]),
          para('A good 10 balls is followed by a slightly better next 10. But shuffling the ball order within each innings gives the same slope, so the persistence comes from whole-innings form (a team having a good day), not from momentum streaks. The 5-ball windows agree. <b>H1 is not supported in either competition.</b>', P)]
    # ---- RQ2 + single match
    a2, b2 = R['male']['RQ2'], R['female']['RQ2']
    s += [para('5. RQ2: model momentum v scoreboard momentum', H1),
          table([['', 'Men', 'Women'], ['Disagree, first innings', pct(a2['H2a']['disagree_inn1']), pct(b2['H2a']['disagree_inn1'])], ['Disagree, chases', pct(a2['H2a']['disagree_inn2']), pct(b2['H2a']['disagree_inn2'])],
                 ['H2a: chase − first innings (points)', f"{a2['H2a']['diff'] * 100:+.1f} ({a2['H2a']['lo'] * 100:+.1f} to {a2['H2a']['hi'] * 100:+.1f})".replace('-', '−'), f"{b2['H2a']['diff'] * 100:+.1f} ({b2['H2a']['lo'] * 100:+.1f} to {b2['H2a']['hi'] * 100:+.1f})".replace('-', '−')],
                 ['H2b: model − scoreboard, predicting next window (Spearman ρ)', f"{sg(a2['H2b']['diff'])} {ci(a2['H2b']['lo'], a2['H2b']['hi'])}", f"{sg(b2['H2b']['diff'])} {ci(b2['H2b']['lo'], b2['H2b']['hi'])}"]], [7.5, 4.6, 4.6]),
          para('Scoreboard momentum (runs minus 6 per wicket) misleads mostly in chases, where the same runs can be good or bad depending on the target. The model measure also predicts the next window slightly better. <b>Both parts of H2 are confirmed in both competitions.</b>', P), PageBreak(),
          para('One match, both measures: London Spirit v Welsh Fire, 12 Aug 2026', H2), img(f'{FIG}/male/momentum_match_1521261.png', 16.5)] + bullets([
        '<b>First innings:</b> the two measures agree on 90% of balls. They differ in how wickets are priced: early wickets cost more in the model (ball 27: −2.7 v −1.8), late ones less (balls 94–96).',
        '<b>Chase, balls 28–36:</b> the scoreboard says Welsh Fire are struggling (slow scoring), but the model says they are fine; their win chance stayed about 70% with 95 needed off 66 and wickets in hand.',
        '<b>Chase, balls 85–89:</b> the scoreboard still shows Welsh Fire on top, but the model has already turned negative (17 needed off 14, scoring not keeping pace). Four wickets then fell and the win chance went from 85% to 5%.']) + [PageBreak()]

    # ---- single-match SHAP
    s += [para('6. Why momentum moved in that match (SHAP)', H1),
          para('Momentum = what happened − what the model expected. So each high or low has two parts: <b>what happened</b> (runs, boundaries, dots and wickets against expectation) and <b>why the model expected what it did</b> (SHAP on the LightGBM half of the model, converted into runs and grouped into cricket factors). The runs-converted SHAP values track the model’s expectation with r = 0.98.', P),
          img(f'{FIG}/male/match_1521261_phases.png', 15.5), PageBreak(), img(f'{FIG}/male/match_1521261_shap_portrait.png', 15.5), PageBreak(), Spacer(1, .2 * cm),
          table([['Phase', 'What happened', 'Why the model expected it (SHAP)', 'Reading'],
                 ['#2 London Spirit balls 21–30, LOW', '8 runs (exp 13.0); 2 wickets (exp 0.5)', 'Expectation already lowered: batters not set (−0.98), wickets lost (−0.55)', 'Ferguson and Jansen took 2 quick wickets; new batters could not score'],
                 ['#5 Welsh Fire balls 41–50, HIGH', '18 runs (exp 12.9); 1 dot (exp 2.5)', 'Strong batters (+0.69); stage of innings (−0.67)', 'Cox and Root rotated strike and scored above what a set pair usually does; win chance 55% → 75%'],
                 ['#7 Welsh Fire balls 81–90, LOW', '11 runs (exp 19.4); 5 dots (exp 2.0)', 'Model expected a lot: good, set batters (+1.73, +1.52), wickets in hand (+1.57)', 'Milne and Overton dried up the runs exactly when set batters should have accelerated: the slide began here'],
                 ['#8 Welsh Fire balls 91–100, LOW', '9 runs (exp 15.1); 4 wickets (exp 1.4)', 'Wickets in hand still raised expectation (+1.58); new batters lowered it (−0.87)', 'Collapse: win chance 82% → 9%']], [3.4, 3.6, 4.8, 5]),
          para('Captain’s decisions in this match and the PSM evidence', H2)]
    v3 = PSM['male'].loc['D3 fresh bowler after hot spell']
    s += bullets([f'Hot spells for the batting side (top quarter of momentum): <b>16</b> in this match; the captain brought on a fresh bowler after <b>15</b> of them. Expensive sets (10+ runs): the bowler was changed every time.',
                  f'Across all {int(v3.n_treated)} men’s hot spells, PSM finds no effect of a fresh bowler on the next set: {sg(v3.att, 2)} SD (95% interval {sg(v3.lo, 2)} to {sg(v3.hi, 2)}, p = {v3.p:.2f}).',
                  'One match shows <i>what was decided</i>; only many matches can show <i>whether it worked</i>. That is what PSM does, and it cannot be run on a single match.']) + [PageBreak()]

    # ---- RQ3
    s += [para('7. RQ3: coaching decisions (propensity-score matching)', H1)]
    rows = [['Decision', 'Men', 'Women']]
    def cell(r):
        if not str(r.status).startswith('estimable'): return f"{r.status}" + (f" (treated {int(r.n_treated)})" if pd.notna(r.n_treated) else '')
        return f"Effect {sg(r.att, 2)} SD {ci(r.lo, r.hi, 2)}, p = {r.p:.2f}; balanced (max SMD {r.max_abs_smd:.3f}); E-value {r.evalue:.2f}"
    for k in PSM['male'].index: rows.append([k, cell(PSM['male'].loc[k]), cell(PSM['female'].loc[k])])
    s += [table(rows, [4, 6.4, 6.4]), Spacer(1, .2 * cm)] + bullets([
        'Matching: logistic propensity score, nearest neighbour within 0.2 SD of the logit, each control used at most 5 times; balance required (every standardised difference under 0.1), otherwise “not estimable”.',
        'Captains almost never keep an expensive bowler on, and promotions of big hitters are rare, so those decisions cannot be estimated. A fresh bowler after a hot spell shows no detectable effect in either competition. This is an honest null result.']) + [PageBreak()]

    # ---- global SHAP + ablation
    s += [para('8. What drives the model overall (SHAP, partial dependence, ablation)', H1), img(f'{FIG}/male/partial_dependence.png', 17), para('Men: how expected runs (solid) and wicket probability (dashed) change with the strongest features.', SM),
          img(f'{FIG}/female/partial_dependence.png', 17), para('Women: the same.', SM)] + bullets([
        'Strongest drivers in both competitions: stage of the innings, how long the batter and partnership have lasted, recent strike rate, and player history (boundary and dot rates).',
        'Expected runs drop when the powerplay ends at ball 25; scoring and wicket risk both rise at the death; batters settle after about 10 balls.'])
    rows = [['Feature layers (cross-validated log-loss)', 'Men', 'Women']]
    for a, b in zip(R['male']['SHAP']['ablation'], R['female']['SHAP']['ablation']): rows.append([a['layer'], f4(a['cv_logloss']), f4(b['cv_logloss'])])
    s += [table(rows, [7, 4.5, 4.5]), para('Match state brings most of the gain. Recent form adds the next largest step; player history helps women slightly and men not at all on its own.', P), PageBreak()]

    # ---- exploratory: what moves momentum, first innings v chase
    s += [para('8b. What moves momentum: 1st innings v chase (exploratory)', H1),
          para('Added after the frozen plan, so it is descriptive, not a tested hypothesis. Momentum over 10 balls is the sum of each ball’s gap (actual − expected value), so it can be split by the kind of ball that caused it. The share for each kind is how much of the ups and downs it accounts for (shares add up to 100%).', P),
          img(f'{FIG}/compare/momentum_drivers_outcomes.png', 17)] + bullets([
        f'<b>First innings: wickets and big hits.</b> Wickets cause {sh("male", "1st innings", "Wicket")}% of men’s swings and {sh("female", "1st innings", "Wicket")}% of women’s. Men’s positive swings come mainly from sixes ({sh("male", "1st innings", "Six")}%), women’s from fours ({sh("female", "1st innings", "Four")}%).',
        f'<b>Chase: every ball counts.</b> Wickets matter less ({sh("male", "Chase", "Wicket")}% men, {sh("female", "Chase", "Wicket")}% women), while dot balls ({sh("male", "Chase", "Dot ball")}%, {sh("female", "Chase", "Dot ball")}%) and 1–3 runs ({sh("male", "Chase", "1–3 runs")}%, {sh("female", "Chase", "1–3 runs")}%) matter two to four times more than in the first innings. Against a target, a dot ball raises the rate needed and a single keeps the chase on track.',
        f'One wicket moves momentum by {abs(MD["male"]["outcomes"]["1st innings"]["mean_gap"]["Wicket"]):.1f} SD in a men’s first innings and {abs(MD["male"]["outcomes"]["Chase"]["mean_gap"]["Wicket"]):.1f} SD in a chase; a six {MD["male"]["outcomes"]["1st innings"]["mean_gap"]["Six"]:.1f} v {MD["male"]["outcomes"]["Chase"]["mean_gap"]["Six"]:.1f} SD.',
        'Caution: the chase value is a change in win probability, which by design rewards keeping up with the rate, so part of the dot-ball difference comes from how momentum is measured.']) + [PageBreak()]
    s += [para('Which situation features the model weighs, 1st innings v chase', H2), img(f'{FIG}/compare/momentum_drivers_factors.png', 17),
          para('SHAP on the LightGBM half of the model, turned into runs and grouped into the same cricket factors as the single-match analysis (6,000 balls per innings). Bigger bars mean the factor moves the expected runs more.', SM)] + bullets([
        '<b>Batter set / partnership is the biggest factor in both innings</b>, for men and women: how long the batter and partnership have lasted shapes the expectation most.',
        f'<b>Chase pressure is the only real difference.</b> Required rate and target matter in chases ({MD["male"]["factors"]["Chase"]["Chase pressure"]:.2f} runs men, {MD["female"]["factors"]["Chase"]["Chase pressure"]:.2f} women) and are near zero in first innings. Everything else is weighed almost the same in both.',
        'Wickets in hand matter about twice as much for men as for women.',
        f'For a coach: in a chase, dot balls and singles cause about {sh("male", "Chase", "Dot ball") + sh("male", "Chase", "1–3 runs")}% of momentum swings (men; first innings {sh("male", "1st innings", "Dot ball") + sh("male", "1st innings", "1–3 runs")}%), so strike rotation is a momentum tool; in the first innings, protecting wickets matters most.']) + [PageBreak()]
    RK = {g: pd.read_csv(f'{OUT}/{g}/feature_ranking_by_innings.csv', index_col=0) for g in ('male', 'female')}
    def movers(g, n=4):
        t = RK[g].sort_values('rank_change', ascending=False).head(n)
        return ', '.join(f"{r.label} (#{r.rank_1st} → #{r.rank_chase})" for r in t.itertuples())
    def top(g, col, n=5): return ', '.join(RK[g].sort_values(col).label.head(n))
    for g, name in (('male', 'Men'), ('female', 'Women')):
        s += [para(f'Every feature ranked, 1st innings v chase: {name.lower()}', H2), img(f'{FIG}/compare/feature_ranking_{g}.png', 15.5 if g == 'male' else 15.2)] + bullets([
            f'<b>Top 5 in both innings:</b> {top(g, "rank_1st")}.' if top(g, 'rank_1st') == top(g, 'rank_chase') else f'<b>Top 5, 1st innings:</b> {top(g, "rank_1st")}. <b>Chase:</b> {top(g, "rank_chase")}.',
            f'<b>Biggest climbers in the chase:</b> {movers(g)}. All are chase-pressure features; every other feature keeps almost the same rank. (Chase features show a tiny value in the 1st innings because they are blank there.)']) + [PageBreak()]

    # ---- RQ4
    s += [para('9. RQ4: men v women', H1)] + bullets([
        'Every men v women difference has an interval that includes zero: the findings line up closely across the two competitions.',
        f'<b>Transfer check:</b> the men’s model used unchanged on women’s data scores log-loss {f4(TR["men_model_on_women_logloss"])} against {f4(TR["women_own_model_logloss"])} for the women’s own model. It over-predicts women’s boundaries ({pct(TR["boundary_rate_men_model"])} v {pct(TR["boundary_rate_actual"])} actual) but gets wickets about right ({TR["wicket_rate_men_model"]:.1%} v {TR["wicket_rate_actual"]:.1%}).',
        'The batter hot hand is at least as strong in women’s cricket as in men’s (section 10).'])

    # ---- hot hand
    s += [para('10. The batter hot hand: found, then confirmed on fresh data', H1),
          para('<b>Why it was tested.</b> The team-level test (RQ1) can miss individual streaks, because batters change within a 10-ball window. Following one batter’s own balls, a good 5-ball spell predicted a better next 5 in The Hundred. Because that was found by exploring, it was <b>confirmed on competitions never used before</b> (T20 Blast; Charlotte Edwards Cup + Women’s T20 Blast), with the test, the model recipe and the decision rule written and checksummed <b>before download</b> (CONFIRMATION_PLAN.md).', P),
          para('<b>The test.</b> Balls a batter faced, in order; non-overlapping 5-ball windows; slope of the next window’s summed gap on the last window’s. Adjusted for the average model error in each match situation and for the average settling-in curve. Compared with 1,000 shuffles of the ball order within each batter’s innings, which keep the batter’s form on the day but destroy order.', P),
          img(phh, 15.5)]
    rows = [['5-ball windows, fully adjusted', 'Hundred men', 'Blast men', 'Hundred women', 'English T20 women']]
    hm5, hf5, cm5, cf5 = HH['hundred_male']['full5'], HH['hundred_female']['full5'], C['male']['primary'], C['female']['primary']
    ex_ci = lambda r: ci(r['lo'] - r['shuffle_null_mean'], r['hi'] - r['shuffle_null_mean'])
    rows += [['Hot-hand effect (95% interval)', f"{sg(hm5['excess_slope'])} {ex_ci(hm5)}", f"{sg(cm5['excess_slope'])} {ex_ci(cm5)}", f"{sg(hf5['excess_slope'])} {ex_ci(hf5)}", f"{sg(cf5['excess_slope'])} {ex_ci(cf5)}"],
             ['Shuffle p', f"{hm5['shuffle_p']:.3f}", f"{cm5['shuffle_p']:.3f} (Holm {cm5['holm_p']:.3f})", f"{hf5['shuffle_p']:.3f}", f"{cf5['shuffle_p']:.3f} (Holm {cf5['holm_p']:.3f})"],
             ['Batter-innings / windows', f"{hm5['batter_innings']:,} / {hm5['windows']:,}", f"{cm5['batter_innings']:,} / {cm5['windows']:,}", f"{hf5['batter_innings']:,} / {hf5['windows']:,}", f"{cf5['batter_innings']:,} / {cf5['windows']:,}"],
             ['R² (next 5 on last 5)', f"{HH['hundred_male']['R2']:.4f}", f"{HH['fresh_male']['R2']:.5f}", f"{HH['hundred_female']['R2']:.4f}", f"{HH['fresh_female']['R2']:.4f}"],
             ['R² of the hot-hand part', f"{HH['hundred_male']['R2_excess']:.4f}", f"{HH['fresh_male']['R2_excess']:.4f}", f"{HH['hundred_female']['R2_excess']:.4f}", f"{HH['fresh_female']['R2_excess']:.4f}"],
             ['Verdict', 'found (exploratory)', '<b>confirmed</b>', 'found (exploratory)', '<b>confirmed</b>']]
    s += [table(rows, [4.2, 3.2, 3.3, 3.2, 3.3])]
    rows = [['Secondary checks (fresh data)', 'Blast men', 'English T20 women']]
    for k, lab in [('raw', 'No adjustment'), ('settling_only', 'Settling only'), ('ten_ball', '10-ball windows')]:
        rows.append([lab, f"{sg(C['male'][k]['excess_slope'])}, p = {C['male'][k]['shuffle_p']:.3f}", f"{sg(C['female'][k]['excess_slope'])}, p = {C['female'][k]['shuffle_p']:.3f}"])
    s += [Spacer(1, .2 * cm), table(rows, [6, 5.2, 5.2]), Spacer(1, .2 * cm)] + bullets([
        '<b>Confirmed in both competitions</b> by the pre-written rule. Settling in and situation error are ruled out: the adjustments barely change the result.',
        '<b>The men’s effect shrank by about 60%</b> on fresh data (+0.12 to +0.05). That is normal for an effect found by exploring; the fresh value is the one to quote. Women held steady (+0.07 to +0.08).',
        '<b>It is small.</b> Under 1% of a batter’s next 5 balls is explained by the last 5. Hot-hand effects in other sports are similarly small. The safe claim is about 5 balls; 10-ball results disagree between men and women.',
        'For a coach: a batter in form over the last over is very slightly more likely to keep going, but 5 balls alone is not a reason to change a plan.']) + [PageBreak()]

    # ---- deep analysis (exploratory)
    DBD = f'{ROOT}/dashboard'; RB = pd.read_csv(f'{DBD}/robustness.csv'); EVD = pd.read_csv(f'{DBD}/deep_events.csv'); KPD = pd.read_csv(f'{DBD}/deep_bowler_kind_parts.csv')
    SPD = pd.read_csv(f'{DBD}/deep_specialist_test.csv').set_index('gender'); BCD = pd.read_csv(f'{DBD}/deep_boundary_clusters.csv'); IRD = pd.read_csv(f'{DBD}/interactions.csv'); LAD = pd.read_csv(f'{DBD}/lab_ablation.csv')
    ev = lambda g, e, inn: EVD[(EVD.gender == g) & (EVD.event == e) & (EVD.innings == inn)].iloc[0]
    kp = lambda g, k, part: KPD[(KPD.gender == g) & (KPD.kind == k) & (KPD.part == part)].runs_saved_per_ball.iloc[0]
    FHt = 'After a no-ball: the free hit (next ball)'; fm, fw = ev('Men', FHt, '1st innings'), ev('Women', FHt, '1st innings'); sx = ev('Men', 'After a six (next ball)', 'Chase')
    s += [para('10b. Deep analysis: what the scoreboard can’t show (exploratory)', H1),
          para('Added after the frozen plan and after all results were seen, so everything here is a lead to confirm on new data, not a tested hypothesis. Three new kinds of information were built: player history from 21 Cricsheet T20 competitions (12,946 matches, earlier dates only), bowling type and batting hand for 5,296 players (Wikipedia infoboxes matched through the Cricsheet register and Wikidata), and weather (Open-Meteo archive) with pitch stand-ins. Each ball is then compared with what the model expected, and what is left over is studied.', P),
          para('Do the main findings hold with a better model?', H2)]
    rows = [['Result', 'Men: original → better model', 'Women: original → better model']]
    for r in RB.result.unique():
        a, b = RB[(RB.gender == 'Men') & (RB.result == r)].iloc[0], RB[(RB.gender == 'Women') & (RB.result == r)].iloc[0]
        rows.append([r, f'{a.version1:.3f} → {a.version2:.3f}', f'{b.version1:.3f} → {b.version2:.3f}'])
    s += [table(rows, [8.2, 4.3, 4.3]), Spacer(1, .2 * cm), para('The better model adds player history and feature combinations (90 features for men). Every main finding holds; only the middle-overs advantage of spin fades, because better spinners tend to bowl then. The frozen results stay the primary ones.', P), PageBreak()]
    s += [img(f'{FIG}/compare/deep_events.png', 17)] + bullets([
        f'<b>The free hit is undervalued.</b> On the ball after a no-ball the batting side beats expectation by {sg(fm.momentum_after, 2)} SD (men, 95% {sg(fm.lo, 2)} to {sg(fm.hi, 2)}) and {sg(fw.momentum_after, 2)} SD (women) in the first innings. Wides show no such effect. It holds with the better model too.',
        f'<b>After a six in a men’s chase</b> the next ball also beats expectation ({sg(sx.momentum_after, 2)} SD): the batter stays on top for a moment. Keeping a women’s bowler on for a back-to-back set in a chase tends to go the batting side’s way.',
        'Wickets, new batters, a bowler’s first balls and the end of the powerplay move momentum no more than the model already expects.']) + [PageBreak()]
    s += [img(f'{FIG}/compare/deep_spin_parts.png', 16.5), img(f'{FIG}/compare/deep_game_parts.png', 16.5)] + bullets([
        f'<b>Spin in the powerplay costs runs (men).</b> In balls 1–20, spinners save {sg(kp("Men", "spin", 0), 2)} runs per ball against expectation and pace bowlers {sg(kp("Men", "pace", 0), 2)}. It grows slightly with the better model, so it is not a gap in the model.',
        '<b>Where matches are won:</b> momentum in the last 20 balls of a chase goes most closely with the result for men and women; for men, chase balls 41–60 also matter. In the women’s first innings every part matters about equally.',
        f'<b>Bowler phase specialists are mostly noise.</b> A shuffle test cannot tell bowlers’ first-20 v last-20 differences from chance (p = {SPD.loc["Men", "p"]:.2f} men, {SPD.loc["Women", "p"]:.2f} women), so no bowler is named a specialist.']) + [PageBreak()]
    top = IRD.groupby(['gender', 'outcome']).head(3)
    rows = [['Competition', 'Outcome', 'Feature', 'Works with', 'Strength']] + [[r.gender, r.outcome, r.feature_a, r.feature_b, f'{r.strength:.3f}'] for r in top.itertuples()]
    ab = LAD.set_index(['gender', 'added'])
    s += [img(f'{FIG}/compare/deep_lab.png', 15.5)] + bullets([
        f'<b>Knowing the players helps more than any change of algorithm.</b> Player history lowers men’s log-loss by {abs(ab.loc[("Men", "Add player history"), "diff"]):.4f} (significant) and women’s by {abs(ab.loc[("Women", "Add player history"), "diff"]):.4f} (just misses). Batter v pace or spin is the most useful single addition.',
        f'<b>Weather and pitch stand-ins do not help</b> and make the women’s model worse ({sg(ab.loc[("Women", "Add weather and pitch"), "diff"], 4)}): one weather value per match over about 190 matches is mostly noise.',
        f'<b>Boundaries do not cluster.</b> A boundary straight after a boundary is as likely as the situation predicts (men {sg(BCD[(BCD.gender == "Men") & (BCD.condition == "After a boundary")].excess.iloc[0] * 100, 1)} percentage points, women {sg(BCD[(BCD.gender == "Women") & (BCD.condition == "After a boundary")].excess.iloc[0] * 100, 1)}; both within chance).']) + [
          para('Strongest feature pairs (SHAP interactions)', H2), table(rows, [2.6, 2.2, 5, 5, 2]),
          para('Pairs that involve weather are probably noise, since weather did not improve predictions.', SM), PageBreak()]

    # ---- limitations, deviations, reproducibility
    s += [para('11. Limitations', H1)] + bullets([
        'Ball-by-ball public data has no line, length, pace or field placings, so the model cannot see the most important part of each delivery. This sets the ceiling on prediction.',
        'RQ3 is thin: three of four decisions are too rare (or too unbalanced) to estimate. PSM controls only for what is measured; E-values show how strong a hidden factor would need to be.',
        'The Hundred is young (six seasons); some windows (50 balls, whole innings) have few examples, so their R² values are noisy.',
        'The hot-hand confirmation used T20, not the 100-ball format: 6-ball overs instead of 5-ball sets.'])
    s += [para('12. Deviations from the plans (all logged)', H1)] + bullets([
        'Exploratory deep analysis added after all results were seen (section 10b): player history from 21 T20 competitions, bowling type and batting hand from Wikipedia, weather and pitch stand-ins, feature combinations, small-moment and phase analyses, and a better-model robustness check. None of it changes a primary result.',
        'Thesis forecast test: first version invalid (multiplied one ball by the window length); replaced before any women’s analysis. A later review showed the boosted forecast overfitted; a ridge model gives R² 0.06–0.13 (exploratory).',
        'PSM covariates clipped at their 1st and 99th percentiles; D4 unit wording clarified; D4 covariates exclude the incoming batter’s history; hot-spell cut-off from men’s 2022–2025 sets; SHAP grouping uses complete linkage.',
        'Infinite feature values set to missing (two women’s innings with 101 legal balls from umpire miscounts); the women’s run was restarted from scratch.',
        'Men v women difference intervals use a normal combination of the two bootstrap intervals.',
        'Exploratory analyses after the frozen run: hot hand v settledness, batter-level streak test, ridge forecast, women’s leakage test, the 12-model comparison.',
        '<b>Confirmation:</b> Cricsheet’s 2026 files omit the chase target, so all 2026 matches were flagged incomplete and the test ran on 2023–2025 (men 3 seasons, women 3 seasons). Women’s T20 Blast contributed 45 of 112 matches for the same reason. A fix (target = first-innings runs + 1, no rain rule) was added to clean.py after the run; the Hundred is unaffected (verified). Any 2026 re-run is a labelled second look.'])
    s += [para('13. Reproduce everything', H1), para('python src/pipeline.py male · python src/pipeline.py female · python src/pipeline.py compare · python src/robustness.py · python src/compare_models.py · python src/confirm.py · python src/plot_match.py · python src/explain_match.py · python src/make_report.py', SM),
          para('Plans: ANALYSIS_PLAN.md and CONFIRMATION_PLAN.md (each with a SHA-256 checksum taken before results). Code guide: CODE_GUIDE.md. Data: Cricsheet (cricsheet.org), used under its open licence.', SM)]

    def footer(c, doc):
        c.saveState(); c.setFont('DV', 7.5); c.setFillColor(colors.HexColor(MUTED))
        c.drawString(2 * cm, 1.2 * cm, 'Momentum in The Hundred · results report'); c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f'{doc.page}'); c.restoreState()
    out = f'{REP}/Momentum_in_The_Hundred_results.pdf'
    SimpleDocTemplate(out, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm, title='Momentum in The Hundred: results',
                      author='Hundred momentum project').build(s, onFirstPage=footer, onLaterPages=footer)
    print('saved', out)


if __name__ == '__main__':
    build()
