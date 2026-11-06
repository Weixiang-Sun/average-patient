"""LaTeX tables for the long paper, generated from outputs/tables/*.csv.

Writes outputs/tables/paper/*.tex; copy them into the paper repository (which
holds no code). Every number in these tables comes from a CSV produced by an
analysis script; nothing is typed by hand.
"""
import sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from mental_act.config import TABLES
from mental_act.figlong import MODELS, SHORT

OUT = TABLES / "paper"; OUT.mkdir(exist_ok=True)
TAG2 = {m[3]: m[0] for m in MODELS}
ORDER = [m[0] for m in MODELS]


def f(x, d=2, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "---"
    return f"{x:.{d}f}"


def write(name, body):
    import re
    body = re.sub(r"(\\begin\{tabular\})", r"\\begin{fitwidth}\n\1", body)
    body = body.replace("\\end{tabular}", "\\end{tabular}\n\\end{fitwidth}")
    # one line per caption, so a caption can be found by its line in the source
    body = re.sub(r"\\caption\{.*?\}\s*\\label", lambda m: " ".join(m.group(0).split("\n")[:-1]).rstrip() + "\n" + m.group(0).split("\n")[-1] if "\n" in m.group(0) else m.group(0), body, flags=re.S)
    body = re.sub(r"[ ]{2,}", " ", body)
    (OUT / f"{name}.tex").write_text(body)
    print("wrote", name)


BLg = pd.read_csv(TABLES / "separability_baselines.csv").set_index("pair").loc["GAD-7: NHIS 2022 vs 2019"].auc

CDb = pd.read_csv(TABLES / "calibration_diversity.csv").query("kind == 'base'").set_index("label")   # written by fig_L4
# ------------------------------------------------------------ Table 1 (body)
G = pd.read_csv(TABLES / "extra_generality.csv")
G = G[G.ref != "coach_vs_nhanes"].dropna(subset=["endorsed_m"])
REFS = [("phq9_nhanes", "PHQ-9", "NHANES", "41{,}714"),
        ("phq9_coach", "PHQ-9", "COACH$^{\\dagger}$", "2{,}359"),
        ("gad7_nhis", "GAD-7", "NHIS", "31{,}132"),
        ("k6", "K6", "NHIS", "82{,}339"),
        ("phq15", "PHQ-15", "C19PRC", "2{,}025"),
        ("ocir", "OCI-R", "C19PRC", "1{,}406"),
        ("itq", "ITQ", "C19PRC", "2{,}025"),
        ("pcl5c", "PCL-5", "Postpartum$^{\\dagger}$", "469"),
        ("edeq", "EDE-Q", "Mexico", "2{,}090")]
MIN_DIV_BINS = 5   # a diversity ratio is reported where at least five totals carry it
rows = []
for key, inst, samp, n in REFS:
    g = G[G.ref == key]; gd = g[g.div_bins >= MIN_DIV_BINS]; ratio = gd.div_h / gd.div_m
    rr_ = f"{ratio.min():.1f}--{ratio.max():.1f}" if len(gd) else "---"
    rows.append(f"{inst} & {samp} & {n} & {g.coverage.median():.0f} & "
                f"{rr_} & {g.auc.min():.2f}--{g.auc.max():.2f} & "
                f"{g.all_items_h.median():.0f} & {g.all_items_m.min():.0f}--{g.all_items_m.max():.0f} \\\\")
write("tab_generality", r"""\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{2.2pt}
\begin{tabular}{@{}llrrrrrr@{}}\toprule
& & & & & & \multicolumn{2}{c}{All items (\%)} \\\cmidrule(l){7-8}
Instr. & Sample & $N$ & Cov. & Div.\ ratio & AUC & Real & Models \\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}
\caption{Profile comparison on nine human references over all totals, each simulated response matched to real respondents with the same total (ranges over models). Cov.: \% of simulated responses whose total real respondents reach. Div.\ ratio: real over model diversity, where five or more totals carry it (none on the postpartum PCL-5; Table~\ref{tab:gen-ratio}). AUC: separability from matched real respondents (chance """ + f"{G.auc_null.min():.2f}--{G.auc_null.max():.2f}" + r"""). All items: \% endorsing every item. Real and Cov.: medians over models. $^{\dagger}$Clinical (COACH) and obstetric (postpartum) samples.}
\label{tab:generality}\end{table}
""")

# ------------------------------------------------ A: PHQ-9 / GAD-7 per model
CI = pd.read_csv(TABLES / "main_ci.csv"); AM = pd.read_csv(TABLES / "all_models.csv"); HB = pd.read_csv(TABLES / "human_baselines.csv")
for instr, J in (("phq9", 9), ("gad7", 7)):
    c = CI[CI.instrument == instr].set_index("model"); a = AM[AM.instrument == instr].set_index("model")
    lines = []
    for lab in ["REAL"] + ORDER:
        if lab not in c.index: continue
        r, s = c.loc[lab], a.loc[lab]
        name = "\\textbf{Real respondents}" if lab == "REAL" else SHORT[lab]
        extra = ""
        if instr == "phq9":
            extra = " & " + (f"{s.all8:.1f} & {s.all8_h:.1f}" if lab != "REAL" else "--- & ---") + " & " + (f"{s.calib:.0f} & {CDb.loc[lab].divp:.0f}" if lab != "REAL" else "--- & ---") + f" & {r.si:.0f}"
        lines.append(f"{name} & {int(s.n_band) if lab != 'REAL' else int(r.n)} & "
                     f"{r.breadth:.2f} [{r.breadth_lo:.2f}, {r.breadth_hi:.2f}] & " + ("---" if lab == "REAL" else f"{s.breadth_h:.2f}") + " & "
                     f"{r.all_items:.1f} [{r.all_items_lo:.1f}, {r.all_items_hi:.1f}] & " + ("---" if lab == "REAL" else f"{s.all_items_h:.1f}") + " & "
                     f"{r.zero:.1f} [{r.zero_lo:.1f}, {r.zero_hi:.1f}] & " + ("---" if lab == "REAL" else f"{s.zero_h:.1f}") + f" & {r.uneven:.2f} & "
                     f"{s.diversity:.2f} & " + ("---" if lab == "REAL" else f"{s.diversity_h:.2f}") + " & " + ("1.00" if lab == "REAL" else f"{s.rho:.2f}") + " & "
                     + ("---" if lab == "REAL" else f"{s.auc:.2f}") + extra + " \\\\")
        if lab == "REAL": lines.append("\\midrule")
    band = "15--24" if instr == "phq9" else "12--19"
    hdr = "Model & $n$ & Breadth & Real$_{=}$ & All items (\\%) & Real$_{=}$ & Not at all (\\%) & Real$_{=}$ & Unev. & Div. & Real$_{=}$ & $\\rho$ & AUC" + \
          (" & All 1--8 & Real$_{=}$ & Calib. & Div.\\,\\%$_{5\\text{--}27}$ & SI" if instr == "phq9" else "")
    cols = "lrrrrrrrrrrrr" + ("rrrrr" if instr == "phq9" else "")
    write(f"tab_{instr}_full", f"""\\begin{{table*}}[t]\\centering\\footnotesize\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{@{{}}{cols}@{{}}}}\\toprule
{hdr} \\\\\\midrule
""" + "\n".join(lines) + f"""
\\bottomrule\\end{{tabular}}
\\caption{{{instr.upper().replace('PHQ9', 'PHQ-9').replace('GAD7', 'GAD-7')}, responses with total {band}, with 95\\% intervals. Breadth: symptoms endorsed (of {J}). Unev.: within-person s.d. Div.: diversity, over the totals with at least eight responses. $\\rho$: Spearman correlation with the human item ordering. Real$_{{=}}$: the preceding measure among real respondents at the model's exact totals. AUC: separability from total-matched real respondents (""" + (f"two real surveys: {HB.auc.max():.2f}" if instr == "phq9" else "two NHIS years: " + f"{BLg:.2f}" + ", Table~\\ref{tab:sep-checks}") + r""").""" +
          (" All 1--8: \\% endorsing every one of items 1--8. Calib.: \\% of all trials whose total falls in the instructed band. Div.\\,\\%$_{5\\text{--}27}$: diversity over totals 5--27 as a percentage of that of real respondents with the same totals (Figure~\\ref{fig:interventions}e). SI: \\% endorsing item 9 at totals 15--24." if instr == "phq9" else "")
          + f"""}}\\label{{tab:{instr}-full}}\\end{{table*}}
""")

# ------------------------------------------- A: generality per model (AUC, variety)
for metric, cap, fmt in (("auc", "Separability (AUC) from total-matched real respondents", "{:.2f}"),
                         ("ratio", "Real-to-model diversity ratio at matched totals", "{:.1f}")):
    G["ratio"] = G.div_h / G.div_m
    head = " & ".join(i for _, i, _, _ in REFS)
    sub = " & ".join(s for _, _, s, _ in REFS)
    lines = []
    for lab in ORDER:
        vals = []
        for key, *_ in REFS:
            q = G[(G.ref == key) & (G.model == lab)]
            ok = len(q) and not np.isnan(q[metric].iloc[0])
            vals.append((fmt.format(q[metric].iloc[0]) + (f" ({int(q.div_bins.iloc[0])})" if metric == "ratio" else "")) if ok else "---")
        lines.append(f"{SHORT[lab]} & " + " & ".join(vals) + " \\\\")
    write(f"tab_generality_{metric}", f"""\\begin{{table*}}[t]\\centering\\footnotesize\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{@{{}}l{'r' * len(REFS)}@{{}}}}\\toprule
& {head} \\\\
& {sub} \\\\\\midrule
""" + "\n".join(lines) + f"""
\\bottomrule\\end{{tabular}}
\\caption{{{cap}, per model and reference{', with the number of totals behind each estimate in parentheses' if metric == 'ratio' else ''}. $^{{\\dagger}}$Clinical.}}
\\label{{tab:gen-{metric}}}\\end{{table*}}
""")

# ------------------------------------------------------------ A: PCL-5 triangle
TC = pd.read_csv(TABLES / "pcl5_triangle_ci.csv").set_index("group")
TR = pd.read_csv(TABLES / "pcl5_triangle.csv")
PC = pd.read_csv(TABLES / "pcl5_profile_ci.csv").set_index("group")   # genuine respondents matched to each group's totals
def prof(g):
    q = PC.loc[g]
    return (f"{q.breadth:.1f} & {q.breadth_gen:.1f} & {q.naa:.1f} & {q.naa_gen:.1f} & {q.naa_gen - q.naa:.1f} & "
            f"{q.naa_ratio:.2f} [{q.naa_lo:.2f}, {q.naa_hi:.2f}]")
pf = lambda v: "$<$0.01" if v < 0.01 else f"{v:.2f}"
SC = pd.read_csv(TABLES / "pcl5_scales.csv").set_index("group")
def orc(g):
    k = g if g in SC.index else TAG2.get(g.replace("LLM: ", ""), "")
    if k not in SC.index: return "---"
    q = SC.loc[k]
    return "0 (none)" if q.none_zero else f"{q['or']:.2f} [{q.or_lo:.2f}, {q.or_hi:.2f}]"
lines = []
hum = TR[TR.source == "human"].set_index("group")
r = hum.loc["human genuine"]
lines.append(f"Genuinely injured & {int(r.n)} & {r.breadth:.1f} & --- & {r.zero:.1f} & --- & --- & --- & --- & {r.pri_flag:.1f} & --- & --- & --- & --- \\\\")
for g, lab in (("human feigner", "People told to fake"), ("human honest", "Uninjured, honest")):
    r, c = hum.loc[g], TC.loc[g]
    lines.append(f"{lab} & {int(r.n)} & {prof(g)} & {orc(g)} & {r.pri_flag:.1f} & {c.auc:.2f} [{c.sub_lo:.2f}, {c.sub_hi:.2f}] & {pf(c.p)} & --- & --- \\\\")
lines.append("\\midrule")
for lab in ORDER:
    tag = [m[3] for m in MODELS if m[0] == lab][0]
    p = TR[(TR.model == tag) & (TR.arm == "patient")]
    if not len(p): continue
    p = p.iloc[0]; fe = TR[(TR.model == tag) & (TR.arm == "feigner")].iloc[0]
    g = f"LLM: {tag}"; c = TC.loc[g]
    lines.append(f"{SHORT[lab]} & {int(p.n)} & {prof(g)} & {orc(g)} & {p.pri_flag:.1f} & "
                 f"{c.auc:.2f} [{c.sub_lo:.2f}, {c.sub_hi:.2f}] & {pf(c.p)} & "
                 f"${fe.total - p.total:+.1f}$ & ${fe.top - p.top:+.1f}$ \\\\")
hq = PC.loc["human feigner"]
TL = TC[TC.index.str.startswith("LLM")]
assert ((TL.p < 0.05) == (TL.p_holm < 0.05)).all(), "Holm changes a model's significance: say so in the caption"
write("tab_pcl5", r"""\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}lrrrrrrrrrrrrr@{}}\toprule
& & \multicolumn{2}{c}{Breadth} & \multicolumn{5}{c}{``Not at all''} & & & & & \\
\cmidrule(lr){3-4}\cmidrule(lr){5-9}
Group & $n$ & Group & Gen.$_{=}$ & Group & Gen.$_{=}$ & Diff. & Ratio [95\% int.] & OR [95\% int.] & PRI & AUC [sub.] & $p$ & $\Delta$total & $\Delta$extr. \\\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}
\caption{PCL-5, OSF feigning study; model rows play a genuinely injured student. Gen.$_{=}$: genuinely injured respondents matched to the group's totals; Diff.: Gen.$_{=}$ minus group, in points; Ratio: group over Gen.$_{=}$, with 95\% bootstrap interval. OR: odds ratio of answering ``not at all'' relative to genuinely injured respondents with the same total (logistic model with item effects and a spline in the total, fit on the genuinely injured; 95\% bootstrap interval); every model's is below the feigners' after Holm adjustment over the 17 models. PRI: \% flagged by the rare-items over-reporting index ($\geq3$). AUC: separability from the genuinely injured group, with the 2.5th--97.5th percentiles over 80\% subsamples and permutation $p$ (every model with $p{<}0.05$ remains so after Holm adjustment over the 17 models). $\Delta$total, $\Delta$extr.: feigning minus patient role in total and in \% ``extremely'' (people: """ + f"${hq.dtotal:+.1f}$ [${hq.dtotal_lo:.1f}$, ${hq.dtotal_hi:.1f}$] and ${hq.dextreme:+.1f}$ [${hq.dext_lo:.1f}$, ${hq.dext_hi:.1f}$]" + r""").}
\label{tab:pcl5}\end{table*}
""")

# ------------------------------------------------------------ A: PCL-5 within the feigners' range of totals
from mental_act.feign import FEIGNER_RANGE
mm = lambda v, d: f"{v:.{d}f}".replace("-", "$-$")
iv = lambda v, lo, hi, d=1: f"{mm(v, d)} [{mm(lo, d)}, {mm(hi, d)}]"
# models with fewer than 10 responses in the range are left out (as for the in-range ratio);
# Holm over the models that remain
from statsmodels.stats.multitest import multipletests
PCm = PC[PC.index.str.startswith("LLM") & (PC.n_in_range >= 10)].copy()
PCm["p_holm10"] = multipletests(PCm.abs_minus_feigner_p, method="holm")[1]
def rrow(lab, q, p, k):
    if q.n_in_range < 10: return f"{lab} & {int(q.n_in_range)} & --- & --- & --- & --- & --- & --- \\\\"
    o = SC.loc[k]; orr = f"{o.or_range:.3f}" if o.or_range < 0.01 else f"{o.or_range:.2f}"
    return (f"{lab} & {int(q.n_in_range)} & {iv(q.naa_abs_in_range, q.naa_abs_in_range_lo, q.naa_abs_in_range_hi)} & {q.naa_ratio_in_range:.2f} & {orr} & "
            f"{iv(q.breadth_gap_in_range, q.gap_in_range_lo, q.gap_in_range_hi, 2)} & {p} \\\\")
lines = [rrow("People told to fake", PC.loc["human feigner"], "--- & ---", "human feigner"), "\\midrule"]
for lab in ORDER:
    g = f"LLM: {[m[3] for m in MODELS if m[0] == lab][0]}"
    if g in PC.index: lines.append(rrow(SHORT[lab], PC.loc[g], f"{pf(PC.loc[g].abs_minus_feigner_p)} & {pf(PCm.p_holm10.get(g, np.nan))}", lab))
write("tab_pcl5_range", r"""\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrrrrrr@{}}\toprule
Group & $n$ & N.a.a.\ diff. & Ratio & OR & Breadth diff. & $p$ & $p_{\text{Holm}}$ \\\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}
\caption{PCL-5 responses with totals in the feigners' 10th--90th percentile (""" + f"{FEIGNER_RANGE[0]:.0f}--{FEIGNER_RANGE[1]:.0f}" + r"""), against genuinely injured respondents matched to these totals, with 95\% bootstrap intervals. N.a.a.\ diff.: matched genuine minus group, points of ``not at all''. Ratio: as in Table~\ref{tab:pcl5}. OR: odds ratio as in Table~\ref{tab:pcl5}, with the responses reweighted to the feigners' distribution of totals (bins of five points); every model's is below the feigners' after Holm adjustment. Breadth diff.: group minus matched genuine. $p$: one-sided bootstrap test that the model's ``not at all'' difference exceeds the feigners'; $p_{\text{Holm}}$: Holm-adjusted over the models shown; ---, fewer than 10 responses in the range.}
\label{tab:pcl5-range}\end{table}
""")

# ------------------------------------------------------------ A: mechanism
LP = pd.read_csv(TABLES / "logprob_mechanism.csv")
SP = pd.read_csv(TABLES / "logprob_sampled_profiles.csv")
AN = pd.read_csv(TABLES / "anchoring.csv").set_index("model")
HP = pd.read_csv(TABLES / "logprob_human_profiles.csv")
WF = pd.read_csv(TABLES / "wholeform_items.csv").set_index("model"); WFR = pd.read_csv(TABLES / "wholeform_items_real.csv")
SEV = ["mild", "moderate", "moderately_severe", "severe"]
lines = []
hl = LP[(LP.frame == "label") & LP.severity.isin(SEV)]
h_ent = hl.groupby("severity").ent_human.first().mean()
WFRb = WFR.set_index("band")
for lab in ORDER:
    tag = [m[3] for m in MODELS if m[0] == lab][0]
    q = LP[(LP.model == tag) & LP.severity.isin(SEV)]
    anc = f"${AN.loc[lab].r_first_rest:.2f}$" if lab in AN.index else "---"
    w = WF.loc[lab]; wf = f"{w.H:.2f} & {w.Ht:.2f} & {w.P0_moderate:.3f} & {w.P0t_moderate:.3f} & {100 * w.lock:.0f} & {100 * w.lockt:.0f} & {w.distinct:.1f} & {w.distinctt:.1f} & {anc}"
    if not len(q):
        lines.append(f"{SHORT[lab]} & {wf} & --- & --- & --- & --- & --- \\\\"); continue
    ql, qi = q[q.frame == "label"], q[q.frame == "impairment"]
    p0 = ql[ql.severity == "moderate"].p0_model.iloc[0]
    dist = SP[SP.model == tag].distinct.mean() * 100 if tag in set(SP.model) else np.nan
    lines.append(f"{SHORT[lab]} & {wf} & {ql.ent_model.mean():.2f} & {p0:.3f} & {100 * ql.label_lock.mean():.0f} & "
                 f"{100 * qi.label_lock.mean():.0f} & {f(dist, 1)} \\\\")
rd = f"{WFR.indep_distinct.min():.0f}--{WFR.indep_distinct.max():.0f}"
write("tab_mechanism", r"""\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrrrrrrrrrrrrr@{}}\toprule
& \multicolumn{9}{c}{Whole forms (""" + f"{len(WF)}" + r""" models)} & \multicolumn{5}{c}{Log-probabilities (""" + f"{LP.model.nunique()}" + r""" models)} \\
\cmidrule(lr){2-10}\cmidrule(l){11-15}
Model & $H$ & Real$_{=}$ & $P_0$ & Real$_{=}$ & Lock & Real$_{=}$ & Dist. & Real$_{=}$ & First $r$ & $H$ & $P_0$ & Lock$_L$ & Lock$_I$ & Dist. \\\midrule
""" + f"Real (bands) & {WFR.H_real50.mean():.2f} & & {WFRb.P0_real['moderate']:.3f} & & & & {rd} & & --- & {h_ent:.2f} & {hl[hl.severity == 'moderate'].p0_human.iloc[0]:.3f} & {100 * HP.label_lock.mean():.0f} & --- & {rd} \\\\\n\\midrule\n"
      + "\n".join(lines) + r"""
\bottomrule\end{tabular}
\caption{Per-item answer distributions, estimated from the answers across the 50 whole-form responses of each severity cell (all models) and from log-probabilities. Real$_{=}$: real respondents with the same totals as the model's forms in each cell (one per form, 200 draws). Real (bands): 50 real respondents per score band (whole forms) or the whole band (log-probabilities). $H$: entropy in bits, mean over items and the four symptomatic severities. $P_0$: share or probability of ``not at all'' at moderate severity. Dist.: \% distinct among 400 forms sampled item by item, mean over severities; for real respondents, sampled the same way from their answers in each band (range over bands). First $r$: correlation of the first answer with the rest of the form, after removing severity and item means. Lock: \% of items whose most common (whole forms) or most likely (log-probabilities) answer equals the severity level (mild 1, moderate 2, severe 3; moderately severe has no single level and is left out), label (L) or impairment (I) framing.}
\label{tab:mechanism}\end{table*}
""")

# ------------------------------------------------------------ A: interventions
FXA = pd.read_csv(TABLES / "fixes.csv")
FXR = FXA[FXA.cond == "-"].iloc[0]; FXA = FXA[FXA.cond != "-"]
NAME = {m[3]: SHORT[m[0]] for m in MODELS if m[3] in set(FXA.model)}   # every model with intervention runs
COND = {"label": "One-line", "hetero": "+ ``people differ''", "biography": "+ biography",
        "numeric": "Total, no label", "fewshot": "+ 5 real sheets"}
FX = FXA[FXA.cond.isin(list(COND))]       # the intervention table; published personas get their own
lines = []
for tag, nm in NAME.items():
    for i, (c, cn) in enumerate(COND.items()):
        r = FX[(FX.model == tag) & (FX.cond == c)].iloc[0]
        lines.append(f"{nm if i == 0 else ''} & {cn} & {r.calib:.0f} & {r.diversity:.2f} & {r.diversity_h:.2f} & {f(r.zero18, 1)} & "
                     f"{f(r.auc)} & {r.si_modsev:.0f} \\\\")
    lines.append("\\midrule")
write("tab_fixes", r"""\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{@{}llrrrrrr@{}}\toprule
Model & Condition & Calib. & Div. & Real$_{=}$ & N.a.a. & AUC & SI$_{\text{ms/s}}$ \\\midrule
""" + "\n".join(lines[:-1]) + r"""
\bottomrule\end{tabular}
\caption{Interventions (PHQ-9). Calib.: \% in the instructed band. Div.: diversity, totals 5--27; Real$_{=}$: held-out real respondents at the same totals. N.a.a.: \% ``not at all'' on items 1--8, totals 15--24 (real: """ + f"{FXR.zero18:.1f}" + r"""). AUC: separability from held-out real respondents (chance """ + f"{FX.auc_null.min():.2f}--{FX.auc_null.max():.2f}" + r"""; ---, too few responses at totals 15--24). SI$_{\text{ms/s}}$: \% endorsing item 9 among moderately severe and severe personas, all totals.}
\label{tab:fixes}\end{table}
""")

# ------------------------------------------------------------ A: published personas
PSRC = {"label": "One-line", "talkdep": "TalkDep", "cactus": "CACTUS", "cactus_cm": "CACTUS + form."}
FX = FXA
lines = []
for tag, nm in NAME.items():
    for i, (c, cn) in enumerate(PSRC.items()):
        r = FX[(FX.model == tag) & (FX.cond == c)].iloc[0]
        lines.append(f"{nm if i == 0 else ''} & {cn} & {int(r.n)} & {r.diversity:.2f} & {r.diversity_h:.2f} & {r.zero18:.1f} & {r.si_band:.1f} & {r.all9_band:.1f} & {f(r.auc)} & {f(r.auc18)} \\\\")
    lines.append("\\midrule")
write("tab_personas", r"""\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{@{}llrrrrrrrr@{}}\toprule
Model & Persona & $n$ & Div. & Real$_{=}$ & N.a.a.\ 1--8 & SI & All 9 & AUC & AUC$_{1\text{--}8}$ \\\midrule
""" + "\n".join(lines) + r"""
\textbf{Real} (held-out) & & """ + f"{int(FXR.n)} & --- & --- & {FXR.zero18:.1f} & {FXR.si_band:.1f} & {FXR.all9_band:.1f} & --- & ---" + r""" \\
\bottomrule\end{tabular}
\caption{Published simulator personas (PHQ-9), intervention protocol. One-line: the severity persona, from the intervention runs. CACTUS + form.: the CACTUS intake form with the automatic thought and cognitive distortions of the client's case formulation. $n$: responses; for Real, held-out respondents at totals 15--24. Div.: diversity, totals 5--27; Real$_{=}$: held-out real respondents at the same totals. At totals 15--24: N.a.a.\ 1--8, \% ``not at all'' on items 1--8; SI, \% endorsing item 9; All 9, \% endorsing every item; AUC, separability from held-out real respondents (chance """ + f"{FX[FX.cond.isin(list(PSRC))].auc_null.min():.2f}--{FX[FX.cond.isin(list(PSRC))].auc_null.max():.2f}" + r"""); AUC$_{1\text{--}8}$, the same without item 9. The reference is the held-out half of NHANES, so one-line values differ slightly from Table~\ref{tab:phq9-full}. TalkDep personas are each answered repeatedly, so their diversity uses pairs of different personas and a persona's responses stay in one cross-validation fold.}
\label{tab:personas}\end{table}
""")

PA = pd.read_csv(TABLES / "published_personas.csv")
PCOL = [("phq9", "one-line"), ("phq9", "talkdep"), ("phq9", "cactus"), ("phq9", "cactus_cm"), ("gad7", "one-line"), ("gad7", "talkdep"), ("gad7", "cactus"), ("gad7", "cactus_cm")]
def pcell(lab, ins, src):
    q = PA[(PA.model == lab) & (PA.instrument == ins) & (PA.persona == src)]
    return "---" if not len(q) else f"{q.auc.iloc[0]:.2f} / {q.div_ratio.iloc[0]:.1f}"
lines = [f"{SHORT[l]} & " + " & ".join(pcell(l, i, s_) for i, s_ in PCOL) + " \\\\" for l in ORDER]
write("tab_personas_all", r"""\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrrrrrrr@{}}\toprule
& \multicolumn{4}{c}{PHQ-9} & \multicolumn{4}{c}{GAD-7} \\
\cmidrule(lr){2-5}\cmidrule(l){6-9}
Model & One-line & TalkDep & CACTUS & + form. & One-line & TalkDep & CACTUS & + form. \\\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}
\caption{Published simulator personas on every model and on both questionnaires. Each cell: separability (AUC) from real respondents with the same totals (NHANES for the PHQ-9, NHIS for the GAD-7; totals $\geq$5) / real-to-model diversity ratio. One-line: the severity persona of the main runs. + form.: CACTUS with the automatic thought and cognitive distortions of the client's case formulation. All totals of at least 5 are used, so one-line values differ slightly from Tables~\ref{tab:phq9-full}, \ref{tab:gad7-full} and~\ref{tab:personas} (totals 15--24). TalkDep personas are each answered repeatedly, so their diversity uses pairs of different personas and a persona's responses stay in one cross-validation fold.}
\label{tab:personas-all}\end{table*}
""")

TD = pd.read_csv(TABLES / "published_personas_talkdep.csv")
COLS = ["bdi", "total_mean", "total_sd", "distinct", "breadth", "breadth_real", "at_one", "at_one_real", "key_endorsed", "other_endorsed", "other_real"]
tdm = TD.groupby("persona")[COLS].median().sort_values("bdi")
lines = [f"{p} & {int(r.bdi)} & {r.total_mean:.1f} & {r.total_sd:.1f} & {r.distinct:.0f} & {r.breadth:.1f} & {r.breadth_real:.1f} & "
         f"{r.at_one:.0f} & {r.at_one_real:.0f} & {r.key_endorsed:.0f} & {r.other_endorsed:.0f} & {r.other_real:.0f} \\\\" for p, r in tdm.iterrows()]
write("tab_talkdep", r"""\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrrrrrrrrr@{}}\toprule
& & & & & \multicolumn{2}{c}{Breadth} & \multicolumn{2}{c}{Several days} & Key & \multicolumn{2}{c}{Other} \\
\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){11-12}
Persona & BDI-II & PHQ-9 & s.d. & Distinct & Model & Real & Model & Real & Model & Model & Real \\\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}
\caption{TalkDep personas, median over """ + f"{TD.model.nunique()}" + r""" models. BDI-II: published score. PHQ-9: mean total over 30 runs, with its s.d. Distinct: \% distinct answer patterns. Breadth: symptoms endorsed. Several days: \% of endorsed answers at ``several days''. Key, Other: \% endorsing the items matching the persona's key symptoms, and the rest. Real: real respondents at the same totals.}
\label{tab:talkdep}\end{table*}
""")

GH = pd.read_csv(TABLES / "grm_heldout.csv").set_index("source")
GN = {"held-out 2013-2023": "Real, 2013--2023 (held out)", "GRM fit 2005-2012": "GRM fit on 2005--2012", "real 2005-2012": "Real, 2005--2012"}
lines = [f"{GN[k]} & " + f"{int(r.n):,}".replace(",", "{,}") + f" & {r.breadth:.2f} & {r.all9:.1f} & {r.zero:.1f} & {r.si:.1f} & {r.diversity:.2f} & {f(r.auc)} \\\\"
         for k, r in GH.iterrows()]
write("tab_grm_heldout", r"""\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{@{}lrrrrrrr@{}}\toprule
Source & $n$ & Breadth & All 9 & N.a.a. & SI & Div. & AUC \\\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}
\caption{Profiles sampled from a GRM fit to NHANES 2005--2012 (Appendix~\ref{app:prompts}), selected to match the total distribution of the held-out 2013--2023 respondents (totals $\geq$5); 2005--2012 respondents are reweighted to it. Div.: diversity. AUC: separability from the held-out respondents.}
\label{tab:grm-heldout}\end{table}
""")

# ------------------------------------------------------------ A: reasoning effort
MSH = {"Gemini 2.5 Flash": "Gemini-2.5-F", "Gemini 2.5 Pro": "Gemini-2.5-Pro"}
RE0 = pd.read_csv(TABLES / "reasoning_effort.csv").set_index(["model", "effort"])
def eff_label(r):   # $^*$ main run (reasoning off), $^{\dagger}$ API default
    m = RE0.loc[(r.model, r.effort)]
    return r.effort + ("$^*$" if m.main else "") + ("$^{\\dagger}$" if m.api_default else "")
def eff_rows(T, cells):
    out, prev = [], None
    for _, r in T.iterrows():
        if prev is not None and r.model != prev: out.append("\\midrule")
        out.append(f"{MSH.get(r.model, r.model) if r.model != prev else ''} & {eff_label(r)} & " + " & ".join(cells(r)) + " \\\\")
        prev = r.model
    return out
sg = lambda v, d=2: "---" if pd.isna(v) else (f"${v:.{d}f}$" if v < 0 else f"{v:.{d}f}")
RE = pd.read_csv(TABLES / "reasoning_effort.csv")
lines = eff_rows(RE, lambda r: [f"{r.tokens:,.0f}" + ("$^\\ddagger$" if r.model == "Kimi-K2.5" else ""), f"{r.calib:.0f}", sg(r.anchoring), f"{r.all_items:.0f}", f(r.diversity), f(r.auc),
                                f(r.gad7_all_items, 0), f(r.gad7_diversity), f(r.gad7_auc),
                                f(r.pcl5_naa_ratio), f"{r.pcl5_auc:.2f}" + ("$^\\S$" if r.pcl5_p >= 0.05 else ""), sg(r.pcl5_dtotal, 1),
                                f(r.ds_auc), f(r.ds_fpr, 1)])
write("tab_reasoning", r"""\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llrrrrrrrrrrrrrr@{}}\toprule
& & & \multicolumn{5}{c}{PHQ-9} & \multicolumn{3}{c}{GAD-7} & \multicolumn{3}{c}{PCL-5 roles} & \multicolumn{2}{c}{Screening} \\
\cmidrule(lr){4-8}\cmidrule(lr){9-11}\cmidrule(lr){12-14}\cmidrule(l){15-16}
Model & Effort & Tokens & Calib. & First $r$ & All 9 & Div. & AUC & All 7 & Div. & AUC & Ratio & AUC & $\Delta$total & AUC & FPR \\\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}
\caption{Reasoning effort; $^*$main run (reasoning off, or at the minimum for Gemini 2.5 Pro), $^{\dagger}$API default; off: no thinking; min: a thinking budget of 128 tokens. Tokens: mean reasoning tokens per PHQ-9 form ($^\ddagger$generated tokens; Kimi-K2.5 does not itemise reasoning). First $r$: as in Table~\ref{tab:mechanism}. PHQ-9 (totals 15--24) and GAD-7 (12--19) measures as in Tables~\ref{tab:phq9-full} and~\ref{tab:gad7-full}. PCL-5 roles: ``not at all'' ratio and separability of the genuine-patient role against the genuinely injured students""" + (r" ($^\S$permutation $p\geq0.05$)" if (RE.pcl5_p >= 0.05).any() else "") + r""", and the change in total from the patient to the feigning role, as in Table~\ref{tab:pcl5}. Screening: model trained on the PHQ-9 responses, as in Table~\ref{tab:downstream}.}
\label{tab:reasoning}\end{table*}
""")

RR = pd.read_csv(TABLES / "reasoning_effort_refs.csv")
RKEYS = [("phq9_nhanes", "PHQ-9"), ("phq9_coach", "COACH"), ("gad7_nhis", "GAD-7"), ("k6", "K6"), ("phq15", "PHQ-15"),
         ("ocir", "OCI-R"), ("itq", "ITQ"), ("pcl5c", "PCL-5 pp."), ("edeq", "EDE-Q")]
if "div_bins" in RR:   # as in Table 1: ratios resting on fewer than five totals are not shown
    RR.loc[RR.div_bins < MIN_DIV_BINS, "div_ratio"] = np.nan
RP = RR.pivot_table(index=["model", "effort"], columns="ref", values=["auc", "div_ratio"], sort=False, dropna=False).reset_index()
RP.columns = [c if isinstance(c, str) else (c[0] if not c[1] else f"{c[0]}|{c[1]}") for c in RP.columns]
RP = RE[["model", "effort"]].merge(RP, on=["model", "effort"], how="left")
lines = eff_rows(RP, lambda r: [f"{f(r.get(f'auc|{k}'))} / {f(r.get(f'div_ratio|{k}'), 1)}" for k, _ in RKEYS])
write("tab_reasoning_refs", r"""\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}ll""" + "r" * len(RKEYS) + r"""@{}}\toprule
Model & Effort & """ + " & ".join(n for _, n in RKEYS) + r""" \\\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}
\caption{Reasoning effort on the nine references, protocol of Table~\ref{tab:generality}; efforts marked as in Table~\ref{tab:reasoning}. Each cell: separability (AUC) / real-to-model diversity ratio, shown where at least five totals carry it. Clinical samples: COACH PHQ-9 and postpartum PCL-5.}
\label{tab:reasoning-refs}\end{table*}
""")

RG = pd.read_csv(TABLES / "reasoning_effort_gemini.csv")
lines = eff_rows(RG, lambda r: [f(r.get("fix_label_auc")), f(r.get("fix_fewshot_auc")), f(r.get("talkdep_auc")), f(r.get("cactus_auc")), f(r.get("cactus_si"), 0),
                                f"{f(r.get('iv_free_all8'), 0)} / {f(r.get('iv_free_all8_h'), 0)}", f"{f(r.get('iv_free_div'))} / {f(r.get('iv_free_div_h'))}",
                                f(r.get("iv_oracle_item_r")), f(r.get("iv_grm_item_r")), f(r.get("iv_oracle_si_agree"), 0)])
write("tab_reasoning_gemini", r"""\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llrrrrrrrrrr@{}}\toprule
& & \multicolumn{2}{c}{Interventions (AUC)} & \multicolumn{3}{c}{Published personas} & \multicolumn{5}{c}{Interviews (rater GPT-4.1)} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-7}\cmidrule(l){8-12}
Model & Effort & One-line & +5 sheets & TalkDep & CACTUS & SI & All 8 & Div. & $r$ real & $r$ sampled & SI agr. \\\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}
\caption{Reasoning effort in the Gemini 2.5 Flash and Pro experiments; efforts marked as in Table~\ref{tab:reasoning}. Interventions and published personas as in Tables~\ref{tab:fixes} and~\ref{tab:personas} (AUC at totals 15--24 against held-out real respondents; SI: \% endorsing item 9 with CACTUS personas). Interviews as in Table~\ref{tab:interview}: severity-persona interviews, all eight endorsed and diversity, each against real respondents at the same totals; $r$: correlation of given and rated answers with a real or a sampled profile; SI agr.: \% of real-profile interviews whose rated suicidal ideation matches the profile.}
\label{tab:reasoning-gemini}\end{table*}
""")

# ------------------------------------------------------------ A: scale
SC = pd.read_csv(TABLES / "scale_ci.csv")
cal = AM[AM.instrument == "phq9"].set_index("model").calib
lines = [f"{r.family} & {SHORT.get(r.model, r.model)} & {r.diversity:.2f} [{r.lo:.2f}, {r.hi:.2f}] & {cal.get(r.model, np.nan):.1f} \\\\"
         for _, r in SC.iterrows()]
write("tab_scale", r"""\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llrr@{}}\toprule
Family & Model & Diversity [2.5th, 97.5th] & Calib.\ (\%) \\\midrule
""" + "\n".join(lines) + r"""
\midrule Real & & """ + f"{AM[(AM.instrument == 'phq9') & (AM.model == 'REAL')].diversity.iloc[0]:.2f}" + r""" & --- \\
\bottomrule\end{tabular}
\caption{Diversity by model tier within families (PHQ-9, totals 15--24), with the 2.5th--97.5th percentiles over 80\% subsamples drawn without replacement.}
\label{tab:scale}\end{table}
""")

# ------------------------------------------------------------ A: tokens per form
DT = pd.read_csv(TABLES / "default_tokens.csv").set_index("model")
tk = lambda v: "---" if pd.isna(v) else f"{v:,.0f}"
lines = [f"{SHORT[l]} & {tk(DT.loc[l].generated)} & {tk(DT.loc[l].reasoning)} & {tk(DT.loc[l].generated_default)} & {tk(DT.loc[l].reasoning_default)} \\\\" for l in ORDER]
write("tab_tokens", r"""\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrr@{}}\toprule
& \multicolumn{2}{c}{Main runs} & \multicolumn{2}{c}{API default} \\
\cmidrule(lr){2-3}\cmidrule(l){4-5}
Model & Generated & Reasoning & Generated & Reasoning \\\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}
\caption{Tokens per PHQ-9 form, mean. Main runs: reasoning off, or at the minimum thinking budget (128 tokens) for Gemini 2.5 Pro. API default: for the four models whose default reasons. Generated: all tokens produced, including reasoning. Reasoning: as itemised by the API (---, not itemised or not run).}
\label{tab:tokens}\end{table}
""")

# ------------------------------------------------------------ A: interviews
IV = pd.concat([pd.read_csv(TABLES / "interview.csv"), pd.read_csv(TABLES / "interview_grm.csv")], ignore_index=True)
AGR = pd.read_csv(TABLES / "interview_agreement.csv")
def src(s):
    if s.startswith("NHANES"): return "NHANES, total $\\geq$5"
    cond, tag = s.split(": ")
    return f"{ {'free': 'severity', 'oracle': 'real profile', 'grm': 'sampled profile'}[cond]}, {ROLE.get(tag, tag)}"
ROLE = {m[3]: SHORT[m[0]] for m in MODELS}
RAT = {"gpt-4.1": "GPT-4.1", "claude-haiku-4-5": "Haiku 4.5", "-": "---"}
lines = []
for _, r in IV.iterrows():
    # Llama's suicidality answers were mostly removed by the platform filter, which
    # also removes denials that restate the question; the remaining ones are a
    # selected subset, so no SI rate is reported for Llama.
    si = "---" if "llama" in r.source else f(r.si, 1)
    agr = "---" if "llama" in r.source or pd.isna(r.si_agree) else f"{r.si_agree:.0f}"
    lines.append(f"{src(r.source)} & {RAT.get(r.rater, r.rater)} & {int(r.n)} & {r.breadth8:.2f} & {r.all8:.1f} & {f(r.all8_h, 1)} & "
                 f"{r.zero:.1f} & {r.diversity:.2f} & {f(r.diversity_h)} & {si} & {agr} & {f(r.item_r)} & {f(r.total_r)} \\\\")
write("tab_interview", r"""\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}llrrrrrrrrrrr@{}}\toprule
Patient & Rater & $n$ & Breadth (/8) & All 8 (\%) & Real$_{=}$ & Not at all (\%) & Div. & Real$_{=}$ & SI (\%) & SI agr. & $r_{\text{item}}$ & $r_{\text{total}}$ \\\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}
\caption{Interviews, PHQ-9 items 1--8 rated from the transcript. Severity: the patient is told a severity; real profile: given a real NHANES profile; sampled profile: given a GRM-sampled profile. Real$_{=}$: real NHANES respondents at the same item 1--8 totals. SI agr.: \% of interviews whose rated suicidal ideation (present or absent) matches the given profile. $r$: correlation of given and rated profile. SI is not reported for Llama-3.3 (Appendix~\ref{app:filter}). The two raters agree within one point on """ + f"{AGR.within1.min():.0f}--{AGR.within1.max():.0f}" + r"""\% of items.}
\label{tab:interview}\end{table*}
""")

num = lambda n: f"{int(n):,}".replace(",", "{,}")
GRT = pd.read_csv(TABLES / "interview_given_rated.csv")
RN = {"gpt-4.1": "GPT-4.1", "claude-haiku-4-5": "Haiku 4.5"}; CN = {"oracle": "Real profile", "grm": "Sampled profile"}
gl = [f"{CN[r.cond]} & {RN[r.rater]} & {num(r.n_items)} & {r.exact:.1f} & {r.within1:.1f} & {r.qwk:.2f} & {r.presence_sens:.1f} & {r.presence_spec:.1f} & {r.exact_present:.1f} & {r.up_given1:.1f} & " + (f"{r['shift']:+.2f}".replace("-", "$-$")) + " \\\\" for _, r in GRT.iterrows()]
write("tab_interview_agreement", r"""\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}llrrrrrrrrr@{}}\toprule
Given & Rater & Items & Exact & Within 1 & $\kappa_w$ & Present$\to$present & Absent$\to$absent & Exact if present & 1 rated higher & Shift \\\midrule
""" + "\n".join(gl) + r"""
\bottomrule\end{tabular}
\caption{Given against rated answers, PHQ-9 items 1--8, pooled over the patient models (a model never rates its own transcripts). Items: given answers compared. Exact, Within 1: \% of items rated at the given level, or within one level. $\kappa_w$: quadratically weighted kappa. Present$\to$present: \% of symptoms given as present (above 0) that are rated present; Absent$\to$absent: \% of symptoms given at 0 that are rated 0. Exact if present: \% of present symptoms rated at the given level. 1 rated higher: \% of symptoms given at ``several days'' rated higher. Shift: mean rated minus given level.}
\label{tab:interview-agreement}\end{table*}
""")

BL = pd.read_csv(TABLES / "separability_baselines.csv"); WB = pd.read_csv(TABLES / "separability_wide_bins.csv"); OH = pd.read_csv(TABLES / "separability_onehot.csv")
rr = lambda v, d=2: f"{v.min():.{d}f}--{v.max():.{d}f}"
cl = ["\\multicolumn{3}{@{}l}{\\emph{Two real samples, protocol of Table~\\ref{tab:generality}}} \\\\"]
import re as _re
yrs = lambda t: _re.sub(r"(\d{4})-(\d{4})", r"\1--\2", t)
cl += [f"{r.pair.split(': ')[0]} & {yrs(r.pair.split(': ')[1])} & {r.auc:.2f} \\\\" for _, r in BL.sort_values("pair").iterrows()]
cl += ["\\midrule", "\\multicolumn{3}{@{}l}{\\emph{Wide total bins: items / total alone / items, real item means at each total removed}} \\\\"]
RN2 = {"ocir": "OCI-R (width 2)", "pcl5c": "PCL-5 postpartum (width 3)", "edeq": "EDE-Q (width 4)"}
cl += [f"{RN2[k]} & 17 models & {rr(q.auc_items)} / {rr(q.auc_total_only)} / {rr(q.auc_residual)} \\\\" for k, q in WB.groupby("ref")]
cl += ["\\midrule", "\\multicolumn{3}{@{}l}{\\emph{Linear item scores / one-hot answer levels}} \\\\"]
for st, nm in (("phq9", "PHQ-9, totals 15--24"), ("gad7", "GAD-7, totals 12--19")):
    q = OH[OH.set == st]; cl.append(f"{nm} & 17 models & {rr(q.auc_linear)} / {rr(q.auc_onehot)} \\\\")
q = OH[OH.set == "pcl5"].iloc[0]; cl.append(f"PCL-5, OSF & people told to fake & {q.auc_linear:.2f} / {q.auc_onehot:.2f} \\\\")
q = OH[OH.set == "phq9_grm"].iloc[0]; cl.append(f"PHQ-9, NHANES 2013--2023 & GRM-sampled profiles & {q.auc_linear:.2f} / {q.auc_onehot:.2f} \\\\")
write("tab_sep_checks", r"""\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llr@{}}\toprule
Reference & Compared & AUC \\\midrule
""" + "\n".join(cl) + r"""
\bottomrule\end{tabular}
\caption{Checks on separability. Two real samples: 400 respondents of the first sample against the second, five draws. Wide bins: within the same balanced bin samples as Table~\ref{tab:generality}, the classifier given only the exact total, and given item scores minus the mean item scores of real respondents with the same total. One-hot: one weight per answer level of each item instead of one per item, same total matching and folds (feigners: 50 matchings; GRM: 600 profiles drawn at the totals of 2013--2023 respondents with totals $\geq$5).}
\label{tab:sep-checks}\end{table}
""")
IPx = pd.read_csv(TABLES / "interview_pipeline.csv")
il = []
for _, r in IPx.iterrows():
    g = lambda k, d=2: "---" if pd.isna(r.get(k)) else f"{r[k]:.{d}f}"
    il.append(f"{SHORT.get(r.model, r.model)} & {r.q_ratio:.1f} & {g('pipe_ratio', 1)} & {g('all8_free_b', 0)} & {g('all8_oracle_b', 0)} & {r.auc_free:.2f} & {g('auc_oracle')} & {g('auc_grm')} \\\\")
write("tab_interview_pipeline", r"""\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrrrrr@{}}\toprule
& \multicolumn{2}{c}{Real-to-model diversity} & \multicolumn{2}{c}{All 8 endorsed (\%)} & \multicolumn{3}{c}{AUC against real respondents} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-8}
Patient model & Questionnaire & Interview & Severity & Real profile & Severity & Real profile & Sampled profile \\\midrule
""" + "\n".join(il) + r"""
\bottomrule\end{tabular}
\caption{Interviews against real profiles passed through the same interviewer and rater (PHQ-9 items 1--8). Questionnaire: real-to-model diversity ratio at totals 15--24, as in Table~\ref{tab:phq9-full}. Interview: diversity of real-profile interviews over severity-persona interviews, both computed within the same bins of three total points (bins with at least three interviews of each kind). All 8: \% of interviews rated as endorsing all eight items, mean over the same bins. AUC: separability of the rated profiles from real NHANES respondents with the same item 1--8 totals. ---: no real-profile interviews for this model.}
\label{tab:interview-pipeline}\end{table*}
""")

HS = pd.read_csv(TABLES / "human_rating_summary.csv")
hv = lambda blk, src, m: HS[(HS.block == blk) & (HS.source == src) & (HS.metric == m)].value.iloc[0]
hi_ = lambda src, m: HS[(HS.block == "interval") & (HS.source == src) & (HS.metric == m)][["lo", "hi"]].iloc[0]
hrows = []
hsrc = sorted(s_ for s_ in HS[HS.block == "given"].source.unique() if s_.startswith("human R"))
for src, lab in [(x, x.replace("human ", "")) for x in hsrc] + [("humans pooled", "All five"), ("human median", "Median of five"), ("LLM rater", "Model rater")]:
    g = lambda m: hv("given", src, m)
    sev = f"{g('several_up'):.0f}"
    if src in ("humans pooled", "LLM rater"):
        ci = hi_(src, "several_up"); sev += f" [{ci.lo:.0f}, {ci.hi:.0f}]"
    fsrc = src if src in ("LLM rater",) or src.startswith("human R") else None
    a8 = f"{hv('free', fsrc, 'all8'):.0f}" if fsrc else ("100" if src == "humans pooled" and all(hv("free", x, "all8") == 100 for x in hsrc) else "---")
    hrows.append(f"{lab} & {g('exact'):.1f} & {g('within1'):.1f} & {g('qwk'):.2f} & {g('absent_absent'):.1f} & {g('exact_if_present'):.1f} & {sev} & " + f"{g('shift'):+.2f}".replace("-", "$-$") + f" & {a8} \\\\")
    if src == hsrc[-1]: hrows.append("\\midrule")
AGq = HS[(HS.block == "agreement") & HS.source.str.startswith("all") ]
agv = lambda m: AGq[AGq.metric == m].value.iloc[0]
LQ = HS[(HS.block == "llm") & (HS.metric == "qwk")].value; LD = HS[(HS.block == "llm") & (HS.metric == "llm_minus_human")].value
write("tab_human_rating", r"""\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrrrrrr@{}}\toprule
& \multicolumn{7}{c}{Against the given profile (50 interviews)} & Severity persona \\
\cmidrule(lr){2-8}\cmidrule(l){9-9}
Rater & Exact & Within 1 & $\kappa_w$ & Absent$\to$absent & Exact if present & 1 rated higher & Shift & All 8 (\%) \\\midrule
""" + "\n".join(hrows) + r"""
\bottomrule\end{tabular}
\caption{Human ratings of 64 interview transcripts, PHQ-9 items 1--8 (Appendix~\ref{app:prompts}). Columns 2--8: the 40 real-profile and 10 sampled-profile interviews against the profile given to the patient model, as in Table~\ref{tab:interview-agreement}; 1 rated higher: \% of symptoms given as ``several days'' rated higher, with 95\% bootstrap intervals over transcripts. All 8: \% of the 14 severity-persona interviews rated as endorsing all eight items. Model rater: the rater used in the paper, on the same transcripts. Agreement between the human raters, all pairs: exact """ + f"{agv('exact_min'):.0f}--{agv('exact_max'):.0f}" + r"""\%, $\kappa_w$ """ + f"{agv('qwk_min'):.2f}--{agv('qwk_max'):.2f}" + r"""; each human rater against the model rater: $\kappa_w$ """ + f"{LQ.min():.2f}--{LQ.max():.2f}" + r""", the model rater higher by """ + f"{LD.min():.2f}--{LD.max():.2f}" + r""" levels on average.}
\label{tab:human-rating}\end{table*}
""")

AE = pd.read_csv(TABLES / "agent_eval.csv"); AE = AE[AE.primary]
AH = pd.read_csv(TABLES / "agent_eval_human.csv")
al = []
for lab in ORDER:
    q = AE[AE.model == lab].set_index("cond")
    if not len(q): continue
    g = lambda c, m, d=0: f"{q.loc[c, m]:.{d}f}" if c in q.index and pd.notna(q.loc[c, m]) else "---"
    al.append(f"{SHORT[lab]} & {g('free', 'exact')} & {g('free', 'within1')} & {g('free', 'above')} & {g('free', 'absent', 1)} & {g('oracle', 'exact')} & {g('oracle', 'within1')} & {g('oracle', 'above')} & {g('oracle', 'absent', 1)} & {g('oracle', 'absent_detect')} & {g('oracle', 'agree')} & {g('grm', 'exact')} & {g('grm', 'within1')} \\\\")
hb = lambda src, cond, m: AH[(AH.source.str.match(r"R\d") if src == "h" else AH.source == "LLM rater") & (AH.cond == cond)][m]
hv_ = lambda src, cond, m: f"{hb(src, cond, m).mean():.0f}" + (f" ({hb(src, cond, m).min():.0f}--{hb(src, cond, m).max():.0f})" if src == "h" else "")
# bands reweighted to equal numbers of transcripts per severity, as in the full design (_bal)
hrow = lambda src, lab: (f"64 transcripts, {lab} & " + " & ".join(hv_(src, c, m) for c, m in (("free", "exact_bal"), ("free", "within1_bal"), ("free", "above_bal"), ("free", "absent"), ("oracle", "exact_bal"), ("oracle", "within1_bal"), ("oracle", "above_bal"), ("oracle", "absent"), ("oracle", "absent_detect"), ("oracle", "agree"), ("grm", "exact_bal"), ("grm", "within1_bal"))) + " \\\\")
al += ["\\midrule", hrow("l", "model rater"), hrow("h", "human raters")]
write("tab_agent", r"""\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrrrrrrrrrr@{}}\toprule
& \multicolumn{4}{c}{Told a severity} & \multicolumn{6}{c}{Given a real profile} & \multicolumn{2}{c}{Sampled profile} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-11}\cmidrule(l){12-13}
Patient model & Exact & Within 1 & Above & Absent & Exact & Within 1 & Above & Absent & Abs.\ det. & Agree & Exact & Within 1 \\\midrule
""" + "\n".join(al) + r"""
\bottomrule\end{tabular}
\caption{The interviewer and model rater as an assessment agent. Exact, Within 1: \% of interviews whose rated PHQ-9 total falls in the patient's severity band (none 0--4, mild 5--9, moderate 10--14, moderately severe 15--19, severe 20--27), or within one band; the patient's severity is the instructed one for patients told a severity and the band of the given profile otherwise. Above: \% rated in a higher band. Absent: \% of symptoms 1--8 rated absent. Abs.\ det.: \% of symptoms absent from the given profile that are rated absent. Agree: \% of symptoms 1--8 whose presence or absence matches the given profile. Bottom rows: the 64 transcripts rated by people (14, 40 and 10 per condition), scored the same way, with Exact, Within 1 and Above averaged with equal weight over the patients' severities, as in the full design; human raters, mean (range) over the five raters. Table~\ref{tab:interview-talkdep} gives the TalkDep personas.}
\label{tab:agent}\end{table*}
""")

# TalkDep personas in interviews (scripts/77): profile, interview ratio and the agent against the persona's label
ITD = pd.read_csv(TABLES / "interview_talkdep.csv"); IPt = pd.read_csv(TABLES / "interview_pipeline.csv").set_index("model")
PRIM = lambda tag: "claude-haiku-4-5" if tag == "gpt41" else "gpt-4.1"
ITD = ITD[[r.rater == PRIM(r.source.split(": ")[1]) for _, r in ITD.iterrows()]].assign(model=lambda q: [TAG2[x.split(": ")[1]] for x in q.source]).set_index("model")
AEt = AE[AE.cond == "talkdep"].set_index("model")
tl = []
for lab in ORDER:
    if lab not in ITD.index: continue
    r, a, ip = ITD.loc[lab], AEt.loc[lab], IPt.loc[lab]
    pr = "---" if pd.isna(ip.get("pipe_ratio_talkdep")) else f"{ip.pipe_ratio_talkdep:.1f}"
    tl.append(f"{SHORT[lab]} & {int(r.n)} & {r.breadth8:.2f} & {r.all8_modplus:.0f} & {r.all8_modplus_h:.0f} & {r.diversity:.2f} & {r.diversity_h:.2f} & {pr} & {ip.auc_talkdep:.2f} & "
              f"{a.exact:.0f} & {a.within1:.0f} & {a.above:.0f} & {a.bdi_rho:.2f} & {a.bdi_rho_persona:.2f} \\\\")
write("tab_interview_talkdep", r"""\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrrrrrrrrrrr@{}}\toprule
& & & \multicolumn{2}{c}{All 8 (\%)} & \multicolumn{2}{c}{Diversity} & & & \multicolumn{3}{c}{Agent against the label} & \multicolumn{2}{c}{$\rho$ with BDI-II} \\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){10-12}\cmidrule(l){13-14}
Patient model & $n$ & Breadth (/8) & TalkDep & Real$_{=}$ & TalkDep & Real$_{=}$ & Ratio & AUC & Exact & Within 1 & Above & Interviews & Personas \\\midrule
""" + "\n".join(tl) + r"""
\bottomrule\end{tabular}
\caption{Interviews with the 12 TalkDep personas, seven per persona, PHQ-9 items 1--8 rated by the rater used in the paper. All 8: \% rated as endorsing all eight items, personas labelled moderate or above. Real$_{=}$: real NHANES respondents at the same item 1--8 totals. Diversity and Ratio use pairs of interviews with different personas; Ratio: diversity of real-profile interviews over TalkDep interviews within the same bins of three total points, as in Table~\ref{tab:interview-pipeline} (---: no real-profile interviews). AUC: separability from real respondents with the same item 1--8 totals, the interviews of one persona in one cross-validation fold. Agent: rated PHQ-9 band against the band named like the persona's severity label (minimal: none 0--4; mild 5--9; moderate 10--14; moderate to severe: 15--19; severe 20--27), \% exact, within one band and above. $\rho$: Spearman correlation of the rated total with the persona's BDI-II score, over interviews and over the persona means.}
\label{tab:interview-talkdep}\end{table*}
""")

RS = pd.read_csv(TABLES / "residual_structure.csv").set_index("group")
sg2 = lambda v: (f"{round(v, 2) + 0.0:+.2f}").replace("-", "$-$")
sm = lambda v: (f"{round(v, 2) + 0.0:.2f}").replace("-", "$-$")
rl = [f"NHANES, random halves & {num(RS.loc['NHANES halves'].n)} & {sg2(RS.loc['NHANES halves'].contrast)} & {RS.loc['NHANES halves'].similarity:.2f} & & & & & & \\\\",
      f"GRM-sampled profiles & {num(RS.loc['GRM-sampled profiles'].n)} & {sg2(RS.loc['GRM-sampled profiles'].contrast)} & {RS.loc['GRM-sampled profiles'].similarity:.2f} & & & & & & \\\\", "\\midrule"]
for lab in ORDER:
    if lab not in RS.index: continue
    r = RS.loc[lab]
    rl.append(f"{SHORT[lab]} & {int(r.n)} & {sg2(r.contrast)} [{sg2(r.contrast_lo)}, {sg2(r.contrast_hi)}] & {r.similarity:.2f} & {sg2(r.matched_contrast)} & {r.matched_similarity:.2f} [{r.matched_similarity_lo:.2f}] & {sg2(r.null_own_contrast)} [{sg2(r.null_own_contrast_hi)}] & {sm(r.null_own_similarity)} [{r.null_own_similarity_hi:.2f}] & {sg2(r.noise_contrast)} & {r.noise_similarity:.2f} \\\\")
RI = pd.read_csv(TABLES / "residual_structure_interviews.csv").set_index("group"); rl.append("\\midrule")
for g, lab in (("real profiles", "real"), ("sampled profiles", "sampled")):
    r = RI.loc[g]; ref_ = f"{sg2(r.ref80_contrast)} & {r.ref80_similarity:.2f} [{r.ref80_similarity_lo:.2f}]"
    rl.append(f"{lab.capitalize()} profiles given & {int(r.n_profiles)} & {sg2(r.given_contrast)} & {r.given_similarity:.2f} & {ref_} & & & & \\\\")
    rl.append(f"Interviews, {lab} profile & {int(r.n_interviews)} & {sg2(r.rated_contrast)} [{sg2(r.rated_contrast_lo)}, {sg2(r.rated_contrast_hi)}] & {r.rated_similarity:.2f} & {ref_} & & & & \\\\")
write("tab_structure", r"""\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrrrrrrr@{}}\toprule
& & \multicolumn{2}{c}{Group} & \multicolumn{2}{c}{Real$_{=}$} & \multicolumn{2}{c}{No covariation} & \multicolumn{2}{c}{Model $+$ noise} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(l){9-10}
Group & $n$ & Contrast & Similarity & Contrast & Similarity & Contrast & Similarity & Contrast & Similarity \\\midrule
""" + "\n".join(rl) + r"""
\bottomrule\end{tabular}
\caption{Item covariance at a fixed PHQ-9 total (totals 5--24). Residuals are item scores minus the group's mean item scores at the same total. Contrast: mean residual correlation within the somatic (sleep, fatigue, appetite, concentration, psychomotor) and the non-somatic items (interest, mood, worthlessness, suicidal thoughts) minus that between them (all NHANES respondents: """ + f"{sg2(RS.loc['NHANES'].contrast)}" + r"""), with 95\% bootstrap intervals for models. Similarity: correlation of the group's 36 residual correlations with those of all NHANES respondents. Real$_{=}$: real respondents drawn at the model's totals, one per response, mean over 100 draws (2.5th percentile in brackets). No covariation: responses drawn, one per model response, from the maximum-entropy distribution with the model's item distributions at each total, mean over 100 draws (97.5th percentile in brackets). Model $+$ noise: each item moved one level up or down with a probability chosen so that diversity reaches that of real respondents with the same totals, mean over 20 draws. Bottom rows: the profiles given in interviews and the PHQ-9 rated from those interviews (primary rater), pooled over the five patient models, which were given the same profiles; intervals resample profiles, and Real$_{=}$ draws as many real respondents as there are profiles, at their totals (200 draws).}
\label{tab:structure}\end{table*}
""")

# main-text summary of the prompt changes and published personas (seven models, median and range)
FXa = pd.read_csv(TABLES / "fixes.csv"); FXreal = FXa[FXa.cond == "-"].iloc[0]; FXa = FXa[FXa.cond != "-"].assign(divp=lambda q: 100 * q.diversity / q.diversity_h)
REM = [("label", "One-line persona"), ("hetero", "+ ``people differ''"), ("biography", "+ biography"), ("numeric", "Target total, no label"),
       ("fewshot", "+ 5 real answer sheets"), ("talkdep", "TalkDep personas"), ("cactus", "CACTUS personas"), ("cactus_cm", "+ case formulation")]
mr = lambda q, c, d=0: f"{q[c].median():.{d}f} ({q[c].min():.{d}f}--{q[c].max():.{d}f})"
rl2 = [f"Real respondents, held out & 100 & {FXreal.zero18:.1f} & --- & --- \\\\", "\\midrule"]
for k, lab in REM:
    q = FXa[FXa.cond == k]
    rl2.append(f"{lab} & {mr(q, 'divp')} & {q.zero18.median():.1f} & {mr(q, 'auc', 2)} & " + (f"{q.calib.median():.0f}" if q.calib.notna().any() else "---") + " \\\\")
    if k == "fewshot": rl2.append("\\midrule")
write("tab_remedies", r"""\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}lrrrr@{}}\toprule
Condition & Diversity (\%) & N.a.a. & AUC & Calib. \\\midrule
""" + "\n".join(rl2) + r"""
\bottomrule\end{tabular}
\caption{Prompt changes and published personas, PHQ-9, seven models: median (range). Diversity: \% of held-out real respondents with the same totals (totals 5--27). N.a.a.: ``not at all'' answers, items 1--8 at totals 15--24 (\%). AUC: separability from held-out real respondents (totals 15--24). Calib.: \% of totals in the instructed band; the personas name no band. TalkDep: pairs of different personas, a persona in one fold.}
\label{tab:remedies}\end{table}
""")

# main-text summary of the structure results (ranges over models; full values in tab_structure)
_rm = RS.drop(["NHANES", "NHANES halves", "GRM-sampled profiles"]); _ri = RI.loc["real profiles"]; _rs = RI.loc["sampled profiles"]
rg2 = lambda s_: f"{s_.min():.2f}--{s_.max():.2f}"
rgs = lambda s_: f"{sg2(s_.min())} to {sg2(s_.max())}"
sl = [f"NHANES, random halves & {RS.loc['NHANES halves'].similarity:.2f} & {sg2(RS.loc['NHANES halves'].contrast)} \\\\",
      f"Real, at each model's totals & {rg2(_rm.matched_similarity)} & {rgs(_rm.matched_contrast)} \\\\",
      f"Models & {rg2(_rm.similarity)} & {rgs(_rm.contrast)} \\\\",
      f"Models' item distributions, no covariation & {sm(_rm.null_own_similarity.min())} to {sm(_rm.null_own_similarity.max())} & {rgs(_rm.null_own_contrast)} \\\\",
      f"Models $+$ noise & {rg2(_rm.noise_similarity)} & {rgs(_rm.noise_contrast)} \\\\",
      f"GRM-sampled profiles & {RS.loc['GRM-sampled profiles'].similarity:.2f} & {sg2(RS.loc['GRM-sampled profiles'].contrast)} \\\\"]
write("tab_structure_summary", r"""\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrr@{}}\toprule
Group & Similarity & Contrast \\\midrule
""" + "\n".join(sl) + r"""
\bottomrule\end{tabular}
\caption{Item covariance at a fixed PHQ-9 total (totals 5--24). Similarity: correlation of a group's 36 residual correlations with those of all NHANES respondents. Contrast: mean residual correlation within the somatic and within the non-somatic items minus that between them. Ranges over the 17 models. Real respondents drawn at each model's totals, and responses with the model's item distributions at each total and no further covariation, match its sample size (100 draws); GRM-sampled profiles: 20{,}000. Table~\ref{tab:structure} gives each model and the interviews.}
\label{tab:structure-summary}\end{table}
""")

SO = pd.read_csv(TABLES / "severity_overshoot.csv"); SO = SO[SO.framing == "label"]; ol = []
for lab in ORDER:
    q = SO[SO.model == lab].set_index(["arm", "severity"])
    if not len(q): continue
    cells = " & ".join(f"{q.loc[('naive', b)].median_total:.0f} & {q.loc[('naive', b)].above:.0f}" for b in ("mild", "moderate", "moderately_severe"))
    ol.append(f"{SHORT[lab]} & {cells} & {q.loc['naive'].drop('severe').above.mean():.0f} & {q.loc['coached'].drop('severe').above.mean():.0f} \\\\")
write("tab_overshoot", r"""\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{2.4pt}
\begin{fitwidth}
\begin{tabular}{@{}lrrrrrrrr@{}}\toprule
& \multicolumn{2}{c}{Mild} & \multicolumn{2}{c}{Moderate} & \multicolumn{2}{c}{Mod.\ severe} & \multicolumn{2}{c}{Above, all} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}
Model & Med. & Above & Med. & Above & Med. & Above & Naive & Coached \\\midrule
""" + "\n".join(ol) + r"""
\bottomrule\end{tabular}
\end{fitwidth}
\caption{PHQ-9 totals by instructed severity. Med.: median total; Above: \% of responses above the instructed band (mild 5--9, moderate 10--14, moderately severe 15--19), naive arm. Above, all: mean over the three bands, naive and coached arms.}
\label{tab:overshoot}\end{table}
""")

# ------------------------------------------------------------ A: downstream
DS = pd.read_csv(TABLES / "downstream_si.csv"); NTEST = int(DS.n_test.iloc[0])
NM = lambda t: {"Real (NHANES)": "\\textbf{Real (NHANES 2005--2012)}", "GRM-sampled profiles": "GRM-sampled profiles",
                "All models pooled": "All models pooled"}.get(t, SHORT.get(t, t))
num = lambda n: f"{int(n):,}".replace(",", "{,}")
lines = []
for _, r in DS.iterrows():
    lines.append(f"{NM(r.train)} & {num(r.n_train)} & {f(r.train_pos, 1)} & {f(r.auc)} & {f(r.brier, 3)} & {f(r.slope)} & {f(r.sens, 1)} & {f(r.fpr, 1)} & {f(r.recal_brier, 3)} \\\\")
    if r.train == "GRM-sampled profiles": lines.append("\\midrule")
DC = pd.read_csv(TABLES / "downstream_calibration.csv")
cl = [f"{int(r.lo)}--{int(r.hi)} & {num(r.n)} & {r.observed:.1f} & {r.pred_real:.1f} & {r.pred_GRM:.1f} & {r['pred_LLM pooled']:.1f} \\\\" for _, r in DC.iterrows()]
CT = pd.read_csv(TABLES / "downstream_controls.csv")
sgn = lambda x, d=3: f"{x:+.{d}f}".replace("-", "$-$")
ci_ = lambda r, k: f"{sgn(r[k])} [{sgn(r[k + '_lo'])}, {sgn(r[k + '_hi'])}]"
ctl = []
for _, r in CT.iterrows():
    if r.train == "All models pooled": ctl.append("\\midrule")
    ctl.append(f"{NM(r.train)} & {f(r.sim_pos, 0)} & {f(r.pos_sev, 0)} & {f(r.auc_sim, 3)} & {f(r.auc_rand, 3)} & {f(r.auc_sev, 3)} & {f(r.auc_sevlab, 3)} & {ci_(r, 'd_sev')} & {ci_(r, 'd_sevlab')} & {f(r.brier_sim_recal, 3)} & {f(r.brier_sev, 3)} \\\\")
DA = pd.read_csv(TABLES / "downstream_augmentation.csv")
al = [f"{int(r.n_real)} & +{int(r.ratio)}$\\times$ & {int(r.draws)} & {r.base_auc:.3f} & {sgn(r.d_auc)} [{sgn(r.d_auc_lo)}, {sgn(r.d_auc_hi)}] & {sgn(1000 * r.d_brier, 1)} [{sgn(1000 * r.d_brier_lo, 1)}, {sgn(1000 * r.d_brier_hi, 1)}] & {sgn(r.d_citl, 1)} \\\\" for _, r in DA.iterrows()]
write("tab_downstream", r"""\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrrrrrr@{}}\toprule
Trained on & $n$ & SI & AUC & Brier & Slope & Sens. & FPR & Recal.\ Brier \\\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}
\caption{Predicting whether PHQ-9 item 9 is endorsed from items 1--8. Test set: NHANES 2013--2023 respondents with items 1--8 total $\geq$5 ($n{=}""" + num(NTEST) + r"""$). $n$, SI: size of the training set and \% endorsing item 9 in it. GRM-sampled: profiles from the GRM fit to NHANES 2005--2012. Slope: calibration slope (1 when calibrated). Sens., FPR: sensitivity and false-positive rate at $p\geq0.5$ (\%). Recal.\ Brier: after logistic recalibration on 1{,}000 held-out 2005--2012 respondents with items 1--8 total $\geq$5.}
\label{tab:downstream}\end{table*}

\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{3.4pt}
\begin{tabular}{@{}lrrrrrrrrrr@{}}\toprule
& \multicolumn{2}{c}{SI in training (\%)} & \multicolumn{4}{c}{AUC} & \multicolumn{2}{c}{AUC difference from real} & \multicolumn{2}{c}{Brier} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}
Simulated patients & Sim. & Real$_{t}$ & Sim. & Real$_{r}$ & Real$_{t}$ & Real$_{t,9}$ & Real$_{t}$ & Real$_{t,9}$ & Sim.\ recal. & Real$_{t}$ \\\midrule
""" + "\n".join(ctl) + r"""
\bottomrule\end{tabular}
\caption{Simulated training sets against real training sets of the same size drawn from NHANES 2005--2012: at random (Real$_{r}$), with the same items 1--8 totals (Real$_{t}$), and with the same totals and the same item-9 answers (Real$_{t,9}$); control values are means over """ + str(50) + r""" draws. Differences: simulated minus real, with 95\% paired bootstrap intervals (1{,}000 resamples of the simulated training set and the test set, fresh real draws each time). Sim.\ recal.: simulated-trained model after recalibration on held-out real respondents. Test set as in Table~\ref{tab:downstream}.}
\label{tab:downstream-controls}\end{table*}

\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrrrr@{}}\toprule
Items 1--8 & $n$ & Observed & Real & GRM & Sim. \\\midrule
""" + "\n".join(cl) + r"""
\bottomrule\end{tabular}
\caption{Observed and predicted rate (\%) of endorsing item 9 by total on items 1--8, all NHANES 2013--2023 respondents, for models trained on real respondents, GRM-sampled profiles and simulated patients (pooled over models).}
\label{tab:downstream-cal}\end{table}

\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{2.4pt}
\begin{tabular}{@{}rrrrrrr@{}}\toprule
Real $n$ & Added & Draws & Base AUC & $\Delta$AUC & $\Delta$Brier ($\times10^{-3}$) & $\Delta$Pred. \\\midrule
""" + "\n".join(al) + r"""
\bottomrule\end{tabular}
\caption{Adding simulated patients (pooled over models) to a small random real training set from NHANES 2005--2012. Mean change over draws with its 95\% interval; draws without both classes are skipped. $\Delta$Pred.: change in mean predicted risk minus the observed rate (points). Test set as in Table~\ref{tab:downstream}.}
\label{tab:augment}\end{table}
""")

# ------------------------------------------------------------ A: severity bands
BS = pd.read_csv(TABLES / "by_severity.csv"); BSM = pd.read_csv(TABLES / "by_severity_models.csv")
lines = []
for band in BS.band.unique():
    q = BS[BS.band == band]
    h = q[q.who.str.startswith("NHANES at")].iloc[0]; m = q[q.who.str.startswith("LLMs")].iloc[0]
    gap = (BSM[BSM.band == band].breadth - BSM[BSM.band == band].breadth_h).median()
    lines.append(f"{band} & {int(m.n)} & {h.breadth:.2f} / {m.breadth:.2f} & ${gap:+.2f}$ & {h.zero:.1f} / {m.zero:.1f} & "
                 f"{h.uneven:.2f} / {m.uneven:.2f} & {h.diversity:.2f} / {m.diversity:.2f} \\\\")
write("tab_severity", r"""\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{2.4pt}
\begin{tabular}{@{}lrrrrrr@{}}\toprule
Band & Models & Breadth & Gap & N.a.a.\ (\%) & Unev. & Div. \\\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}
\caption{PHQ-9 by total-score band. Each cell: real respondents at the models' totals / models, median over models with at least 20 responses in the band. Gap: median over models of each model's breadth minus that of real respondents at its totals.}
\label{tab:severity}\end{table}
""")

# ------------------------------------------------------------ A: GRM + person fit
for key, csv, lab in (("phq9", "grm_item_parameters.csv", "PHQ-9, NHANES ($N{=}41{,}714$)"),
                      ("gad7", "grm_items_gad7.csv", "GAD-7, NHIS ($N{=}31{,}132$)")):
    g = pd.read_csv(TABLES / csv)
    lines = [f"{r.symptom[:34]} & {r.a:.2f} & {r.b1:.2f} & {r.b2:.2f} & {r.b3:.2f} & {r.pct_endorsed:.1f} \\\\" for _, r in g.iterrows()]
    write(f"tab_grm_{key}", r"""\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{2.4pt}
\begin{tabular}{@{}lrrrrr@{}}\toprule
Item & $a$ & $b_1$ & $b_2$ & $b_3$ & \%end. \\\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}
\caption{Graded response model item parameters, """ + lab + r""".}
\label{tab:grm-""" + key + r"""}\end{table}
""")
PF = pd.read_csv(TABLES / "person_fit_probes.csv")
lines = []
for instr in ("phq9", "gad7"):
    lines.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{'PHQ-9' if instr == 'phq9' else 'GAD-7'}}}}} \\\\")
    for _, r in PF[PF.instrument == instr].iterrows():
        lines.append(f"\\quad {SHORT.get(r.model, r.model)} & {r.lz:+.2f} & {r.guttman_normed:+.2f} & "
                     f"{r.rare_symptom_surprisal:+.2f} & {r.symptom_combo_surprisal:+.2f} & {r.extreme_severity:+.2f} \\\\")
write("tab_personfit", r"""\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{@{}lrrrrr@{}}\toprule
Model & $l_z$ & $G^p$ & Rare & Comb. & Extr. \\\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}
\caption{Person-fit statistics and base-rate probes, Cohen's $d$ against total-matched real respondents. Positive $l_z$ and negative $G^p$: better fit than real responses. Rare, Comb., Extr.: rare-symptom, rare-combination and extreme-severity probes.}
\label{tab:personfit}\end{table}
""")

# ------------------------------------------------------------ A: which models each analysis uses
AMc = pd.read_csv(TABLES / "all_models.csv"); LPc = pd.read_csv(TABLES / "logprob_mechanism.csv")
FXc = pd.read_csv(TABLES / "fixes.csv"); IVc = pd.read_csv(TABLES / "interview.csv"); SCc = pd.read_csv(TABLES / "scale_ci.csv")
T2L = {m[3]: m[0] for m in MODELS}
def names(labels):
    labels = [l for l in ORDER if l in set(labels)]
    return "all" if len(labels) == len(ORDER) else ", ".join(SHORT[l] for l in labels)
cov = [("PHQ-9, GAD-7, PCL-5 roles, six extension instruments; person fit, first-answer correlation, screening", set(AMc[AMc.model != "REAL"].model)),
       ("Log-probability probe", {T2L[t] for t in LPc.model.unique()}),
       ("Interventions; interviews, told a severity", {T2L[t] for t in FXc[FXc.cond.isin(["label", "hetero", "biography", "numeric", "fewshot"])].model.unique()}),
       ("Published personas in the intervention protocol (TalkDep, CACTUS, CACTUS with case formulation)", {T2L[t] for t in set(FXc[FXc.cond == "talkdep"].model) & set(FXc[FXc.cond == "cactus"].model) & set(FXc[FXc.cond == "cactus_cm"].model)}),
       ("Published personas, PHQ-9 and GAD-7 (the same three sets)", (lambda P: {m for m, g in P.groupby("model") if len(g) == 8})(pd.read_csv(TABLES / "published_personas.csv"))),
       ("Interviews, given a real profile", {T2L[x.split(": ")[1]] for x in IVc.source if x.startswith("oracle")}),
       ("Interviews, given a sampled profile", {T2L[x.split(": ")[1]] for x in pd.read_csv(TABLES / "interview_grm.csv").source}),
       ("Scale ladders", set(SCc.model)),
       ("Robustness (temperature, format, framing)", {"GPT-4o-mini", "GPT-5.2"}),
       ("Reasoning effort (two to four settings, including the API default): questionnaires, feigning roles, nine references and screening; for the Gemini models also interventions, published personas and interviews",
        {m.replace("Gemini 2.5 ", "Gemini-2.5-") for m in pd.read_csv(TABLES / "reasoning_effort.csv").model})]
lines = [f"{a} & {len(b)} & {names(b)} \\\\" for a, b in cov]
write("tab_coverage", r"""\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}p{0.34\textwidth}rp{0.56\textwidth}@{}}\toprule
Analysis & $n$ & Models \\\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}
\caption{Models used in each analysis.}
\label{tab:coverage}\end{table*}
""")
