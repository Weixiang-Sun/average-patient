"""Emit every appendix table as LaTeX into paper/appendix_tables.tex."""
import sys, json, zlib
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from mental_act import analysis as A, scoring
from mental_act.config import DATA_PROC, GAD7_ITEMS, GAD7_LABELS, PHQ9_ITEMS, PHQ9_LABELS, TABLES

OUT = []
def emit(s): OUT.append(s)

def esc(x):
    return str(x).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

SHORT = {"GPT-4o-mini":"GPT-4o-mini","GPT-5.4-mini":"GPT-5.4-mini","GPT-5.2":"GPT-5.2",
         "Claude Haiku 4.5":"Haiku 4.5","Grok-4.1-fast":"Grok-4.1","Llama-3.3-70B":"Llama-3.3",
         "Kimi-K2.5":"Kimi-K2.5"}

# ---------------------------------------------------- C. GRM parameters
for tag, csv, lab, cap in [
    ("phq9", "grm_item_parameters.csv", PHQ9_LABELS,
     "Graded response model parameters for the PHQ-9, fitted by marginal ML on "
     "NHANES ($N{=}41{,}714$). $a$ is discrimination, $b_k$ the $k$th category "
     "threshold. Suicidal ideation is the hardest item and depressed mood the "
     "most discriminating, reproducing the published PHQ-9 IRT literature."),
    ("gad7", "grm_items_gad7.csv", GAD7_LABELS,
     "Graded response model parameters for the GAD-7, fitted on NHIS "
     "($N{=}31{,}132$).")]:
    d = pd.read_csv(TABLES/csv)
    emit(r"\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{2.5pt}")
    emit(r"\begin{tabular}{@{}lrrrrr@{}}\toprule")
    emit(r"Item & $a$ & $b_1$ & $b_2$ & $b_3$ & \%end. \\\midrule")
    for _, r in d.iterrows():
        nm = r["symptom"][:25]
        emit(f"{esc(nm)} & {r['a']:.2f} & {r['b1']:.2f} & {r['b2']:.2f} & "
             f"{r['b3']:.2f} & {r['pct_endorsed']:.1f} \\\\")
    emit(r"\bottomrule\end{tabular}")
    emit(r"\caption{" + cap + r"}\label{tab:grm-" + tag + r"}\end{table}")
    emit("")

# ------------------------------------- D. full profile measures
d = pd.read_csv(TABLES/"deep_analysis.csv")
emit(r"\begin{table*}[t]\centering\small")
emit(r"\begin{tabular}{llrrrrrrrr}\toprule")
emit(r"Model & Vendor & Brd. & All\% & Depth & Uneven. & Top\% & $\rho_{\mathrm{ord}}$ & $d_{\mathrm{corr}}$ & AUC \\\midrule")
for instr, name in [("phq9", r"\multicolumn{10}{l}{\emph{PHQ-9 (depression), NHANES reference}}\\"),
                    ("gad7", r"\multicolumn{10}{l}{\emph{GAD-7 (anxiety), NHIS reference}}\\")]:
    s = d[d.instrument == instr]
    emit(name)
    h = s.iloc[0]
    emit(rf"\quad Real respondents & --- & {h.breadth_h:.2f} & {h.all_items_pct_h:.1f} & "
         rf"{h.depth_h:.2f} & {h.spikiness_h:.2f} & {100*h.top_share_h:.1f} & 1.00 & 0.00 & 0.50 \\")
    for _, r in s.iterrows():
        emit(rf"\quad {SHORT.get(r.model, r.model)} & {r.vendor} & {r.breadth:.2f} & "
             rf"{r.all_items_pct:.1f} & {r.depth:.2f} & {r.spikiness:.2f} & "
             rf"{100*r.top_share:.1f} & {r.order_rho:.2f} & {r.corr_dist:.2f} & {r.separability:.2f} \\")
    if instr == "phq9": emit(r"\midrule")
emit(r"\bottomrule\end{tabular}")
emit(r"""\caption{Full profile measures, computed within a fixed total-score band
(PHQ-9 15--24, GAD-7 12--19). Brd.: mean symptoms endorsed. All\%: percentage of
responses endorsing every item. Uneven.: within-person s.d. of item scores.
Top\%: share of endorsed symptoms at the top category.
$\rho_{\mathrm{ord}}$: Spearman correlation between model and human item mean
scores. $d_{\mathrm{corr}}$: RMS difference between symptom correlation matrices.
AUC: cross-validated separability from total-matched humans.}
\label{tab:full-measures}\end{table*}""")
emit("")

# ------------------------------------- E. person-fit and base-rate probes
rows = []
for instr, models in [("phq9", A.PHQ9_ALL), ("gad7", A.GAD7_ALL)]:
    norms = scoring.load_norms(tag="" if instr == "phq9" else instr)
    hidx = pd.read_csv(DATA_PROC/("human_indices.csv" if instr == "phq9"
                                  else f"human_indices_{instr}.csv"))
    for lab, run, vend in models:
        if A.load_run(run) is None: continue
        X = A.run_X(run)
        s = scoring.score(X, norms)
        # 20 total-matched humans per response, seeded per model so a value does
        # not depend on which models precede it in the registry
        rng = np.random.default_rng(zlib.crc32(f"{instr}|{lab}".encode()))
        m = scoring.matched_human_reference(hidx, s["total"], rng=rng, n_per=20)
        rows.append(dict(instrument=instr, model=lab,
                         **{k: scoring.cohens_d(s[k], m[k]) for k in
                            ["lz", "guttman_normed", "rare_symptom_surprisal",
                             "symptom_combo_surprisal", "extreme_severity"]}))
pf = pd.DataFrame(rows); pf.to_csv(TABLES/"person_fit_probes.csv", index=False)
emit(r"\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{2.2pt}")
emit(r"\begin{tabular}{@{}lrrrrr@{}}\toprule")
emit(r"Model & $l_z$ & $G^p$ & Rare & Comb. & Extr. \\\midrule")
for instr in ("phq9", "gad7"):
    emit(r"\multicolumn{6}{l}{\emph{" + ("PHQ-9" if instr == "phq9" else "GAD-7") + r"}}\\")
    for _, r in pf[pf.instrument == instr].iterrows():
        emit(rf"\quad {SHORT.get(r.model, r.model)} & {r.lz:+.2f} & {r.guttman_normed:+.2f} & "
             rf"{r.rare_symptom_surprisal:+.2f} & {r.symptom_combo_surprisal:+.2f} & "
             rf"{r.extreme_severity:+.2f} \\")
    if instr == "phq9": emit(r"\midrule")
emit(r"\bottomrule\end{tabular}")
emit(r"""\caption{Person-fit statistics and base-rate probes, as Cohen's $d$
against total-matched real respondents (0 = indistinguishable). $l_z$:
standardised log-likelihood from the fitted GRM; positive values indicate
responses \emph{more} likely than the model expects, i.e.\ over-consistency.
$G^p$: polytomous Guttman errors normalised by the maximum attainable at the
same total. Rare / Comb. / Extr.: rare-symptom, rare-combination and
extreme-severity probes calibrated on the reference sample. Every model is
over-consistent on $G^p$ and most are positive on $l_z$.}
\label{tab:personfit}\end{table}""")
emit("")

# ------------------------------------- F. item-level endorsement
ip = pd.read_csv(TABLES/"item_profiles.csv")
emit(r"\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{2.5pt}")
emit(r"\begin{tabular}{@{}lrrr@{}}\toprule")
emit(r"Item & Real & Models & Gap \\\midrule")
for instr in ("phq9", "gad7"):
    emit(r"\multicolumn{4}{l}{\emph{" + ("PHQ-9" if instr == "phq9" else "GAD-7") + r"}}\\")
    for _, r in ip[ip.instrument == instr].iterrows():
        emit(rf"\quad {esc(r.label[:27])} & {r.human:.1f} & {r.models:.1f} & {r.gap:+.1f} \\")
    if instr == "phq9": emit(r"\midrule")
emit(r"\bottomrule\end{tabular}")
emit(r"""\caption{Percentage of respondents endorsing each item at all, within the
fixed total-score band, ordered from most to least endorsed by real
respondents. Models column is the mean across models. The gap is monotone in the
human rate on both instruments.}
\label{tab:items}\end{table}""")
emit("")

# ------------------------------------- G. dose-response
TARGET = {"none":(0,4),"mild":(5,9),"moderate":(10,14),"moderately_severe":(15,19),"severe":(20,27)}
ORD = list(TARGET)
emit(r"\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{2.5pt}")
emit(r"\begin{tabular}{@{}lrrrrrr@{}}\toprule")
emit(r"Model & none & mild & mod. & m.sev & severe & Hit\% \\")
emit(r" & (0--4) & (5--9) & (10--14) & (15--19) & (20--27) & \\\midrule")
for lab, run, v in A.PHQ9_ALL:
    d0 = A.load_run(run)
    if d0 is None: continue
    d0 = d0.copy(); d0["tot"] = [sum(o) for o in d0["ordered"]]
    g = d0.groupby("severity").tot.mean()
    hit = np.mean([(lo <= t <= hi) for t, (lo, hi) in zip(d0.tot, d0.severity.map(TARGET))])*100
    emit(rf"{SHORT.get(lab,lab)} & " + " & ".join(f"{g.get(s, np.nan):.1f}" for s in ORD) + rf" & {hit:.1f} \\")
emit(r"\bottomrule\end{tabular}")
emit(r"""\caption{Mean PHQ-9 total produced for each instructed severity, with the
target range in parentheses. Hit\% is the percentage of all trials landing in
the instructed band. Kimi-K2.5 is near-perfectly calibrated yet has among the
least human-like profiles (Table~\ref{tab:full-measures}).}
\label{tab:dose}\end{table}""")
emit("")

# ------------------------------------- H. robustness
GROUPS = [("Model", [(l, r) for l, r, _ in A.PHQ9_ALL]),
          ("Temperature", [("$t=0.0$","temp00"),("$t=0.5$","temp05"),("$t=1.0$","pilot_gpt4omini")]),
          ("Format", [("whole form, array","pilot_gpt4omini"),
                      ("whole form, object","gpt4omini_objschema"),
                      ("item by item","peritem")]),
          ("Prompt", [("diagnostic label","gpt4omini_objschema"),
                      ("impairment only","4om_impairment"),
                      ("clinical chart","4om_chart"),
                      ("label, GPT-5.2","gpt52"),
                      ("impairment, GPT-5.2","g52_impairment"),
                      ("chart, GPT-5.2","g52_chart")])]
Hb = A.in_band(A.human_X("phq9"), "phq9")
emit(r"\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{2.5pt}")
emit(r"\begin{tabular}{@{}llrrr@{}}\toprule")
emit(r"Axis & Condition & Brd. & Unev. & $n$ \\\midrule")
emit(rf"--- & Real respondents & {A.breadth(Hb).mean():.2f} & {A.spikiness(Hb).mean():.2f} & {len(Hb)} \\\midrule")
for gi, (g, runs) in enumerate(GROUPS):
    for j, (nm, run) in enumerate(runs):
        if A.load_run(run) is None: continue
        M = A.in_band(A.run_X(run), "phq9")
        emit(rf"{g if j==0 else ''} & {esc(nm)} & {A.breadth(M).mean():.2f} & "
             rf"{A.spikiness(M).mean():.2f} & {len(M)} \\")
    if gi < len(GROUPS)-1: emit(r"\addlinespace")
emit(r"\bottomrule\end{tabular}")
emit(r"""\caption{Breadth and unevenness under every robustness condition,
within the PHQ-9 15--24 band. No condition moves breadth close to the human
value. Item-by-item administration \emph{reduces} unevenness furthest, because
without the rest of the form in context models answer nearly every item at
ceiling.}
\label{tab:robust}\end{table}""")

(TABLES/"appendix_tables.tex").write_text("\n".join(OUT))
print(f"wrote {TABLES/'appendix_tables.tex'}  ({len(OUT)} lines)")
