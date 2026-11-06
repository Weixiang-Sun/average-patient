"""Recompute every number quoted in the paper text that is not read straight
from a generated table, from the raw runs and the analysis CSVs."""
import sys, json, glob
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from mental_act import analysis as A
from mental_act.feign import FEIGNER_RANGE as FG_RANGE
from mental_act.config import RUNS, TABLES, DATA_PROC

T = lambda f: pd.read_csv(TABLES / f"{f}.csv")
def show(k, v): print(f"{k:60s} {v}")
def rng_(s, d=2, pct=False): return f"{s.min():.{d}f}--{s.max():.{d}f}"

AM = T("all_models"); CI = T("main_ci"); DA = T("deep_analysis")
ph = AM[(AM.instrument == "phq9") & (AM.model != "REAL")]; phR = AM[(AM.instrument == "phq9") & (AM.model == "REAL")].iloc[0]
ga = AM[(AM.instrument == "gad7") & (AM.model != "REAL")]; gaR = AM[(AM.instrument == "gad7") & (AM.model == "REAL")].iloc[0]
print("=== Section: profiles (PHQ-9 / GAD-7) ===")
show("real breadth / depth", f"{phR.breadth:.2f} / {DA[DA.instrument=='phq9'].depth_h.iloc[0]:.2f}")
show("model breadth range", rng_(ph.breadth)); show("model depth range", rng_(DA[DA.instrument == "phq9"].depth))
show("real not-at-all / nearly-every-day", f"{phR.zero:.1f} / {phR.top:.1f}"); show("model not-at-all", rng_(ph.zero, 1))
Hb = A.in_band(A.human_X("phq9"), "phq9")
opt2 = [100 * (A.in_band(A.run_X(m[1]), "phq9") == 2).mean() for m in A.ALL_MODELS]
show("option 2 share: model mean / real", f"{np.mean(opt2):.1f} / {100 * (Hb == 2).mean():.1f}")
# real respondents at exactly each model's totals, weighted by the model's total counts (as in 36_all_models)
def _at_totals(M, fn, X=Hb):
    tt, cc = np.unique(M.sum(1), return_counts=True); th = X.sum(1)
    return float(np.sum([fn(X[th == t]) * c for t, c in zip(tt, cc)]) / cc.sum())
_o2h = pd.Series([_at_totals(A.in_band(A.run_X(m[1]), "phq9"), lambda Q: 100 * (Q == 2).mean()) for m in A.ALL_MODELS])
show("option 2 share, real at each model's totals (range over models)", rng_(_o2h, 0))
show("all nine: models / real", f"{rng_(ph.all_items, 1)} / {phR.all_items:.1f}")
show("GAD-7 all seven: models / real", f"{rng_(ga.all_items, 1)} / {gaR.all_items:.1f}")
show("diversity real (all) / real at each model's totals / models / ratio (same totals)", f"{phR.diversity:.2f} / {rng_(ph.diversity_h)} / {rng_(ph.diversity)} / {(ph.diversity_h / ph.diversity).min():.1f}--{(ph.diversity_h / ph.diversity).max():.1f}")
show("GAD-7 diversity: real at each model's totals / models / ratio", f"{rng_(ga.diversity_h)} / {rng_(ga.diversity)} / {(ga.diversity_h / ga.diversity).min():.1f}--{(ga.diversity_h / ga.diversity).max():.1f}")
show("AUC phq9 / gad7", f"{rng_(ph.auc)} / {rng_(ga.auc)}")
show("hierarchy rho phq9 / gad7", f"{rng_(ph.rho)} / {rng_(ga.rho)}")
BSM = T("by_severity_models")
for b in BSM.band.unique():
    q = BSM[BSM.band == b]; show(f"band {b}: models, breadth gap vs same totals (median, range), zero model/real", f"{len(q)}, {(q.breadth - q.breadth_h).median():+.2f} [{(q.breadth - q.breadth_h).min():+.2f}, {(q.breadth - q.breadth_h).max():+.2f}], {q.zero.median():.0f} / {q.zero_h.median():.0f}")
show("real at models' totals (15-24): breadth / all items / zero", f"{rng_(ph.breadth_h)} / {rng_(ph.all_items_h, 0)} / {rng_(ph.zero_h, 0)}; GAD-7 all items {rng_(ga.all_items_h, 0)}")
IP = T("item_profiles").set_index("item")
for it in ("DPQ040", "DPQ080", "DPQ090"): show(f"item {it} real -> models", f"{IP.loc[it].human:.1f} -> {IP.loc[it].models:.1f}")
show("SI at totals 15-24: models / real", f"{rng_(CI[(CI.instrument == 'phq9') & (CI.model != 'REAL')].si, 1)} / {CI[(CI.instrument == 'phq9') & (CI.model == 'REAL')].si.iloc[0]:.1f}")
PF = T("person_fit_probes")
for ins in ("phq9", "gad7"):
    q = PF[PF.instrument == ins]
    show(f"person fit {ins}: lz range, #lz>0, Gp range, rare, comb, extr",
         f"{rng_(q.lz)}, {(q.lz > 0).sum()}/{len(q)}, {rng_(q.guttman_normed)}, {rng_(q.rare_symptom_surprisal)}, {rng_(q.symptom_combo_surprisal)}, {rng_(q.extreme_severity)}")
PP = T("prototype_pole_vs_human")
_ph = {m[0]: A.run_X(m[1]) for m in A.ALL_MODELS}
show("model mean total within 15-24 / real", f"{rng_(pd.Series([X[(X.sum(1) >= 15) & (X.sum(1) <= 24)].sum(1).mean() for X in _ph.values()]), 1)} / {A.human_X('phq9')[(A.human_X('phq9').sum(1) >= 15) & (A.human_X('phq9').sum(1) <= 24)].sum(1).mean():.1f}")
print("=== Generality ===")
G = T("extra_generality"); Gm = G[G.ref != "coach_vs_nhanes"].dropna(subset=["endorsed_m"]); Gm = Gm.assign(ratio=Gm.div_h / Gm.div_m)
show("variety ratio over all refs", f"{Gm.ratio.min():.1f}--{Gm.ratio.max():.1f}")
show("AUC K6+ITQ", rng_(Gm[Gm.ref.isin(["k6", "itq"])].auc)); show("min AUC per ref", Gm.groupby("ref").auc.min().round(2).to_dict())
show("AUC null range", rng_(G.auc_null.dropna()))
_fx = T("fixes"); show("label-shuffled chance across protocols (Table 1, interventions, personas)", f"{min(G.auc_null.min(), _fx.auc_null.min()):.2f}--{max(G.auc_null.max(), _fx.auc_null.max()):.2f}")
_dt = T("default_tokens").set_index("model"); _rs = _dt[(_dt.reasoning >= 50) | (_dt.generated > 300)]; _rd = _dt[(_dt.reasoning_default >= 50) | (_dt.generated_default > 300)]
show("main runs: models that reason / their reasoning tokens", f"{list(_rs.index)} / {rng_(_rs.reasoning.dropna(), 0)}")
show("main runs: generated tokens, GPT-5-mini / Gemini Flash / Kimi", f"{_dt.loc['GPT-5-mini'].generated:.0f} / {_dt.loc['Gemini-2.5-Flash'].generated:.0f} / {_dt.loc['Kimi-K2.5'].generated:.0f}")
show("API default (switched models): reasoning tokens / Kimi generated", f"{list(_rd.index)} {rng_(_rd.reasoning_default.dropna(), 0)} / {_dt.loc['Kimi-K2.5'].generated_default:.0f}")
hi = Gm.groupby("ref").apply(lambda g: (g.all_items_m > g.all_items_h).mean()); show("refs where all models exceed real all-items", f"{(hi == 1).sum()} ({hi.round(2).to_dict()})")
EI = T("extra_items"); print(EI[EI.instrument.isin(["ocir", "phq15"])].sort_values("gap").head(6).round(1).to_string(index=False))
cn = G[G.ref == "coach_vs_nhanes"]; print(cn[["all_items_m", "all_items_h", "auc"]].round(3).to_string(index=False))
co = Gm[Gm.ref == "phq9_coach"]; show("COACH all-nine real (median over models' matched sets)", f"{co.all_items_h.median():.1f}")
print("=== Feigning ===")
TC = T("pcl5_triangle_ci").set_index("group"); L = TC[TC.index.str.startswith("LLM")]
show("feigner auc [lo,hi] p", f"{TC.loc['human feigner'].auc:.3f} [{TC.loc['human feigner'].sub_lo:.3f}, {TC.loc['human feigner'].sub_hi:.3f}] {TC.loc['human feigner'].p:.3f}")
show("models auc range / max p / min lower", f"{rng_(L.auc, 3)} / {L.p.max():.3f} / {L.sub_lo.min():.3f}")
sig = L[L.p_holm < 0.05]; ns = L[L.p_holm >= 0.05]
show("models p<0.05 raw / Holm", f"{(L.p < 0.05).sum()} / {len(sig)} of {len(L)}")
show("Holm-significant: AUC range / lower bounds; others", f"{rng_(sig.auc)} / {rng_(sig.sub_lo)}; " + ", ".join(f"{i[5:]} {r.auc:.2f} (p={r.p:.3f}, Holm {r.p_holm:.2f})" for i, r in ns.iterrows()))
show("direct test vs feigners: p<0.05 raw / Holm; not significant (raw)", f"{(L.direct_p < 0.05).sum()} / {(L.direct_p_holm < 0.05).sum()} of {len(L)}; " + ", ".join(f"{i[5:]} {r.direct_p:.3f}" for i, r in L[L.direct_p >= 0.05].iterrows()))
show("real-patient baseline at model totals / models in same splits", f"{rng_(L.base_real)} / {rng_(L.base_group)}")
show("real-patient baseline: feigners, honest", f"{TC.loc['human feigner'].base_real:.2f}, {TC.loc['human honest'].base_real:.2f}")
show("subsample interval lower bounds, all models", rng_(L.sub_lo))
from mental_act.feign import GEN as _G, HONEST as _Ho
show("honest totals median / at 0; genuine at 0 / at <=5", f"{np.median(_Ho.sum(1)):.1f} / {(_Ho.sum(1) == 0).sum()}; {(_G.sum(1) == 0).sum()} / {(_G.sum(1) <= 5).sum()}")
from scipy.stats import spearmanr as _sp
_tq = np.digitize(_G.sum(1), [20, 40, 60])
show("genuine PCL-5: \"not at all\" % by total (<20, 20-39, 40-59, >=60); Spearman total vs share",
     " / ".join(f"{100 * (_G[_tq == q] == 0).mean():.0f}" for q in range(4)) + f"; rho {_sp(_G.sum(1), (_G == 0).mean(1))[0]:.2f}")
PCf = T("pcl5_profile_ci").set_index("group"); PL = PCf[PCf.index.str.startswith("LLM")]; ph_ = PCf.loc["human feigner"]; ho_ = PCf.loc["human honest"]
show("feigner naa ratio [lo,hi]; breadth / gen; naa / gen", f"{ph_.naa_ratio:.2f} [{ph_.naa_lo:.2f}, {ph_.naa_hi:.2f}]; {ph_.breadth:.1f} / {ph_.breadth_gen:.1f}; {ph_.naa:.0f} / {ph_.naa_gen:.0f}")
show("honest naa ratio [lo,hi]", f"{ho_.naa_ratio:.2f} [{ho_.naa_lo:.2f}, {ho_.naa_hi:.2f}]")
show("models naa ratio range / max upper bound", f"{rng_(PL.naa_ratio)} / {PL.naa_hi.max():.2f}")
show("models: breadth / gen; naa / gen (matched)", f"{rng_(PL.breadth, 1)} / {rng_(PL.breadth_gen, 1)}; {rng_(PL.naa, 0)} / {rng_(PL.naa_gen, 0)}")
show("feigner minus model ratio, Bonferroni lower bound: min (all > 0?)", f"{PL.feigner_minus_lo.min():.2f} ({(PL.feigner_minus_lo > 0).all()})")
show("feigners' 10-90% total range; models' ratio within it; feigner within it", f"{rng_(pd.Series(FG_RANGE), 0)}; {rng_(PL.naa_ratio_in_range)}; {ph_.naa_ratio_in_range:.2f}")
show("breadth gap: feigner [lo,hi]; models range / min lower", f"{ph_.breadth_gap:+.2f} [{ph_.gap_lo:+.2f}, {ph_.gap_hi:+.2f}]; {rng_(PL.breadth_gap, 1)} / {PL.gap_lo.min():.2f}")
show("people shift: dtotal [lo,hi]; dextreme [lo,hi]", f"{ph_.dtotal:+.1f} [{ph_.dtotal_lo:.1f}, {ph_.dtotal_hi:.1f}]; {ph_.dextreme:+.1f} [{ph_.dext_lo:.1f}, {ph_.dext_hi:.1f}]")
show("models raising both with 95% intervals above zero", f"{((PL.dtotal_lo > 0) & (PL.dext_lo > 0)).sum()} of {len(PL)}")
_both = PL[(PL.dtotal_lo > 0) & (PL.dext_lo > 0)]
show("those models: lower bounds of change in total / in extremely", f"{rng_(_both.dtotal_lo, 1)} / {rng_(_both.dext_lo, 1)}")
show("absolute not-at-all diff (gen - group), all totals: feigner / models", f"{ph_.naa_gen - ph_.naa:.1f} / {rng_(PL.naa_gen - PL.naa, 1)}")
from statsmodels.stats.multitest import multipletests as _mt
_PR = PL[PL.n_in_range >= 10].copy(); _PR["p_holm10"] = _mt(_PR.abs_minus_feigner_p, method="holm")[1]
show("in feigners' range: n per model (all) / models with n<10 / abs diff models (n>=10) / feigner [lo,hi]", f"{rng_(PL.n_in_range, 0)} / {list(PL[PL.n_in_range < 10].index)} {PL[PL.n_in_range < 10].n_in_range.tolist()} / {rng_(_PR.naa_abs_in_range, 1)} / {ph_.naa_abs_in_range:.1f} [{ph_.naa_abs_in_range_lo:.1f}, {ph_.naa_abs_in_range_hi:.1f}]")
show("in range (n>=10): models' abs diff lower bounds min; breadth diff / min lower; feigner [lo,hi]", f"{_PR.naa_abs_in_range_lo.min():.1f}; {rng_(_PR.breadth_gap_in_range, 1)} / {_PR.gap_in_range_lo.min():.2f}; {ph_.breadth_gap_in_range:.2f} [{ph_.gap_in_range_lo:.2f}, {ph_.gap_in_range_hi:.2f}]")
show("in range (n>=10): abs diff exceeds the feigners' (p<0.05 raw / Holm over these)", f"{(_PR.abs_minus_feigner_p < 0.05).sum()} / {(_PR.p_holm10 < 0.05).sum()} of {len(_PR)}")
show("ratio vs feigners, Bonferroni lower bound > 0: models / exceptions / min over the rest", f"{(PL.feigner_minus_lo > 0).sum()} of {len(PL)} / {list(PL[PL.feigner_minus_lo <= 0].index)} / {PL[PL.feigner_minus_lo > 0].feigner_minus_lo.min():.2f}")
show("overall breadth gap lower bounds: min / which", f"{PL.gap_lo.min():.2f} / {PL.gap_lo.idxmin()}")
FCq = T("pcl5_feign_checks"); FCq = FCq[FCq.group != "human feigner"]
show("models: AUC with repeats split across folds (old pipeline)", rng_(FCq.auc_items_ungrouped))
show("distinct genuine patients per matching (models) / max reuse", f"{rng_(FCq.distinct_genuine, 0)} / {rng_(FCq.max_reuse, 0)}")
show("feign checks: AUC total only / ipsative models / ipsative people / overlap models", f"{rng_(T('pcl5_feign_checks').auc_total)} / {rng_(FCq.auc_ipsative)} / {T('pcl5_feign_checks').set_index('group').loc['human feigner'].auc_ipsative:.2f} / {rng_(FCq.auc_overlap.dropna())}")
for tag in ("gpt5mini", "gpt54mini"):
    d = pd.DataFrame([json.loads(l) for l in open(RUNS / f"pcl5__{tag}.jsonl")]); d = d[(d.status == "ok") & (d.arm == "feigner")].drop_duplicates("trial_id")
    tot = np.array(d.ordered.tolist()).sum(1)
    show(f"{tag} feigner trials: n, total<=5, total<=10", f"{len(tot)}, {(tot <= 5).sum()}, {(tot <= 10).sum()}")
FCk = T("pcl5_feign_checks"); FCm = FCk[FCk.group != "human feigner"]; FCh = FCk[FCk.group == "human feigner"].iloc[0]
show("feigners at matched totals: breadth / genuine; zero / genuine", f"{FCh.breadth:.1f} / {FCh.breadth_matched_genuine:.1f}; {FCh.zero:.0f} / {FCh.zero_matched_genuine:.0f}")
show("models: breadth / genuine at matched totals", f"{rng_(FCm.breadth, 1)} / {rng_(FCm.breadth_matched_genuine, 1)}")
show("models: not at all / genuine at matched totals", f"{rng_(FCm.zero, 0)} / {rng_(FCm.zero_matched_genuine, 0)}")
show("feign checks: total gap / exact match %", f"{rng_(FCk.total_gap, 1)} / {rng_(FCk.exact_pct, 0)}")
Cn = T("pcl5_triangle_ci"); show("PCL-5 permutation null mean (median over groups)", f"{Cn.null_mean.median():.2f}")
FA = T("filter_audit_runs"); pre = FA[~FA.group.str.startswith(("Extension", "Published personas"))]
show("filtered prompts before the extension (n / total / %)", f"{int(pre.prompt_filter.sum() + pre.response_filter.sum())} / {int(pre.total.sum())} / {100 * (pre.prompt_filter.sum() + pre.response_filter.sum()) / pre.total.sum():.2f}")
print("=== Mechanism ===")
LP = T("logprob_mechanism"); SEV = ["mild", "moderate", "moderately_severe", "severe"]
q = LP[(LP.frame == "label") & LP.severity.isin(SEV)]
show("log-prob models (n)", f"{q.model.nunique()}")
_wf = T("wholeform_items"); _wr = T("wholeform_items_real")
show("whole-form item entropy, all models / real (50 per band) range / mean", f"{len(_wf)}: {rng_(_wf.H)} / {rng_(_wr.H_real50)} / {_wr.H_real50.mean():.2f}")
show("whole-form P(not at all) at moderate, all models range / real", f"{rng_(_wf.P0_moderate)} / {_wr.set_index('band').P0_real['moderate']:.2f}")
show("whole-form independent draws, all models: distinct % mean over severities / max in any severity / breadth", f"{rng_(_wf.distinct, 0)} / {_wf.distinct_max.max():.0f} / {rng_(_wf.breadth, 1)}")
show("independent draws from real item distributions: distinct % / breadth; real breadth", f"{rng_(_wr.indep_distinct, 0)} / {rng_(_wr.indep_breadth, 1)}; {rng_(_wr.real_breadth, 1)}")
show("entropy model per-model mean range / human by-band range", f"{rng_(q.groupby('model').ent_model.mean())} / {rng_(q.groupby('severity').ent_human.first())}")
show("human entropy mean over bands", f"{q.groupby('severity').ent_human.first().mean():.2f}")
mo = q[q.severity == "moderate"]; show("P0 moderate: max model / human", f"{mo.p0_model.max():.3f} / {mo.p0_human.iloc[0]:.3f}")
LL = LP[LP.severity.isin(["mild", "moderate", "severe"])].groupby(["model", "frame"]).label_lock.mean().unstack()
show("label lock (label) / (impairment)", f"{rng_(LL.label * 100, 0)} / {rng_(LL.impairment * 100, 0)}")
SP = T("logprob_sampled_profiles"); HP = T("logprob_human_profiles")
show("sampled breadth by model", rng_(SP.groupby("model").breadth.mean(), 1)); show("sampled breadth by cell", rng_(SP.breadth, 1))
show("distinct by model / by cell / human", f"{rng_(SP.groupby('model').distinct.mean() * 100, 1)} / {rng_(SP.distinct * 100, 1)} / {rng_(HP.distinct * 100, 0)}")
PE = T("position_effects"); AN = T("anchoring").set_index("model")
show("label lock, real respondents by band (mild/moderate/severe) / mean", f"{' / '.join(f'{100 * v:.0f}' for v in HP.label_lock.dropna())} / {100 * HP.label_lock.mean():.0f}")
show("position slope mean", f"{PE.slope_per_position.mean():.4f}")
rest = AN.drop(["GPT-5-mini", "Kimi-K2.5"]); show("anchoring others / GPT-5-mini / Kimi", f"{rng_(rest.r_first_rest)} / {AN.loc['GPT-5-mini'].r_first_rest:.2f} / {AN.loc['Kimi-K2.5'].r_first_rest:.2f}")
cal = ph.set_index("model").calib; show("calib GPT-5-mini / Kimi / others", f"{cal['GPT-5-mini']:.1f} / {cal['Kimi-K2.5']:.1f} / {rng_(cal.drop(['GPT-5-mini', 'Kimi-K2.5']), 0)}")
print("=== Interventions / scale ===")
FXA = T("fixes"); FXR = FXA[FXA.cond == "-"].iloc[0]; FX = FXA[FXA.cond.isin(["label", "hetero", "biography", "numeric", "fewshot"])]
base = FX[FX.cond == "label"].set_index("model")
for c in ["hetero", "biography", "numeric", "fewshot"]:
    x = FX[FX.cond == c].set_index("model"); dd = (x.auc - base.auc).dropna()
    show(f"{c}: mean dAUC (n models)", f"{dd.mean():+.3f} ({len(dd)})")
fs = FX[FX.cond == "fewshot"].set_index("model")
show("llama label->fewshot AUC", f"{base.loc['llama3370binstruct'].auc:.3f} -> {fs.loc['llama3370binstruct'].auc:.3f}")
show("copy rate range / fewshot diversity / held-out real at the same totals / % of it", f"{rng_(fs.copy_rate, 1)} / {rng_(fs.diversity)} / {rng_(fs.diversity_h)} / {rng_(100 * fs.diversity / fs.diversity_h, 0)}")
_fxa = FXA[FXA.cond.isin(["label", "hetero", "biography", "numeric", "fewshot"])].pivot_table(index="model", columns="cond", values="auc")
show("fixes AUC: few-shot lowest (model, value); one-line least separable; does it rise under every change", f"{_fxa.fewshot.idxmin()} {_fxa.fewshot.min():.2f}; {_fxa.label.idxmin()} {_fxa.label.min():.2f}; {all((_fxa.loc[_fxa.label.idxmin(), c] > _fxa.label.min()) for c in ['hetero', 'biography', 'numeric', 'fewshot'] if pd.notna(_fxa.loc[_fxa.label.idxmin(), c]))}")
show("AUC chance (fixes)", rng_(FX.auc_null.dropna()))
FSm = T("fixes_summary"); FSm = FSm[FSm.measure == "auc"].set_index("cond")
show("dAUC vs one-line [95% t over models]", "; ".join(f"{c} {FSm.loc[c, 'mean']:+.2f} [{FSm.loc[c, 'lo']:+.2f}, {FSm.loc[c, 'hi']:+.2f}]" for c in ["hetero", "biography", "numeric", "fewshot"]))
_cal = AM[(AM.instrument == "phq9") & (AM.model != "REAL")].set_index("model").calib; _top = _cal.idxmax()
g5 = A.run_X({m[0]: m[1] for m in A.ALL_MODELS}[_top])
_cd = T("calibration_diversity"); _cb = _cd[(_cd.kind == "base") & (_cd.label == _top)].iloc[0]
show("best-calibrated model / calib / diversity 5-27 / real same totals / % (Fig 4e)", f"{_top} / {_cal.max():.1f} / {_cb['div']:.2f} / {_cb.div_h:.2f} / {_cb.divp:.0f}")
show("Fig 4e: diversity % of real at same totals, all configurations (min-max)", rng_(_cd.divp, 0))
d = pd.DataFrame([json.loads(l) for l in open(RUNS / "fixes__llama3370binstruct.jsonl")]); d = d[(d.status == "ok") & (d.arm == "numeric")].drop_duplicates("trial_id")
tot = np.array(d.ordered.tolist()).sum(1); show("Llama numeric totals mean/sd/IQR", f"{tot.mean():.1f} / {tot.std():.1f} / {np.percentile(tot, 25):.0f}-{np.percentile(tot, 75):.0f}")
print("=== Published personas (TalkDep, CACTUS) ===")
for c in ["label", "talkdep", "cactus", "cactus_cm"]:
    q = FXA[FXA.cond == c]
    show(f"{c}: diversity / AUC / N.a.a. 1-8 / SI / all9 (15-24)", f"{rng_(q.diversity)} / {rng_(q.auc)} / {rng_(q.zero18, 1)} / {rng_(q.si_band, 0)} / {rng_(q.all9_band, 0)}")
for c in ["label", "talkdep", "cactus", "cactus_cm"]:
    show(f"{c}: AUC items 1-8 (15-24)", rng_(FXA[FXA.cond == c].auc18))
show("CACTUS intake forms mentioning suicidality", T("published_personas_si_mentions").iloc[0].to_dict())
show("real held-out: diversity / N.a.a. 1-8 / SI / all9 (15-24)", f"{FXR.diversity:.2f} / {FXR.zero18:.1f} / {FXR.si_band:.0f} / {FXR.all9_band:.0f}")
FS = T("fixes_summary")
for c in ["talkdep", "cactus", "cactus_cm"]:
    r = FS[(FS.cond == c) & (FS.measure == "auc")].iloc[0]; show(f"{c}: mean dAUC vs one-line [95% t]", f"{r['mean']:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}]")
    r = FS[(FS.cond == c) & (FS.measure == "diversity")].iloc[0]; show(f"{c}: mean d diversity vs one-line [95% t]", f"{r['mean']:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}]")
q = FXA[FXA.cond.isin(["talkdep", "cactus", "cactus_cm"])]; show("published personas (incl. formulation): diversity as % of held-out real at the same totals", rng_(100 * q.diversity / q.diversity_h, 0))
TD = T("published_personas_talkdep"); tdm = TD.groupby("persona")[["bdi", "total_sd", "distinct", "key_endorsed", "other_endorsed"]].median()
tdm_r = TD.groupby("persona")[["bdi", "other_real"]].median()
show("TalkDep per persona (median over models): total s.d. / distinct %", f"{rng_(tdm.total_sd, 1)} / {rng_(tdm.distinct, 0)}")
show("TalkDep key endorsed / other endorsed (BDI>=12) / other (BDI<12)", f"{rng_(tdm.key_endorsed, 0)} / {rng_(tdm[tdm.bdi >= 12].other_endorsed, 0)} ({(tdm.bdi >= 12).sum()} personas) / {rng_(tdm[tdm.bdi < 12].other_endorsed, 0)}")
show("TalkDep 'other' endorsed, real at same total (BDI>=12) / (BDI<12)", f"{rng_(tdm_r[tdm_r.bdi >= 12].other_real, 0)} / {rng_(tdm_r[tdm_r.bdi < 12].other_real, 0)}")
show("TalkDep model-persona pairs with more symptoms endorsed than real at the same totals; exceptions", f"{(TD.breadth > TD.breadth_real).sum()} of {len(TD)}; " + ", ".join(f"{r.model} {r.persona} {r.breadth:.2f}/{r.breadth_real:.2f}" for _, r in TD[TD.breadth <= TD.breadth_real].iterrows()))
_mild = TD[TD.bdi < 12]; _mm = _mild.groupby("persona")[["breadth", "breadth_real", "at_one_real"]].median()
show("TalkDep mildest three: breadth model / real (medians); several days, every model / real", f"{rng_(_mm.breadth, 1)} / {rng_(_mm.breadth_real, 1)}; {_mild.at_one.min():.0f}--{_mild.at_one.max():.0f} / {rng_(_mm.at_one_real, 0)}")
q = FXA[FXA.cond == "cactus"]; show("CACTUS diversity as % of held-out real at the same totals", rng_(100 * q.diversity / q.diversity_h, 0))
q = FXA[FXA.cond == "cactus_cm"]; show("CACTUS + formulation diversity as % of held-out real at the same totals", rng_(100 * q.diversity / q.diversity_h, 0))
q = FXA[FXA.cond == "talkdep"]; show("TalkDep diversity as % of held-out real at the same totals", rng_(100 * q.diversity / q.diversity_h, 0))
_PA = T("published_personas")
for _ins in ("phq9", "gad7"):
    for _src in ("one-line", "talkdep", "cactus", "cactus_cm"):
        _q = _PA[(_PA.instrument == _ins) & (_PA.persona == _src)]
        if len(_q): show(f"all models, {_ins} {_src}: n models / AUC / div ratio (real/model) / all items model / real", f"{len(_q)} / {rng_(_q.auc)} / {rng_(_q.div_ratio, 1)} / {rng_(_q.all_items, 0)} / {rng_(_q.all_items_h, 0)}")
_pub = _PA[_PA.persona != "one-line"]
show("all models, published personas, both instruments: AUC range / min div ratio / rows with ratio <= 1", f"{rng_(_pub.auc)} / {_pub.div_ratio.min():.1f} / {(_pub.div_ratio <= 1).sum()} of {len(_pub)}")
import re as _re
from mental_act import published_personas as _PP
_SUI = _re.compile(r"suicid|self[- ]harm|kill (my|him|her)self|end (my|his|her) life|better off dead|want(s|ed)? to die|take (my|his|her) (own )?life|no reason to live|ending it", _re.I)
show("CACTUS formulations (form + thought + distortions) mentioning suicidality", f"{sum(bool(_SUI.search(t)) for t in _PP.cactus_formulations())} of 400")
print("=== GRM profiles fit on 2005-2012 ===")
GH = T("grm_heldout").set_index("source"); print(GH.round(3).to_string())
CH = T("coaching"); show("coaching drop within severity (all / excl GPT-5-mini)", f"{rng_(CH.drop_within_severity)} / {rng_(CH[CH.model != 'GPT-5-mini'].drop_within_severity, 1)}")
SC = T("scale_ci"); print(SC.round(2).to_string(index=False))
show("all models below real, ratio (same totals)", f"{(ph.diversity_h / ph.diversity).min():.1f}--{(ph.diversity_h / ph.diversity).max():.1f}")
print("=== Reasoning effort ===")
RE = T("reasoning_effort"); print(RE.round(2).to_string(index=False))
RR = T("reasoning_effort_refs"); RG = T("reasoning_effort_gemini")
import glob as _g
# effort runs beyond the main runs: the reasoning-off runs were copied to the main names (scripts/65)
_main_eff = lambda f: f.endswith(("__gpt5mini__minimal.jsonl", "__gemini25flash__none.jsonl", "__kimik25__none.jsonl", "__gemini25pro__budget128.jsonl"))
_q = sum(len({json.loads(l)["trial_id"] for l in open(f) if l.strip() and json.loads(l)["status"] == "ok"}) for f in _g.glob(str(RUNS / "eff__*.jsonl")) if "interview" not in f and not _main_eff(f))
_i = sum(len({json.loads(l)["trial_id"] for l in open(f) if l.strip() and json.loads(l)["status"] == "ok"}) for f in _g.glob(str(RUNS / "eff__interview*.jsonl")) if not _main_eff(f))
show("effort runs: questionnaire administrations / interviews", f"{_q} / {_i}")
_isp = lambda f: "__gemini25pro__" in f
_qp = sum(len({json.loads(l)["trial_id"] for l in open(f) if l.strip() and json.loads(l)["status"] == "ok"}) for f in _g.glob(str(RUNS / "eff__*.jsonl")) if "interview" not in f and not _main_eff(f) and _isp(f))
_ip = sum(len({json.loads(l)["trial_id"] for l in open(f) if l.strip() and json.loads(l)["status"] == "ok"}) for f in _g.glob(str(RUNS / "eff__interview*.jsonl")) if not _main_eff(f) and _isp(f))
show("  of which Gemini 2.5 Pro at its API default: administrations / interviews", f"{_qp} / {_ip}")
_sw = RE[RE.model.isin(["GPT-5-mini", "Kimi-K2.5", "Gemini 2.5 Pro"])]
show("effort: GPT-5-mini / Kimi / Gemini Pro main calib & anchoring; with reasoning calib range / max anchoring",
     ", ".join(f"{r.model} {r.calib:.0f}% r={r.anchoring:.2f}" for _, r in _sw[_sw.main].iterrows()) + f"; {rng_(_sw[~_sw.main].calib, 0)} / {_sw[~_sw.main].anchoring.max():.2f}")
show("effort: Gemini Flash anchoring range over settings", rng_(RE[RE.model == "Gemini 2.5 Flash"].anchoring))
show("effort: settings whose ratio differs from the feigners' (Bonferroni lower bound > 0); exceptions",
     f"{(RE.pcl5_feigner_minus_lo > 0).sum()} of {len(RE)}; " + ", ".join(f"{r.model} {r.effort}" for _, r in RE[RE.pcl5_feigner_minus_lo <= 0].iterrows()))
show("effort (Gemini): one-line -> five sheets AUC, Flash settings / Pro settings",
     " ; ".join(f"{m}: " + ", ".join(f"{r.effort} {r.fix_label_auc:.2f}->{r.fix_fewshot_auc:.2f}" for _, r in RG[RG.model == m].iterrows()) for m in RG.model.unique()))
show("effort: PHQ-9 all nine / GAD-7 all seven / PHQ-9 div / GAD-7 div", f"{rng_(RE.all_items, 0)} / {rng_(RE.gad7_all_items, 0)} / {rng_(RE.diversity)} / {rng_(RE.gad7_diversity)}")
show("effort: PHQ-9 AUC / GAD-7 AUC / nine-reference AUC / div ratio (estimates on >= 5 totals)", f"{rng_(RE.auc)} / {rng_(RE.gad7_auc)} / {rng_(RR.auc)} / {rng_(RR[RR.div_bins >= 5].div_ratio, 1)}")
show("effort: PCL-5 AUC; not significant", f"{rng_(RE.pcl5_auc)}; " + "; ".join(f"{r.model} {r.effort} {r.pcl5_auc:.2f} (p={r.pcl5_p:.3f})" for _, r in RE[RE.pcl5_p >= 0.05].iterrows()))
show("effort: PCL-5 not-at-all ratio range / max upper / feigner-minus lower bound min", f"{rng_(RE.pcl5_naa_ratio)} / {RE.pcl5_naa_hi.max():.2f} / {RE.pcl5_feigner_minus_lo.min():.2f}")
from mental_act.feign import GEN as _G2
_gx = pd.DataFrame([json.loads(l) for l in open(RUNS / "eff__pcl5__gemini25flash__none.jsonl")]); _gx = _gx[(_gx.status == "ok") & (_gx.arm == "patient")].drop_duplicates("trial_id")
show("effort: Gemini off patient-role median total / genuine >= 60 / its ratio upper bound", f"{np.median(np.array(_gx.ordered.tolist()).sum(1)):.0f} / {(_G2.sum(1) >= 60).sum()} / {RE[(RE.model == 'Gemini 2.5 Flash') & (RE.effort == 'off')].pcl5_naa_hi.iloc[0]:.2f}")
show("effort: ratio upper bounds excluding Gemini off", f"{RE[~((RE.model == 'Gemini 2.5 Flash') & (RE.effort == 'off'))].pcl5_naa_hi.max():.2f}")
show("effort: PCL-5 change in total, GPT-5-mini by effort / Gemini", f"{RE[RE.model == 'GPT-5-mini'].pcl5_dtotal.round(1).tolist()} / {rng_(RE[RE.model == 'Gemini 2.5 Flash'].pcl5_dtotal, 1)}; Kimi {RE[RE.model == 'Kimi-K2.5'].pcl5_dtotal.round(1).tolist()}")
show("effort: screening AUC / FPR", f"{rng_(RE.ds_auc)} / {rng_(RE.ds_fpr, 1)}")
show("effort (Gemini): fixes one-line AUC / few-shot AUC / TalkDep / CACTUS AUC / CACTUS SI", f"{rng_(RG.fix_label_auc)} / {rng_(RG.fix_fewshot_auc)} / {rng_(RG.talkdep_auc)} / {rng_(RG.cactus_auc)} / {rng_(RG.cactus_si, 0)}")
show("effort (Gemini): free all8 / real same totals; div / real; real-profile r / sampled r / SI agreement", f"{rng_(RG.iv_free_all8, 1)} / {rng_(RG.iv_free_all8_h, 0)}; {rng_(RG.iv_free_div)} / {rng_(RG.iv_free_div_h)}; {rng_(RG.iv_oracle_item_r)} / {rng_(RG.iv_grm_item_r)} / {rng_(pd.concat([RG.iv_oracle_si_agree, RG.iv_grm_si_agree]), 0)}")
print("=== Interviews ===")
IV = T("interview"); fr = IV[IV.source.str.startswith("free")]; orc = IV[IV.source.str.startswith("oracle")]; ref = IV.iloc[0]
show("free all8 / real all8", f"{rng_(fr.all8, 1)} / {ref.all8:.1f}")
show("free div / real div", f"{rng_(fr.diversity)} / {ref.diversity:.2f}")
f2 = fr[fr.source.str.split(": ").str[1].isin(set(orc.source.str.split(": ").str[1]))]; show("free div (oracle models) / oracle div / given div", f"{rng_(f2.diversity)} / {rng_(orc.diversity)} / {rng_(orc.true_div, 1)}")
show("oracle: rated div as % of given div, per model", rng_(100 * orc.diversity / orc.true_div, 0))
show("oracle item r / breadth rated / given / free", f"{rng_(orc.item_r)} / {rng_(orc.breadth8)} / {orc.true_breadth8.iloc[0]:.2f} / {rng_(f2.breadth8)}")
show("oracle SI given / rated (4o-mini) / free 4o-mini", f"{rng_(orc.true_si, 0)} / {orc[orc.source.str.contains('gpt4omini')].si.round(0).tolist()} / {fr[fr.source.str.contains('gpt4omini')].si.round(1).tolist()}")
show("given 1 rated 2 (%)", rng_(orc.given1_rated2, 0))
IG = T("interview_grm")
if len(IG):
    show("GRM interviews: patients / n per model", f"{IG.source.nunique()} / {IG.groupby('source').n.first().tolist()}")
    show("GRM given breadth / rated / oracle rated", f"{IG.true_breadth8.iloc[0]:.2f} / {rng_(IG.breadth8)} / {rng_(orc.breadth8)}")
    show("GRM given all8 / rated / oracle given / rated", f"{IG.true_all8.iloc[0]:.1f} / {rng_(IG.all8, 1)} / {orc.true_all8.iloc[0]:.1f} / {rng_(orc.all8, 1)}")
    show("GRM given div / rated div / oracle given / rated", f"{rng_(IG.true_div)} / {rng_(IG.diversity)} / {rng_(orc.true_div)} / {rng_(orc.diversity)}")
    show("GRM item r / total r / oracle item r", f"{rng_(IG.item_r)} / {rng_(IG.total_r)} / {rng_(orc.item_r)}")
    nl = IG[~IG.source.str.contains("llama")]
    show("GRM SI given / rated (excl. Llama)", f"{rng_(nl.true_si, 1)} / {rng_(nl.si, 1)}")
    show("GRM given1 rated2 / oracle", f"{rng_(IG.given1_rated2, 0)} / {rng_(orc.given1_rated2, 0)}")
    show("GRM excess breadth reduction vs free (%)", rng_(100 * (1 - (IG.breadth8 - IG.true_breadth8) / (8.0 - IG.true_breadth8)), 0))
IVf = IV[IV.source.str.startswith("free")]
show("free: all8 / real same totals; zero / real; div / real same totals", f"{rng_(IVf.all8, 1)} / {rng_(IVf.all8_h, 0)}; {rng_(IVf.zero, 1)} / {rng_(IVf.zero_h, 1)}; {rng_(IVf.diversity)} / {rng_(IVf.diversity_h)}")
show("free: real/model diversity ratio (same totals)", rng_(IVf.diversity_h / IVf.diversity, 0))
IVo = IV[IV.source.str.startswith("oracle")]; show("oracle SI per-interview agreement / kappa", f"{rng_(IVo.si_agree.dropna(), 0)} / {rng_(IVo.si_kappa.dropna())}")
if len(IG): show("GRM SI per-interview agreement / kappa", f"{rng_(IG.si_agree.dropna(), 0)} / {rng_(IG.si_kappa.dropna())}")
AG = T("interview_agreement"); show("rater within-1 agreement", rng_(AG.within1, 1))
SE = T("interview_sensitivity"); print(SE.round(2).to_string(index=False))
Xh = A.human_X("phq9"); mild = Xh[(Xh.sum(1) >= 5) & (Xh.sum(1) <= 9)]; show("real SI in mild band", f"{100 * (mild[:, 8] > 0).mean():.1f}")
for tag, rater in (("gpt4omini", "gpt-4.1"), ("gpt41", "claude-haiku-4-5"), ("claudehaiku45", "gpt-4.1")):
    d = pd.DataFrame([json.loads(l) for l in open(RUNS / f"interview__free__{tag}.jsonl")]); d = A.drop_filtered_interviews(d[d.status == "ok"].drop_duplicates("trial_id"))
    R = np.array(d[f"rated__{rater}"].tolist()); show(f"mild SI free {tag}", f"{100 * (R[(d.severity == 'mild').values][:, 8] > 0).mean():.0f}")
# Figure 5b, c: same loading as scripts/fig_L5 (GPT-4.1 transcripts rated by Haiku, others by GPT-4.1)
from mental_act import figlong as _FL
_HAS = lambda cond, t: (RUNS / f"interview__{cond}__{t}.jsonl").exists()
_PAT = [m[3] for m in _FL.MODELS if _HAS("free", m[3])]; _RAT = {t: "claude-haiku-4-5" if t == "gpt41" else "gpt-4.1" for t in _PAT}
def _ld(cond, tag):
    d = pd.DataFrame([json.loads(l) for l in open(RUNS / f"interview__{cond}__{tag}.jsonl")])
    d = A.drop_filtered_interviews(d[d.status == "ok"].drop_duplicates("trial_id"))
    ok = np.array(["DPQ090" not in (x if isinstance(x, list) else []) for x in d.get("filtered_items", [[]] * len(d))])
    return d, np.array(d[f"rated__{_RAT[tag]}"].tolist(), int), ok
_FR = {t: _ld("free", t) for t in _PAT}; _OR = {t: _ld("oracle", t) for t in _PAT if _HAS("oracle", t)}
_SIO = [t for t in _PAT if _FR[t][2].mean() > 0.8]
_si = {t: {s_: 100 * (R[(d.severity == s_).values & ok][:, 8] > 0).mean() for s_ in ["mild", "moderate", "moderately_severe", "severe"]} for t, (d, R, ok) in ((t, _FR[t]) for t in _SIO)}
_mild = {_FL.TAG2LABEL[t]: round(v["mild"]) for t, v in _si.items()}
show("Fig 5b: mild SI by model (min, max) / moderate and above, min-max", f"{min(_mild.items(), key=lambda x: x[1])}, {max(_mild.items(), key=lambda x: x[1])} / {min(min(v[k] for k in ['moderate', 'moderately_severe', 'severe']) for v in _si.values()):.0f}--{max(max(v[k] for k in ['moderate', 'moderately_severe', 'severe']) for v in _si.values()):.0f}")
show("free SI (all severities) for the oracle models with SI observable", rng_(pd.Series([100 * (R[ok][:, 8] > 0).mean() for t, (d, R, ok) in _FR.items() if t in _OR and t in _SIO]), 0))
_T = np.vstack([np.array(_OR[t][0].true_profile.tolist(), int)[:, :8] for t in _OR]); _Rt = np.vstack([_OR[t][1][:, :8] for t in _OR])
show("Fig 5c: given 'several days' rated one level higher, pooled (%)", f"{100 * np.mean(_Rt[_T == 1] == 2):.0f}")
_GR = T("interview_given_rated"); _GO = _GR[_GR.cond == "oracle"]; _GG = _GR[_GR.cond == "grm"]
show("given vs rated (real profile; both raters): exact / within 1 / weighted kappa / shift", f"{rng_(_GO.exact, 0)} / {rng_(_GO.within1, 0)} / {rng_(_GO.qwk)} / {rng_(_GO['shift'])}")
show("given vs rated (sampled profile; both raters): exact / within 1 / weighted kappa / shift", f"{rng_(_GG.exact, 0)} / {rng_(_GG.within1, 0)} / {rng_(_GG.qwk)} / {rng_(_GG['shift'])}")
show("presence: present->present / absent->absent / exact if present / several days rated higher (all, both raters)", f"{rng_(_GR.presence_sens, 0)} / {rng_(_GR.presence_spec, 0)} / {rng_(_GR.exact_present, 0)} / {rng_(_GR.up_given1, 0)}")
print("=== Downstream ===")
DS = T("downstream_si").set_index("train"); sim = DS[DS.kind == "sim"]; RL, GR, PO = DS.loc["Real (NHANES)"], DS.loc["GRM-sampled profiles"], DS.loc["All models pooled"]
show("test n (2013-2023, items 1-8 >= 5) / SI %", f"{int(RL.n_test)} / {RL.observed:.1f}")
show("real-trained: n / AUC / Brier / slope / FPR / sens", f"{int(RL.n_train)} / {RL.auc:.2f} / {RL.brier:.3f} / {RL.slope:.2f} / {RL.fpr:.1f} / {RL.sens:.1f}")
show("GRM-trained: AUC / Brier / slope / FPR / sens", f"{GR.auc:.2f} / {GR.brier:.3f} / {GR.slope:.2f} / {GR.fpr:.1f} / {GR.sens:.1f}")
show("sim: n / SI% in training", f"{rng_(sim.n_train, 0)} / {rng_(sim.train_pos, 1)}")
show("sim: AUC / Brier / slope / FPR / sens", f"{rng_(sim.auc)} / {rng_(sim.brier, 3)} / {rng_(sim.slope)} / {rng_(sim.fpr, 1)} / {rng_(sim.sens, 1)}")
show("pooled: AUC / Brier / slope", f"{PO.auc:.2f} / {PO.brier:.3f} / {PO.slope:.2f}")
show("recalibrated Brier: sim / real / GRM", f"{rng_(sim.recal_brier, 3)} / {RL.recal_brier:.3f} / {GR.recal_brier:.3f}")
DC = T("downstream_calibration"); print(DC[["lo", "hi", "n", "n_si", "observed", "pred_real", "pred_GRM", "pred_LLM pooled"]].round(1).to_string(index=False))
pm = [c for c in DC.columns if c.startswith("pred_") and c not in ("pred_real", "pred_GRM", "pred_LLM pooled")]
low, high = DC[DC.hi <= 9], DC[DC.lo >= 15]
show("below 10: observed with SI / expected (pooled sim); pred % per model max", f"{int(low.n_si.sum())} / {(low['pred_LLM pooled'] / 100 * low.n).sum():.0f}; {low[pm].max().max():.1f}")
show("15 and above: observed with SI / expected (pooled sim)", f"{int(high.n_si.sum())} / {(high['pred_LLM pooled'] / 100 * high.n).sum():.0f}")
show("pred above 15, per model (min-max) / observed", f"{high[pm].min().min():.0f}--{high[pm].max().max():.0f} / {rng_(high.observed, 1)}")
CT = T("downstream_controls").set_index("train"); cs = CT.drop("All models pooled"); cp = CT.loc["All models pooled"]
show("controls AUC: sim / random / same totals / same totals+item 9", f"{rng_(cs.auc_sim)} / {rng_(cs.auc_rand)} / {rng_(cs.auc_sev)} / {rng_(cs.auc_sevlab)}")
show("SI% in training: sim / same-totals real", f"{rng_(cs.sim_pos, 0)} / {rng_(cs.pos_sev, 0)}")
show("vs same totals: n below 0 / n CI below 0 / n CI above 0", f"{(cs.d_sev < 0).sum()} / {(cs.d_sev_hi < 0).sum()} / {(cs.d_sev_lo > 0).sum()} of {len(cs)}")
show("vs same totals: range / pooled [CI]", f"{rng_(cs.d_sev, 3)} / {cp.d_sev:+.3f} [{cp.d_sev_lo:+.3f}, {cp.d_sev_hi:+.3f}]")
show("vs same totals+item 9: n CI below 0 / n CI above 0", f"{(cs.d_sevlab_hi < 0).sum()} / {(cs.d_sevlab_lo > 0).sum()} of {len(cs)}")
show("vs same totals+item 9: range / pooled [CI]", f"{rng_(cs.d_sevlab, 3)} / {cp.d_sevlab:+.3f} [{cp.d_sevlab_lo:+.3f}, {cp.d_sevlab_hi:+.3f}]")
show("Brier: sim recal / same-totals real; n recal-minus-real CI includes 0", f"{rng_(cs.brier_sim_recal, 3)} / {rng_(cs.brier_sev, 3)}; {((cs.d_brier_recal_sev_lo <= 0) & (cs.d_brier_recal_sev_hi >= 0)).sum()} of {len(cs)}")
show("calibration slope: sim / recalibrated / same-totals real", f"{rng_(cs.slope_sim)} / {rng_(cs.slope_sim_recal)} / {rng_(cs.slope_sev)}")
AU = T("downstream_augmentation"); print(AU.round(3).to_string(index=False))
show("augmentation: dAUC range / n CI>0", f"{rng_(AU.d_auc, 3)} / {(AU.d_auc_lo > 0).sum()} of {len(AU)}")
show("augmentation: dBrier x1e3 range / smallest real n with every CI > 0", f"{rng_(1000 * AU.d_brier, 1)} / {AU[AU.groupby('n_real').d_brier_lo.transform('min') > 0].n_real.min()}")
CD = T("calibration_diversity"); from scipy.stats import spearmanr
show("calibration vs diversity (% of real at same totals), all configurations: n / Spearman rho [p]", f"{len(CD)} / {spearmanr(CD.calib, CD.divp).statistic:.2f} [{spearmanr(CD.calib, CD.divp).pvalue:.2f}]")
print("=== Samples / filter ===")
nh = pd.read_csv(DATA_PROC / "nhanes_phq9.csv"); t = A.human_X("phq9").sum(1)
show("NHANES N, mean, sd, %>=10", f"{len(nh)}, {t.mean():.2f}, {t.std(ddof=1):.2f}, {100 * (t >= 10).mean():.2f}")
show("NHIS GAD-7 N / PHQ-8 N", f"{len(pd.read_csv(DATA_PROC / 'nhis_gad7.csv'))} / {len(pd.read_csv(DATA_PROC / 'nhis_phq8.csv'))}")
for k in ("k6", "ocir", "phq15", "itq", "edeq", "phq9_coach", "pcl5c"):
    x = pd.read_csv(DATA_PROC / f"extra_{k}.csv"); show(f"extra {k} N", len(x))
co = pd.read_csv(DATA_PROC / "extra_phq9_coach.csv"); pc5 = pd.read_csv(DATA_PROC / "extra_pcl5c.csv")
show("COACH mean total / PCL5 postpartum median", f"{co.total.mean():.1f} / {pc5.total.median():.0f}")
FA = T("filter_audit_runs"); ed = FA[FA.group == "Extension: EDE-Q"]
show("EDE-Q OpenAI prompt-filter range", rng_(ed[ed.prompt_filter > 0].prompt_filter, 0))
for f in sorted(glob.glob(str(RUNS / "x_edeq__gpt*.jsonl"))):
    d = pd.DataFrame([json.loads(l) for l in open(f)]).drop_duplicates("trial_id", keep="last")
    ok = d.status.eq("ok"); show(f"  {f.split('__')[1][:-6]} blocked % by severity", (100 * (~ok).groupby(d.severity).mean()).round(0).to_dict())
tot_admin = FA[~FA.group.isin(["Log-prob probes"])].total.sum()
_pp = FA[FA.group.str.startswith("Published personas")]; show("published-persona runs: administrations / filtered / other failures", f"{int(_pp.total.sum())} / {int((_pp.prompt_filter + _pp.response_filter).sum())} / {int(_pp.other.sum())}")
show("questionnaire administrations (all runs except log-prob)", tot_admin)
print("=== Counts over models (for 'N of M models' statements) ===")
from mental_act.figlong import MODELS as FM
inv = {m[3]: m[0] for m in FM}; NM = len(FM)
TR = T("pcl5_triangle"); pat = TR[TR.arm == "patient"].set_index("model"); fei = TR[TR.arm == "feigner"].set_index("model")
show("PRI: models with 0% flagged as patients", f"{(pat.pri_flag == 0).sum()} of {NM}; others: " + ", ".join(f"{inv[t]} {v:.1f}" for t, v in pat.pri_flag.items() if v > 0))
d = pd.DataFrame({"dtot": fei.total - pat.total, "dext": fei.top - pat.top})
both = d[(d.dtot > 0) & (d.dext > 0)]
show("feign: models raising both total and extreme", f"{len(both)} of {NM}; median dtot {d.dtot.median():+.1f}, dext {d.dext.median():+.1f}")
show("feign: models not raising both", "; ".join(f"{inv[t]} {r.dtot:+.1f}/{r.dext:+.1f}" for t, r in d.drop(both.index).iterrows()))
for tag in d.index:
    f = RUNS / f"pcl5__{tag}.jsonl"; dd = pd.DataFrame([json.loads(l) for l in open(f)])
    dd = dd[(dd.status == "ok") & (dd.arm == "feigner")].drop_duplicates("trial_id")
    tot = np.array(dd.ordered.tolist()).sum(1)
    if (tot <= 2).sum() >= 3: show(f"  {inv[tag]} feigner trials with total <= 2", f"{(tot <= 2).sum()} of {len(tot)} (total 0: {(tot == 0).sum()})")
show("anchoring: models with r >= 0.2", f"{(AN.r_first_rest >= 0.2).sum()} of {len(AN)}; below: " + ", ".join(f"{m} {v:.2f}" for m, v in AN.r_first_rest.items() if v < 0.2))
show("anchoring range among r >= 0.2", rng_(AN.r_first_rest[AN.r_first_rest >= 0.2]))
show("calibration sorted (top 4)", ", ".join(f"{m} {v:.1f}" for m, v in cal.sort_values(ascending=False).head(4).items()))
show("coaching: models where coaching lowers breadth", f"{(CH.drop_within_severity > 0).sum()} of {len(CH)}; range {rng_(CH[CH.drop_within_severity > 0].drop_within_severity, 2)}; others: " + ", ".join(f"{r.model} {-r.drop_within_severity:+.2f}" for _, r in CH[CH.drop_within_severity <= 0].iterrows()))
for ins in ("phq9", "gad7"):
    q = PF[PF.instrument == ins]; show(f"person fit {ins}: lz<=0 models", ", ".join(q[q.lz <= 0].model))
show("models per table (all_models phq9)", len(ph))
print("=== Sixth review: new checks ===")
_AM = T("all_models"); _p9 = _AM[(_AM.instrument == "phq9") & (_AM.model != "REAL")]
show("PHQ-9 totals 15-24: all of items 1-8 endorsed, models / real at the same totals", f"{rng_(_p9.all8, 1)} / {rng_(_p9.all8_h, 0)}")
_SC = T("pcl5_scales").set_index("group"); _sm = _SC.drop("human feigner"); _sf = _SC.loc["human feigner"]
show("PCL-5 'not at all' odds ratio vs genuine at the same total: feigners [CI] / models (min-max) / models with no 'not at all'",
     f"{_sf['or']:.2f} [{_sf.or_lo:.2f}, {_sf.or_hi:.2f}] / {rng_(_sm['or'])} / {list(_sm[_sm.none_zero].index)}")
show("odds ratio below the feigners after Holm (all totals) / in the feigners' range (reweighted), n",
     f"{(_sm.diff_p_holm < 0.05).sum()} of {len(_sm)} / {(_sm.diff_range_p_holm < 0.05).sum()} of {_sm.diff_range_p_holm.notna().sum()}")
show("in-range odds ratio: feigners / models (min-max)", f"{_sf.or_range:.2f} / {rng_(_sm.or_range.dropna(), 3)}")
show("AUC vs genuine-genuine at the same totals (halvings): group / real (min-max); models p<0.05 unadjusted / Holm",
     f"{rng_(_sm.base_group)} / {rng_(_sm.base_real)}; {(_sm.base_p < 0.05).sum()} / {(_sm.base_p_holm < 0.05).sum()} of {len(_sm)}")
_WF = T("wholeform_items")
show("whole forms: entropy models / real at the same totals", f"{rng_(_WF.H)} / {rng_(_WF.Ht)}")
show("entropy bound: 2^(9 x mean item entropy), models / real at the same totals", f"{2 ** (9 * _WF.H.min()):.0f}--{2 ** (9 * _WF.H.max()):.0f} / {2 ** (9 * _WF.Ht.min()):,.0f}--{2 ** (9 * _WF.Ht.max()):,.0f}")
show("whole forms: P(not at all) at moderate, models / real at the same totals; models below their own reference",
     f"{rng_(_WF.P0_moderate)} / {rng_(_WF.P0t_moderate)}; {(_WF.P0_moderate < _WF.P0t_moderate).sum()} of {len(_WF)}")
show("whole forms: modal answer = level, models / real at the same totals; models above their own reference",
     f"{rng_(100 * _WF.lock, 0)} / {rng_(100 * _WF.lockt, 0)}; {(_WF.lock > _WF.lockt).sum()} of {len(_WF)}")
_LPm = [TAG2[t] for t in T("logprob_mechanism").model.unique() if t in TAG2] if "TAG2" in dir() else None
_lpl = {m[0] for m in __import__("mental_act.figlong", fromlist=["MODELS"]).MODELS if m[3] in set(T("logprob_mechanism").model)}
show("log-prob models: real modal = level at the same totals as their whole forms", rng_(100 * _WF[_WF.model.isin(_lpl)].lockt, 0))
show("item-by-item draws: distinct % models / real at the same totals", f"{rng_(_WF.distinct, 0)} / {rng_(_WF.distinctt, 0)}")
_BL = T("separability_baselines"); show("real-vs-real AUC, Table 1 protocol", "; ".join(f"{r.pair}: {r.auc:.2f}" for _, r in _BL.iterrows()))
_WB = T("separability_wide_bins"); show("wide bins: item AUC / total-only AUC / residualised item AUC (all three refs)", f"{rng_(_WB.auc_items)} / {rng_(_WB.auc_total_only)} / {rng_(_WB.auc_residual)}")
_OH = T("separability_onehot")
for _s in ("phq9", "gad7"):
    _q = _OH[_OH.set == _s]; show(f"one-hot AUC {_s}: linear / one-hot", f"{rng_(_q.auc_linear)} / {rng_(_q.auc_onehot)}")
show("one-hot AUC feigners / GRM", f"{_OH[_OH.set == 'pcl5'].auc_onehot.iloc[0]:.2f} / {_OH[_OH.set == 'phq9_grm'].auc_onehot.iloc[0]:.2f} (linear {_OH[_OH.set == 'pcl5'].auc_linear.iloc[0]:.2f} / {_OH[_OH.set == 'phq9_grm'].auc_linear.iloc[0]:.2f})")
_IP = T("interview_pipeline"); _ipo = _IP.dropna(subset=["pipe_ratio"])
show("interviews vs real-profile interviews (same pipeline, bins of 3): diversity ratio / questionnaire ratio same models / all 7 interview models",
     f"{rng_(_ipo.pipe_ratio, 1)} / {rng_(_ipo.q_ratio, 1)} / {rng_(_IP.q_ratio, 1)}")
show("interviews: all eight endorsed, severity persona / real profile through the pipeline (same bins)", f"{rng_(_ipo.all8_free_b, 0)} / {rng_(_ipo.all8_oracle_b, 0)}")
show("interviews: AUC severity persona / real profile / sampled profile", f"{rng_(_IP.auc_free)} / {rng_(_ipo.auc_oracle)} / {rng_(_ipo.auc_grm.dropna())}")
_BS = T("by_severity_models"); show("severity bands: real diversity at the models' totals, median per band", _BS.groupby("band").diversity_h.median().round(2).to_dict())
print("=== Human ratings of interview transcripts ===")
_HS = T("human_rating_summary"); _hv = lambda b, s, m: _HS[(_HS.block == b) & (_HS.source == s) & (_HS.metric == m)].value.iloc[0]
_hc = lambda s, m: _HS[(_HS.block == "interval") & (_HS.source == s) & (_HS.metric == m)][["lo", "hi"]].iloc[0]
_ag = _HS[(_HS.block == "agreement") & _HS.source.str.startswith("all")].set_index("metric").value
show("raters / pairwise exact (min-max) / pairwise weighted kappa (min-max)", f"{_HS[_HS.block == 'given'].source.str.startswith('human R').sum() // 8} / {_ag.exact_min:.0f}--{_ag.exact_max:.0f} / {_ag.qwk_min:.2f}--{_ag.qwk_max:.2f}")
_c1, _c2 = _hc("humans pooled", "several_up"), _hc("LLM rater", "several_up")
show("several days rated higher: humans pooled [CI] / model rater [CI] / per human rater",
     f"{_hv('given', 'humans pooled', 'several_up'):.0f} [{_c1.lo:.0f}, {_c1.hi:.0f}] / {_hv('given', 'LLM rater', 'several_up'):.0f} [{_c2.lo:.0f}, {_c2.hi:.0f}] / "
     f"{rng_(_HS[(_HS.block == 'given') & _HS.source.str.startswith('human R') & (_HS.metric == 'several_up')].value, 0)}")
show("exact vs given: humans pooled / model rater; shift humans / model", f"{_hv('given', 'humans pooled', 'exact'):.0f} / {_hv('given', 'LLM rater', 'exact'):.0f}; {_hv('given', 'humans pooled', 'shift'):+.2f} / {_hv('given', 'LLM rater', 'shift'):+.2f}")
show("severity-persona interviews rated all eight, every human rater / model rater; n", f"{rng_(_HS[(_HS.block == 'free') & _HS.source.str.startswith('human R') & (_HS.metric == 'all8')].value, 0)} / {_hv('free', 'LLM rater', 'all8'):.0f}; {_hv('free', 'LLM rater', 'n'):.0f}")
show("unsure: all ratings / given several days", f"{_hv('unsure', 'humans pooled', 'all_items'):.0f} / {_hv('unsure', 'humans pooled', 'given_several_days'):.0f}")
print("=== Assessment agent (interviewer + model rater) ===")
_AE = T("agent_eval"); _AE = _AE[_AE.primary]
for _c in ("free", "oracle", "grm"):
    _q = _AE[_AE.cond == _c]; show(f"{_c}: models / exact / within one band / above / below", f"{len(_q)} / {rng_(_q.exact, 0)} / {rng_(_q.within1, 0)} / {rng_(_q['above'], 0)} / {rng_(_q.below, 0)}")
_o = _AE[_AE.cond == "oracle"]
show("real profile: given absent / absence detected / presence agreement / all-present agent agreement", f"{rng_(_o.given_absent, 1)} / {rng_(_o.absent_detect, 0)} / {rng_(_o.agree, 0)} / {rng_(_o.agree_allpresent, 1)}")
show("told a severity: symptoms 1-8 rated absent (%)", rng_(_AE[_AE.cond == "free"].absent, 1))
_CF = T("agent_eval_confusion")
for _c, _t, _r in (("free", "mild", "moderately_severe"), ("free", "moderate", "severe"), ("free", "moderately_severe", "severe")):
    _q = _CF[(_CF.cond == _c) & (_CF.truth == _t)]; show(f"{_c}: told {_t}, rated {_r} (%)", f"{100 * _q[_q.rated == _r].n.sum() / _q.n.sum():.0f}")
_q = _CF[_CF.cond == "free"]; _BO = ["none", "mild", "moderate", "moderately_severe", "severe"]
show("free: rated above the told band, mild / moderate / mod. severe / severe (%)", " / ".join(f"{100 * _q[(_q.truth == b) & (_q.rated.map(_BO.index) > _BO.index(b))].n.sum() / _q[_q.truth == b].n.sum():.0f}" for b in _BO[1:]))
show("free: placed in the told band, mild / moderate / mod. severe / severe (%)", " / ".join(f"{100 * _q[(_q.truth == b) & (_q.rated == b)].n.sum() / _q[_q.truth == b].n.sum():.0f}" for b in _BO[1:]))
_AH = T("agent_eval_human"); _hh = _AH[_AH.source.str.match(r"R\d")]
show("64 transcripts, told a severity: human raters, symptoms 1-8 rated absent (%) per rater", rng_(_hh[_hh.cond == "free"].absent, 1))
for _c in ("free", "oracle"):
    show(f"64 transcripts, {_c}: above, human mean (range) / model rater", f"{_hh[_hh.cond == _c]['above'].mean():.0f} ({rng_(_hh[_hh.cond == _c]['above'], 0)}) / {_AH[(_AH.source == 'LLM rater') & (_AH.cond == _c)]['above'].iloc[0]:.0f}")
print("=== Severity overshoot on the questionnaire (PHQ-9) ===")
_SO = T("severity_overshoot"); _SI = _SO[(_SO.framing == "impairment") & (_SO.arm == "naive")]; _SO = _SO[_SO.framing == "label"]
show("impairment framing (no severity word), naive: above mild band % / median total, per model", "; ".join(f"{m}: {_SI[(_SI.model == m) & (_SI.severity == 'mild')]['above'].iloc[0]:.0f} / {_SI[(_SI.model == m) & (_SI.severity == 'mild')].median_total.iloc[0]:.0f}" for m in _SI.model.unique()))
_FX = T("fixes").pivot_table(index="model", columns="cond", values="calib")
show("% in the instructed band: severity label / target total (7 models); models >= 99 (rounded) with a target total", f"{rng_(_FX['label'], 0)} / {rng_(_FX['numeric'], 0)}; {(_FX['numeric'] >= 98.5).sum()}")
_sn = _SO[(_SO.arm == "naive") & (_SO.severity != "severe")]
show("naive, mild-mod. severe: above the band per model (mean of 3 bands) range / median", f"{rng_(_sn.groupby('model')['above'].mean(), 0)} / {_sn.groupby('model')['above'].mean().median():.0f}")
show("coached: same, median over models", f"{_SO[(_SO.arm == 'coached') & (_SO.severity != 'severe')].groupby('model')['above'].mean().median():.0f}")
show("naive: median total per severity, median over models (mild / moderate / mod. severe / severe)", " / ".join(f"{_SO[(_SO.arm == 'naive') & (_SO.severity == b)].median_total.median():.0f}" for b in ("mild", "moderate", "moderately_severe", "severe")))
show("naive: % above the band, median over models (mild / moderate / mod. severe)", " / ".join(f"{_SO[(_SO.arm == 'naive') & (_SO.severity == b)]['above'].median():.0f}" for b in ("mild", "moderate", "moderately_severe")))
show("naive mild: median total, median over models (range)", f"{_SO[(_SO.arm == 'naive') & (_SO.severity == 'mild')].median_total.median():.0f} ({rng_(_SO[(_SO.arm == 'naive') & (_SO.severity == 'mild')].median_total, 0)})")
print("=== Residual structure at a fixed total ===")
_RS = T("residual_structure").set_index("group"); _rm = _RS.drop(["NHANES", "NHANES halves", "GRM-sampled profiles"])
show("NHANES contrast / halves similarity / GRM contrast, similarity", f"{_RS.loc['NHANES'].contrast:.2f} / {_RS.loc['NHANES halves'].similarity:.2f} / {_RS.loc['GRM-sampled profiles'].contrast:.2f}, {_RS.loc['GRM-sampled profiles'].similarity:.2f}")
show("similarity: models / real at the same totals (mean) / models below the matched 2.5th percentile", f"{rng_(_rm.similarity)} / {rng_(_rm.matched_similarity)} / {(_rm.similarity < _rm.matched_similarity_lo).sum()} of {len(_rm)}")
show("contrast: models / real at the same totals / models whose upper bound is below the matched mean / lower bound above", f"{rng_(_rm.contrast)} / {rng_(_rm.matched_contrast)} / {(_rm.contrast_hi < _rm.matched_contrast).sum()} / {(_rm.contrast_lo > _rm.matched_contrast).sum()} ({', '.join(_rm.index[_rm.contrast_lo > _rm.matched_contrast])})")
_RI = T("residual_structure_interviews").set_index("group")
for _g in ("real profiles", "sampled profiles"):
    _r = _RI.loc[_g]; show(f"interviews, {_g}: profiles / given contrast, similarity / rated contrast [CI], similarity [CI] / 80-real similarity mean [2.5, 97.5]",
                          f"{int(_r.n_profiles)} / {_r.given_contrast:.2f}, {_r.given_similarity:.2f} / {_r.rated_contrast:.2f} [{_r.rated_contrast_lo:.2f}, {_r.rated_contrast_hi:.2f}], {_r.rated_similarity:.2f} [{_r.rated_similarity_lo:.2f}, {_r.rated_similarity_hi:.2f}] / {_r.ref80_similarity:.2f} [{_r.ref80_similarity_lo:.2f}, {_r.ref80_similarity_hi:.2f}]")
_RP = T("residual_structure_pairs").set_index(["item_a", "item_b"])
for _pr in (("sleep", "fatigue"), ("mood", "suicidal"), ("mood", "concentration"), ("interest", "concentration")):
    _r = _RP.loc[_pr]; show(f"pair {_pr}: NHANES / matched real / models median / models below matched", f"{_r.nhanes:.2f} / {_r.matched_real:.2f} / {_r.models_median:.2f} / {int(_r.models_below)} of 17")
show("models + noise to real diversity: contrast / similarity", f"{rng_(_rm.noise_contrast)} / {rng_(_rm.noise_similarity)}")
print("=== Review round 9 (2026-10-11): structure-free reference, TalkDep interviews, reweighted human ratings ===")
show("no covariation (model item distributions): similarity / contrast range; models above its 97.5th pct",
     f"{rng_(_rm.null_own_similarity)} / {rng_(_rm.null_own_contrast)}; {(_rm.similarity > _rm.null_own_similarity_hi).sum()} of {len(_rm)}")
show("GRM at the models' sizes and totals: similarity range", rng_(_rm.null_grm_similarity))
_IT = T("interview_talkdep"); _prim = lambda src: "claude-haiku-4-5" if src.endswith("gpt41") else "gpt-4.1"
_IT = _IT[[r.rater == _prim(r.source) for _, r in _IT.iterrows()]]
show("TalkDep interviews (primary rater): all 8, labels moderate+ / real at same totals", f"{rng_(_IT.all8_modplus, 0)} / {rng_(_IT.all8_modplus_h, 0)}")
show("TalkDep interviews: diversity / real at same totals", f"{rng_(_IT.diversity)} / {rng_(_IT.diversity_h)}")
_IPt = T("interview_pipeline")
show("TalkDep vs real-profile interviews, same bins: diversity ratio / all 8 TalkDep / all 8 real profile",
     f"{rng_(_IPt.pipe_ratio_talkdep.dropna(), 1)} / {rng_(_IPt.all8_talkdep_b.dropna(), 0)} / {rng_(_IPt.all8_oracle_tb.dropna(), 0)}")
_AE = T("agent_eval"); _AEt = _AE[_AE.primary & (_AE.cond == "talkdep")]
show("agent vs TalkDep label (primary rater): exact / within 1 / above", f"{rng_(_AEt.exact, 0)} / {rng_(_AEt.within1, 0)} / {rng_(_AEt.above, 0)}")
show("agent: Spearman with BDI-II, interviews / persona means", f"{rng_(_AEt.bdi_rho)} / {rng_(_AEt.bdi_rho_persona)}")
_AEf = _AE[_AE.primary & (_AE.cond.isin(["free", "oracle"]))]
show("agent exact: told a severity / real profile (primary rater)", f"{rng_(_AEf[_AEf.cond == 'free'].exact, 0)} / {rng_(_AEf[_AEf.cond == 'oracle'].exact, 0)}")
from mental_act.published_personas import talkdep_severity as _tds
_td = pd.DataFrame([(n, b, l) for n, (b, l) in _tds().items()], columns=["persona", "bdi", "band"])
show("TalkDep BDI-II by label band", "; ".join(f"{b}: {q.bdi.min()}-{q.bdi.max()}" for b, q in _td.groupby("band")))
_k = pd.read_csv(ROOT / "outputs" / "human_rating" / "key" / "key.csv") if "ROOT" in dir() else None
from mental_act.config import ROOT as _ROOT
_k = pd.read_csv(_ROOT / "outputs" / "human_rating" / "key" / "key.csv")
show("human-rated severity-persona transcripts: n, severe", f"{(_k.cond == 'free').sum()}, {((_k.cond == 'free') & (_k.severity == 'severe')).sum()}")
for _c in ("free", "oracle"):
    _h = _hh[_hh.cond == _c]; _l = _AH[(_AH.source == "LLM rater") & (_AH.cond == _c)].iloc[0]
    show(f"64 transcripts, {_c}, severities weighted equally: exact human mean (range) / model; above human mean / model",
         f"{_h.exact_bal.mean():.0f} ({rng_(_h.exact_bal, 0)}) / {_l.exact_bal:.0f}; {_h.above_bal.mean():.0f} / {_l.above_bal:.0f}")
show("64 transcripts, free, unweighted exact: human mean / model", f"{_hh[_hh.cond == 'free'].exact.mean():.0f} / {_AH[(_AH.source == 'LLM rater') & (_AH.cond == 'free')].exact.iloc[0]:.0f}")
_PP = T("published_personas"); _pc = _PP[(_PP.instrument == "phq9") & (_PP.persona == "cactus")]
show("CACTUS, PHQ-9, 17 models: % distinct answer patterns / real at the same totals", f"{rng_(_pc.distinct, 0)} / {rng_(_pc.distinct_h, 0)}")
_EI = T("extra_items").set_index("item")
show("PHQ-15 fainting / OCI-R saving, collecting: models vs real (%)",
     f"{_EI.loc['SOM_07'].models:.0f} vs {_EI.loc['SOM_07'].human:.0f}; {_EI.loc['OCI_01'].models:.0f} vs {_EI.loc['OCI_01'].human:.0f}, {_EI.loc['OCI_07'].models:.0f} vs {_EI.loc['OCI_07'].human:.0f}")
show("PHQ-15 / OCI-R: other items, models above real (count of items)",
     f"{(_EI[_EI.instrument == 'phq15'].gap > 0).sum()} of {(_EI.instrument == 'phq15').sum()} / {(_EI[_EI.instrument == 'ocir'].gap > 0).sum()} of {(_EI.instrument == 'ocir').sum()}")
# interviews: answers removed by the platform filter (Llama-3.3 only), per condition
for _f in sorted(glob.glob(str(RUNS / "interview__*__*.jsonl"))):
    _d = pd.DataFrame([json.loads(l) for l in open(_f)]).query("status == 'ok'").drop_duplicates("trial_id")
    _fi = [x if isinstance(x, list) else [] for x in _d.get("filtered_items", pd.Series([[]] * len(_d)))]
    _si, _n18, _i18 = sum("DPQ090" in x for x in _fi), sum(sum(i != "DPQ090" for i in x) for x in _fi), sum(any(i != "DPQ090" for i in x) for x in _fi)
    if _si or _n18: show(f"filter, {_f.split('/')[-1][:-6]}: SI removed / answers 1-8 removed in interviews / remaining", f"{_si} of {len(_d)} ({100 * _si / len(_d):.1f}%) / {_n18} in {_i18} / {len(_d) - _i18}")
print("=== Review round 10 (2026-10-11): repeated personas, diversity support, entropy bounds ===")
_G = T("extra_generality"); _G = _G[_G.ref != "coach_vs_nhanes"].dropna(subset=["endorsed_m"]).assign(ratio=lambda q: q.div_h / q.div_m)
show("nine references, real/model diversity: all estimates / estimates on >= 5 totals", f"{rng_(_G.ratio, 1)} / {rng_(_G[_G.div_bins >= 5].ratio, 1)}")
show("postpartum PCL-5: totals behind the diversity estimate (range over models); PHQ-15 models with >= 5", f"{int(_G[_G.ref == 'pcl5c'].div_bins.min())}-{int(_G[_G.ref == 'pcl5c'].div_bins.max())}; {(_G[_G.ref == 'phq15'].div_bins >= 5).sum()} of {(_G.ref == 'phq15').sum()}")
_PP2 = T("published_personas"); _pub2 = _PP2[_PP2.persona.isin(["talkdep", "cactus", "cactus_cm"])]
show("published personas (TalkDep: pairs of different personas), both questionnaires, 17 models: real/model diversity", rng_(_pub2.div_ratio, 1))
for _src in ("talkdep", "cactus", "cactus_cm"):
    _q = _pub2[_pub2.persona == _src]; show(f"  {_src}: ratio PHQ-9 / GAD-7; AUC PHQ-9 / GAD-7", f"{rng_(_q[_q.instrument == 'phq9'].div_ratio, 1)} / {rng_(_q[_q.instrument == 'gad7'].div_ratio, 1)}; {rng_(_q[_q.instrument == 'phq9'].auc)} / {rng_(_q[_q.instrument == 'gad7'].auc)}")
_FX2 = T("fixes"); _FX2 = _FX2[_FX2.cond == "talkdep"].assign(divp=lambda q: 100 * q.diversity / q.diversity_h)
show("intervention protocol, TalkDep (7 models): diversity % median (range) / AUC median (range)", f"{_FX2.divp.median():.0f} ({rng_(_FX2.divp, 0)}) / {_FX2.auc.median():.2f} ({rng_(_FX2.auc)})")
_W = T("wholeform_items"); _sv = ["mild", "moderate", "moderately_severe", "severe"]
_bc = 2 ** (9 * _W[[f"H_{v}" for v in _sv]].to_numpy()); _bt = 2 ** (9 * _W[[f"Ht_{v}" for v in _sv]].to_numpy())
show("perplexity bound per severity cell: models / real at the same totals", f"{_bc.min():.0f}-{_bc.max():.0f} / {_bt.min():.0f}-{_bt.max():.0f}")
show("perplexity bound at the average item entropy: models / real", f"{2 ** (9 * _W.H.min()):.0f}-{2 ** (9 * _W.H.max()):.0f} / {2 ** (9 * _W.Ht.min()):.0f}-{2 ** (9 * _W.Ht.max()):.0f}")
