#!/usr/bin/env bash
# Re-run every analysis on the final data, in dependency order. No API calls.
# 78 runs before 60 and 56: it writes calibration_diversity.csv, which both read (and Figure 9e).
set -u
cd "$(dirname "$0")/.."
mkdir -p outputs/analysis_logs
for s in 12_deep_analysis 13_appendix_tables 30_downstream 31_pcl5_triangle 34_logprob_analysis 36_all_models 38_fixes_analysis 49_published_personas_analysis 50_grm_heldout \
         39_interview_analysis 73_human_rating_analysis 74_agent_evaluation 40_by_severity 41_position 42_main_ci 37_triangle_ci \
         43_scale_ci 44_human_baselines 45_feign_checks 63_reasoning_analysis 64_default_tokens 67_wholeform_items 75_residual_structure 76_severity_overshoot 52_extra_analysis 69_feign_scales 71_separability_checks 53_extra_items 54_filter_audit 78_calibration_diversity 60_paper_tables 55_coverage_audit 56_verify_paper_numbers; do
  printf "%-24s " "$s"
  if uv run --quiet python "scripts/$s.py" > "outputs/analysis_logs/$s.log" 2>&1; then
    echo "ok"
  else
    echo "FAILED (see outputs/analysis_logs/$s.log)"
  fi
done
