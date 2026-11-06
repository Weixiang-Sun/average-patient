"""Administer the PCL-5 in the three roles of the 6b8jf human study."""
import argparse, sys
sys.path.insert(0, "src")
from mental_act import llm_runner, pcl5
p = argparse.ArgumentParser()
p.add_argument("--model", required=True); p.add_argument("--provider", default="azure")
p.add_argument("--out", required=True); p.add_argument("--n", type=int, default=60)
p.add_argument("--workers", type=int, default=12); p.add_argument("--limit", type=int)
a = p.parse_args()
t = pcl5.build_trials(n_per_cell=a.n)
print(f"{len(t)} trials  model={a.model}")
llm_runner.run(t, model=a.model, provider=a.provider, out_name=a.out,
               limit=a.limit, workers=a.workers)
