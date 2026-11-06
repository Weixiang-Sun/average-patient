import sys, json
sys.path.insert(0, "src")
from mental_act import llm_runner, fixes
from mental_act.config import RUNS
CONDS = ["label", "hetero", "numeric", "biography", "fewshot"]
MODELS = [(sys.argv[1], sys.argv[2])]
for prov, model in MODELS:
    tag = model.replace('.','').replace('-','').lower()
    t = fixes.build_trials(CONDS, n_per_cell=40)
    # keep few-shot exemplars with each record so copy-rate can be computed
    meta = {x["trial_id"]: x["meta"] for x in t}
    out = f"fixes__{tag}"
    llm_runner.run(t, model=model, provider=prov, out_name=out, workers=10)
    rs = [json.loads(l) for l in (RUNS/f"{out}.jsonl").read_text().splitlines()]
    (RUNS/f"{out}.meta.json").write_text(json.dumps(meta))
    print(f"{model}: {sum(r['status']=='ok' for r in rs)}/{len(rs)} ok", flush=True)
