# Write one canonical_eval_<name>-cv<f> summary per model from the joint per-fold evaluation (numbers unchanged: same
# seed and clips, so the bootstrap resamples are shared), in the layout cv_aggregate.py reads. Saved through the results store.
import json
from prototype_transfer.evaluation import results
store = results.Store("hsn")
NAMES = ["soundscape-only", "focal-only", "r0", "r1a", "r1b", "r2", "r3", "r4", "tf-quantile", "tf-reweight"]
for f in range(3):
    doc = json.loads(store.summary(f"canonical_eval_all-cv{f}").read_text())
    for n in NAMES:
        keep = ["pretrained-xcl", f"{n}-cv{f}"]
        sub = {**doc, "models": {m: doc["models"][m] for m in keep}}
        results.save_summary(store, f"canonical_eval_{n}-cv{f}", sub, headline={"n_models": len(keep)}, entry={"protocol": doc["protocol"]})
print("wrote", 3 * len(NAMES), "per-method summaries")
