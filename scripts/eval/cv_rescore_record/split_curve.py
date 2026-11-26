# One canonical_eval_<name>-cv<f> summary per model from the joint evaluation canonical_eval_curve-cv<f> (same seed and
# clips, so numbers are unchanged), in the layout cv_aggregate.py reads. Saved through the results store.
import json
from prototype_transfer.evaluation import results
store = results.Store("hsn")
NAMES = ["lb100", "lb250", "lb500", "lb1000", "lb2000", "r0x-nopush", "r0x-paperortho", "r0x-nopush-paperortho"]
for f in range(3):
    doc = json.loads(store.summary(f"canonical_eval_curve-cv{f}").read_text())
    for n in NAMES:
        keep = ["pretrained-xcl", f"{n}-cv{f}"]
        results.save_summary(store, f"canonical_eval_{n}-cv{f}", {**doc, "models": {m: doc["models"][m] for m in keep}},
                             headline={"n_models": len(keep)}, entry={"protocol": doc["protocol"]})
print("wrote", 3 * len(NAMES), "per-method summaries")
