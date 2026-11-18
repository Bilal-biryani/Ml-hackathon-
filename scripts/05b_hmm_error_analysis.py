"""Error analysis of the final pipeline (classifier + HMM) on the student-testing trips.

Every wrongly matched fix (reliable label) is put in exactly one category:
  missing_candidate : the reference arc is not among the fix's candidates (no decoder can recover it)
  reverse_twin      : predicted arc is the opposite direction of the reference arc
  adjacent_arc      : predicted arc shares a node with the reference arc (junction / arc boundary)
  other             : anything else (e.g. a parallel street)
Output: results/hmm_error_analysis.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src import config
from src.inference import MapMatcher
from src.io import list_trips
from src.metrics import RELIABLE_M
from src.pipeline import reference_labels


def main():
    mm = MapMatcher()
    counts = {k: {"missing_candidate": 0, "reverse_twin": 0, "adjacent_arc": 0, "other": 0}
              for k in ("classifier_only", "hmm")}
    n_eval = 0
    for d in list_trips(config.STUDENT_TEST_DIR):
        res = mm.predict(d)
        prep, cands = res["prep"], res["cands"]
        lab, _ = reference_labels(d, prep)
        ev = lab.align_dist.to_numpy() <= RELIABLE_M
        y = lab.label_arc.to_numpy()
        cand_sets = cands.groupby("obs")["arc"].apply(set).to_dict()
        arcs, rev = prep.net.arcs, prep.net.reverse_arc
        n_eval += int(ev.sum())
        for name, pred in (("classifier_only", res["pred_clf"]), ("hmm", res["pred_hmm"])):
            for i in np.nonzero(ev & (pred != y))[0]:
                t, p = y[i], pred[i]
                if t not in cand_sets.get(i, set()):
                    k = "missing_candidate"
                elif p >= 0 and rev[t] == p:
                    k = "reverse_twin"
                elif p >= 0 and len(set(arcs[t]) & set(arcs[p])) > 0:
                    k = "adjacent_arc"
                else:
                    k = "other"
                counts[name][k] += 1
    out = {"n_eval_fixes": n_eval, "errors": counts,
           "error_share": {m: {k: v / max(1, sum(c.values())) for k, v in c.items()} for m, c in counts.items()}}
    (config.RESULTS_DIR / "hmm_error_analysis.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
