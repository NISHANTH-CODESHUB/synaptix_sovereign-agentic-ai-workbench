"""Router eval harness.

Measures the cascade on a HELD-OUT set (none of these appear in
data/exemplars.jsonl). Reports per-lane accuracy so the stage-2 threshold is a
measured decision, not a guess.

    python scripts/eval_router.py            # evaluate current settings
    python scripts/eval_router.py --sweep    # sweep the stage-2 margin threshold
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import router as R  # noqa: E402
from backend.registry_loader import REGISTRY  # noqa: E402
from backend.runtime import RUNTIME  # noqa: E402

# Without this the registry reports nothing installed, stage 2 silently never
# fires, and the harness measures the kNN lane while calling it the cascade.
REGISTRY.mark_installed(RUNTIME.installed())

HELD_OUT = [
    # (query, gold task class)
    ("Summarise the Unit-2 shutdown completion report and list the pending items", "doc_summarise"),
    ("Give me a brief on what the last turnaround achieved", "doc_summarise"),
    ("Condense this 30 page audit into one page for the GM", "doc_summarise"),
    ("Boil down the vendor's technical bid into the key deviations", "doc_summarise"),

    ("Write a Python script that computes remaining life for each CML", "code_write"),
    ("Build a small utility to merge the shift log CSVs", "code_write"),
    ("Generate code that reads the strapping table and returns tank volume", "code_write"),
    ("Automate the renaming of inspection photographs by tag number", "code_write"),

    ("Fix the KeyError this parser throws on the second file", "code_debug"),
    ("My groupby is returning NaN for the night shift, what's wrong", "code_debug"),
    ("This script fails with index out of range, correct it", "code_debug"),

    ("Calculate remaining life for CML-07 with measured 6.1 mm, t-min 6.4 mm, rate 0.21 mm/year", "calculation"),
    ("What line size do I need for 200 m3/hr of diesel at 3 m/s", "calculation"),
    ("Work out the heat duty from these inlet and outlet temperatures", "calculation"),
    ("How much will we save annually if flaring drops by 3 percent", "calculation"),

    ("Draft a note for approval for the spares indent worth 42 lakh rupees", "doc_draft"),
    ("Prepare an approval note for the deviation on line 12in-P-0421", "doc_draft"),
    ("Write a letter to the vendor about the delayed tube bundle", "doc_draft"),
    ("Put together a memo on the revised permit procedure", "doc_draft"),

    ("As per our SOP what approval is required when thickness is below t-min", "kb_qa"),
    ("What does our delegation of authority allow a DGM to sanction", "kb_qa"),
    ("Per our manual how long is a deviation approval valid", "kb_qa"),
    ("Which clause in our standard covers reduced inspection intervals", "kb_qa"),

    ("Pull out the thickness readings and CML numbers into a table", "structured_extract"),
    ("Extract every observation with its severity from this audit", "structured_extract"),
    ("Tabulate the equipment tags and their last inspection dates", "structured_extract"),
    ("List the line numbers with design pressures in a structured form", "structured_extract"),

    ("Explain the difference between a PSV and a rupture disc", "general_qa"),
    ("What is corrosion under insulation and why is it a problem", "general_qa"),
    ("Tell me how risk based inspection works in principle", "general_qa"),
    ("What does NPSH available mean for a pump", "general_qa"),
]


def evaluate(threshold=None, quiet=False):
    if threshold is not None:
        R.MARGIN_THRESHOLD = threshold
    knn_ok = casc_ok = stage2 = 0
    lat = []
    rows = []
    for text, gold in HELD_OUT:
        v = R.ROUTER.embedder([text])[0]
        sims = sorted(((c, float(v @ m)) for c, m in R.ROUTER.centroids.items()),
                      key=lambda kv: -kv[1])
        knn_pred = sims[0][0]
        t0 = time.perf_counter()
        d = R.ROUTER.route(text)
        lat.append((time.perf_counter() - t0) * 1000)
        knn_ok += knn_pred == gold
        casc_ok += d.task_class == gold
        stage2 += d.stage == 2
        rows.append((text[:52], gold, knn_pred, d.task_class, d.stage,
                     round(sims[0][1] - sims[1][1], 4)))
    n = len(HELD_OUT)
    if not quiet:
        print(f"\n{'query':54} {'gold':19} {'kNN':19} {'cascade':19} st  margin")
        print("-" * 128)
        for r in rows:
            mark = "  " if r[3] == r[1] else "!!"
            print(f"{mark}{r[0]:52} {r[1]:19} {r[2]:19} {r[3]:19} {r[4]}  {r[5]}")
    lat.sort()
    return {
        "n": n,
        "knn_acc": knn_ok / n,
        "cascade_acc": casc_ok / n,
        "stage2_rate": stage2 / n,
        "p50_ms": round(lat[len(lat) // 2], 1),
        "p95_ms": round(lat[int(len(lat) * 0.95) - 1], 1),
        "threshold": R.MARGIN_THRESHOLD,
    }


def main():
    if "--sweep" in sys.argv:
        print("\n  judge       margin   kNN-only   cascade   stage2-rate   p50      p95")
        print("  " + "-" * 76)
        best = None
        for judge in ("none", "qwen3-06b", "qwen3-4b"):
            R.JUDGE_MODEL_ID = judge
            for th in (0.0, 0.035, 0.055, 0.09):
                if judge == "none" and th > 0.0:
                    continue
                m = evaluate(th, quiet=True)
                print(f"  {judge:<11} {th:<8.3f} {m['knn_acc']:<10.1%} "
                      f"{m['cascade_acc']:<9.1%} {m['stage2_rate']:<13.1%} "
                      f"{m['p50_ms']:<8} {m['p95_ms']}")
                if best is None or m["cascade_acc"] > best[1]["cascade_acc"]:
                    best = ((judge, th), m)
        print(f"\n  best: judge={best[0][0]} margin={best[0][1]} → {best[1]['cascade_acc']:.1%} "
              f"accuracy at {best[1]['stage2_rate']:.0%} stage-2 rate\n")
        return

    m = evaluate()
    print(f"\n  held-out queries    : {m['n']}")
    print(f"  kNN-only accuracy   : {m['knn_acc']:.1%}")
    print(f"  full cascade        : {m['cascade_acc']:.1%}")
    print(f"  stage-2 invoked on  : {m['stage2_rate']:.1%} of requests")
    print(f"  routing latency     : p50 {m['p50_ms']} ms · p95 {m['p95_ms']} ms")
    print(f"  margin threshold    : {m['threshold']}\n")


if __name__ == "__main__":
    main()
