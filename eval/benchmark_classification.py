"""Human Diseases and Tabula Sapiens zero-shot classification (macro AUROC).

Four tasks, 1,000 held-out profiles each (the released subset, seed 0):

* ``geo_disease_subtype`` / ``geo_tissue`` — Human Diseases (CellWhisperer bulk RNA-seq benchmark)
* ``ts_geo_cell_type_top20`` / ``ts_geo_tissue`` — Tabula Sapiens single cells (GSE201048)

Each prompt is ``<omics>`` + question + candidate labels; every label is scored by length-normalised
log-likelihood (constrained scoring), and macro AUROC is computed one-vs-rest.

    python eval/benchmark_classification.py --benchmark human_diseases
    python eval/benchmark_classification.py --benchmark tabula_sapiens --workers 4
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from thalamus_reasoning import batch
from thalamus_reasoning.hub import model_path, release_path

HERE = Path(__file__).resolve().parent

CELL_TYPE_QUESTION = "Identify the cell type of this sample based on the above gene expression profile."
TISSUE_QUESTION = "Identify the tissue of origin for this sample based on the above gene expression profile."

BENCHMARKS: dict[str, dict[str, dict]] = {
    "human_diseases": {
        "geo_disease_subtype": {
            "question": "What is the disease subtype for this sample based on the above gene expression profile?",
            "opt_prefix": "Options: ",
        },
        "geo_tissue": {
            "question": "What tissue of origin does the above gene expression profile suggest for this sample?",
            "opt_prefix": "Candidate tissues: ",
        },
    },
    "tabula_sapiens": {
        "ts_geo_cell_type_top20": {"question": CELL_TYPE_QUESTION, "opt_prefix": "Possible answers: "},
        "ts_geo_tissue": {"question": TISSUE_QUESTION, "opt_prefix": "Possible answers: "},
    },
}


def make_prompt(question: str, opt_prefix: str, options) -> str:
    """The user turn: the profile, the question, then the candidate labels."""
    opts = ", ".join(f"'{o}'" for o in sorted(options))
    return f"<omics>\n{question}\n{opt_prefix}{opts}"


def macro_auroc(probs: np.ndarray, gold_idx: np.ndarray) -> float:
    """One-vs-rest AUROC averaged over classes with both positives and negatives."""
    aucs = [roc_auc_score(gold_idx == c, probs[:, c]) for c in range(probs.shape[1])
            if 0 < (gold_idx == c).sum() < len(gold_idx)]
    return float(np.mean(aucs)) if aucs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, choices=sorted(BENCHMARKS))
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Local checkpoint (default: $THALAMUS_REASONING_HOME or the Hub)")
    ap.add_argument("--data", type=Path, default=None, help="Folder with the task files (default: the model repo)")
    ap.add_argument("--log_dir", type=Path, default=None)
    ap.add_argument("--tasks", default="", help="Comma-separated subset of task names.")
    ap.add_argument("--workers", type=int, default=1,
                    help="GPU worker processes, each holding a full model replica (default: 1).")
    ap.add_argument("--max_samples", type=int, default=0,
                    help="Evaluate a random N-sample subset of each task (default: all released rows).")
    ap.add_argument("--seed", type=int, default=0, help="Seed for --max_samples.")
    args = ap.parse_args()

    tasks = BENCHMARKS[args.benchmark]
    selected = [t.strip() for t in args.tasks.split(",") if t.strip()] or list(tasks)
    data = args.data or release_path(f"eval/{args.benchmark}")
    checkpoint = model_path(args.checkpoint)
    log_dir = args.log_dir or HERE / "logs" / args.benchmark
    log_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    for name in selected:
        spec = tasks[name]
        options = json.loads((data / f"{name}_options.json").read_text())
        labels = pd.read_parquet(data / f"{name}_labels.parquet")
        emb = np.load(data / f"{name}_embeddings.npy")
        idx = np.arange(len(labels))
        if 0 < args.max_samples < len(idx):  # seeded random subset, original order
            idx = np.sort(np.random.default_rng(args.seed).choice(idx, args.max_samples, replace=False))
        gold = labels.label.to_numpy()[idx]
        prompt = make_prompt(spec["question"], spec["opt_prefix"], options)
        print(f"\n[{name}] {len(idx)} samples; {len(set(gold))} gold labels; {len(options)} options", flush=True)
        probs = batch.constrained_inference(checkpoint, [prompt] * len(idx), [[emb[i]] for i in idx], options,
                                            workers=args.workers)
        class_to_idx = {c: i for i, c in enumerate(options)}
        auroc = macro_auroc(probs, np.array([class_to_idx[g] for g in gold]))
        summary[name] = {"n": len(idx), "macro_auroc": auroc}
        print(f"[{name}] macro_auroc={auroc:.4f}", flush=True)
        with (log_dir / f"{name}.jsonl").open("w") as f:
            for i, g, p in zip(idx, gold, probs):
                f.write(json.dumps({"obs_name": labels.obs_name.iloc[i], "gold": g,
                                    "pred": options[int(np.argmax(p))]}) + "\n")

    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
