"""Standalone PerturbQA direction scoring; no model or training dependencies."""

import hashlib
import importlib.util
import json
import re
import shutil
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

CELLS = ("k562", "rpe1", "hepg2", "jurkat")
COUNTS = dict(zip(CELLS, (3119, 5365, 3603, 4121), strict=True))
KEYS = ["cell", "pert", "gene"]
RUN_IDS = (0, 1, 2)
# Inputs and control profiles come from the model repository (eval/perturbqa); the official PerturbQA
# labels and evaluator are downloaded on first use into OFFICIAL_DIR, from the release evaluated in the paper.
OFFICIAL_DIR = Path(__file__).resolve().parent / "perturbqa_official"
OFFICIAL_URL = "https://raw.githubusercontent.com/Genentech/PerturbQA/df2fe3a7802f2a46a43876b4f8674734aad3a30a"
OFFICIAL_FILES = {
    "eval.py": "perturbqa/eval.py", "README.md": "README.md", "LICENSE.txt": "LICENSE.txt",
    **{f"{c}-{t}.csv": f"perturbqa/datasets/{c}-{t}.csv" for c in CELLS for t in ("dir", "de")},
}


def download_official() -> None:
    """Fetch the official PerturbQA labels and evaluator (Genentech/PerturbQA, pinned revision)."""
    OFFICIAL_DIR.mkdir(parents=True, exist_ok=True)
    for name, source in OFFICIAL_FILES.items():
        destination = OFFICIAL_DIR / name
        if destination.is_file():
            continue
        partial = destination.with_name(name + ".part")
        print(f"Downloading {name}", flush=True)
        try:
            with urlopen(f"{OFFICIAL_URL}/{source}", timeout=120) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output, length=1 << 20)
            partial.replace(destination)
        finally:
            partial.unlink(missing_ok=True)


def asset_file(root: Path, name: str) -> Path:
    if name.startswith("official/"):
        path = OFFICIAL_DIR / name.removeprefix("official/")
        if not path.is_file():
            download_official()
    else:
        path = root / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing asset: {path}")
    return path


def sampling_seed(run_id: int, row_index: int) -> int:
    """A distinct reproducible random stream for each example and replicate."""
    # The seed string keeps the pre-rename model name so the released PerturbQA runs reproduce exactly.
    identity = f"perturbqa:omicslm:{run_id}:{row_index}".encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") % (2**63)


def training_priors(root: Path) -> pd.Series:
    parts = []
    for cell in CELLS:
        frame = pd.read_csv(asset_file(root, f"official/{cell}-dir.csv"))
        frame = frame.loc[frame.split.eq("train")].copy()
        frame["gene"] = frame.gene.str.upper()
        frame["cell"] = cell
        parts.append(frame)
    return pd.concat(parts).groupby(["cell", "gene"]).label.mean()


def aggregate_runs(
    gold: pd.DataFrame, predictions: pd.DataFrame, priors: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align three complete runs, then average scores per question before AUROC.

    Invalid answers are excluded, falling back to the TRAIN readout prior only when all three fail.
    """
    if "run_id" not in predictions or set(predictions.run_id) != set(RUN_IDS):
        raise ValueError("Exactly three complete runs with run_id 0, 1, 2 are required")
    parts = []
    identity_columns = (["row_index"] if "row_index" in gold else []) + KEYS + ["label"]
    averaged = gold[identity_columns].copy().reset_index(drop=True)
    for run_id in RUN_IDS:
        run = align_predictions(gold, predictions[predictions.run_id.eq(run_id)])
        if "prompt" in run and "prompt" in gold and run.prompt.tolist() != gold.prompt.tolist():
            raise ValueError("Thalamus-Reasoning prompts differ from the benchmark")
        classes = pd.Series([parse_tr_answer(r.response, r.finish_reason) for r in run.itertuples()])
        run["score"] = classes.map({0: 0.0, 1: 1.0})
        run["predicted_class"] = classes
        run["valid_answer"] = classes.ne(-1)
        averaged[f"score_run_{run_id}"] = run.score.to_numpy()
        averaged[f"valid_run_{run_id}"] = run.valid_answer.to_numpy()
        parts.append(run)
    averaged["score"] = averaged[[f"score_run_{r}" for r in RUN_IDS]].mean(axis=1)
    averaged["valid_answers"] = averaged[[f"valid_run_{r}" for r in RUN_IDS]].sum(axis=1)
    averaged["training_prior_fallback"] = averaged.score.isna()
    missing = averaged.training_prior_fallback
    if missing.any():
        index = pd.MultiIndex.from_frame(averaged.loc[missing, ["cell", "gene"]])
        averaged.loc[missing, "score"] = priors.reindex(index).to_numpy()
    if not np.isfinite(averaged.score).all():
        raise ValueError("Missing TRAIN prior or nonfinite averaged score")
    averaged["predicted_class"] = np.where(
        averaged.score > 0.5, 1, np.where(averaged.score < 0.5, 0, -1)
    )
    return pd.concat(parts, ignore_index=True), averaged


def save_three_run_metrics(
    gold: pd.DataFrame, predictions: pd.DataFrame, data: Path, output: Path, provenance: dict
) -> dict:
    runs, averaged = aggregate_runs(gold, predictions, training_priors(data))
    individual = []
    for run_id in RUN_IDS:
        run = runs[runs.run_id.eq(run_id)].copy()
        # Individual binary Thalamus-Reasoning runs retain the historical neutral fallback.
        run["score"] = run.score.fillna(0.5)
        individual.append(direction_metrics(run).assign(run_id=run_id))
    pd.concat(individual).to_csv(output / "run_metrics.csv", index=False)
    averaged.to_parquet(output / "predictions_averaged.parquet", index=False)
    provenance.update(
        run_ids=list(RUN_IDS),
        aggregation="mean of per-question scores before AUROC",
        valid_answers_by_run=runs.groupby("run_id").valid_answer.sum().to_dict(),
        training_prior_fallback=int(averaged.training_prior_fallback.sum()),
        answer_mapping="Up=1, Down=0; invalid excluded, all-invalid TRAIN gene prior",
    )
    evaluator = asset_file(data, "official/eval.py")
    spec = importlib.util.spec_from_file_location("official_perturbqa_eval", evaluator)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    for result in direction_metrics(averaged).itertuples():
        group = averaged[averaged.cell.eq(result.cell)]
        expected = official.auc_per_gene(
            list(zip(group.pert, group.gene, strict=True)),
            group.score.to_numpy(),
            group.label.to_numpy(),
        )
        if abs(result.auroc - expected) >= 1e-12:
            raise ValueError(f"AUROC differs from the official evaluator: {result.cell}")
    provenance["official_evaluator_verified"] = True
    return save_metrics(averaged, output, provenance)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_inputs(root: Path) -> pd.DataFrame:
    frame = pd.read_parquet(asset_file(root, "inputs.parquet"))
    gold = []
    for cell in CELLS:
        source = asset_file(root, f"official/{cell}-dir.csv")
        table = pd.read_csv(source).query("split == 'test'").copy()
        table["cell"] = cell
        table[["pert", "gene"]] = table[["pert", "gene"]].apply(lambda s: s.str.upper())
        gold.append(table[KEYS + ["label"]])
    gold = pd.concat(gold, ignore_index=True)
    if gold.groupby("cell").size().to_dict() != COUNTS:
        raise ValueError("Unexpected official TEST counts")
    aligned = align_predictions(gold, frame)
    if not frame[KEYS + ["label"]].equals(aligned[KEYS + ["label"]]):
        raise ValueError("Input order differs from official TEST")
    if frame.row_index.tolist() != list(range(len(gold))):
        raise ValueError("Invalid row indices")
    if not frame.prompt.map(
        lambda p: isinstance(p, str) and p.count("<image>") == 1
    ).all():
        raise ValueError("Expected one control-profile placeholder per question")
    return frame


def align_predictions(gold: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Reject duplicates, missing/extra pairs and changed labels; restore gold order."""
    for frame in (gold, predictions):
        if frame[KEYS + ["label"]].isna().any().any():
            raise ValueError("Missing identities or labels")
        if frame.duplicated(KEYS).any() or not frame.label.isin([0, 1]).all():
            raise ValueError("Duplicate identities or invalid direction labels")
    merged = (
        gold[KEYS + ["label"]]
        .assign(_gold_order=range(len(gold)))
        .merge(
            predictions,
            on=KEYS,
            how="outer",
            validate="one_to_one",
            suffixes=("_gold", ""),
            indicator=True,
            sort=False,
        )
    )
    if (
        not merged._merge.eq("both").all()
        or not merged.label.eq(merged.label_gold).all()
    ):
        raise ValueError("Predictions do not match complete TEST identities and labels")
    return (
        merged.sort_values("_gold_order")
        .drop(columns=["label_gold", "_merge", "_gold_order"])
        .reset_index(drop=True)
    )


def parse_tr_answer(response: str, finish_reason: str) -> int:
    """Return Up=1, Down=0, or invalid=-1, using the paper's completed-answer parser."""
    if finish_reason != "stop":
        return -1
    text = (
        response.strip()
        .removesuffix("<|im_end|>")
        .removesuffix("<|endoftext|>")
        .strip()
    )
    candidate = text.rpartition("</think>")[2] if "</think>" in text else text
    match = re.fullmatch(
        r"(increase|increased|increasing|up|upregulated|up-regulated|"
        r"decrease|decreased|decreasing|down|downregulated|down-regulated)[.!]?",
        candidate.strip(),
        re.IGNORECASE,
    )
    if match:
        return int(match.group(1).lower().startswith(("increas", "up")))
    return -1


def direction_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Score an aligned full TEST with separate continuous scores and class calls.

    predicted_class is 0/1 for Down/Up, -1 for ND, invalid or an exact score tie.
    These rows remain false negatives of their gold class in direction F1.
    """
    if frame.groupby("cell").size().to_dict() != COUNTS:
        raise ValueError("Scoring requires all four complete TEST cells")
    if not np.isfinite(frame.score.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite prediction scores")
    if not frame.predicted_class.isin([-1, 0, 1]).all():
        raise ValueError("Invalid predicted direction class")
    records = []
    for cell in CELLS:
        group = frame[frame.cell.eq(cell)]
        aucs = [
            roc_auc_score(g.label, g.score)
            for _, g in group.groupby("gene")
            if g.label.nunique() == 2
        ]
        if not aucs:
            raise ValueError(f"No eligible readout genes: {cell}")
        down, up = f1_score(
            group.label,
            group.predicted_class,
            labels=[0, 1],
            average=None,
            zero_division=0,
        )
        records.append(
            {
                "cell": cell,
                "rows": len(group),
                "genes_scored": len(aucs),
                "auroc": float(np.mean(aucs)),
                "f1_up": up,
                "f1_down": down,
                "macro_f1": (up + down) / 2,
                "non_direction": int(group.predicted_class.eq(-1).sum()),
            }
        )
    return pd.DataFrame(records)


def save_metrics(frame: pd.DataFrame, output: Path, provenance: dict) -> dict:
    table = direction_metrics(frame)
    table.to_csv(output / "metrics.csv", index=False)
    summary = {
        "rows": len(frame),
        **{
            key: float(table[key].mean())
            for key in ("auroc", "f1_up", "f1_down", "macro_f1")
        },
        "non_direction": int(table.non_direction.sum()),
    }
    write_json(output / "summary.json", summary)
    write_json(output / "protocol.json", provenance)
    return summary
