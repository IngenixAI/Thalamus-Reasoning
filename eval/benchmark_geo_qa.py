"""GEO QA benchmark for Thalamus-Reasoning.

For each entry in the GEO QA JSONs, every GEO sample id (``GSM\\d+``)
in the question is replaced with an ``<omics>`` placeholder; the
matching row from ``embeddings.npz`` (both from ``eval/geo_omics_qa`` in
the model repository, or ``--data``) is supplied per slot. The model
then generates a free-form answer.

Yes/no JSONs (``geo_qa_binary*``) get a ``\\nAnswer with Yes or No.``
suffix and are scored as plain accuracy on the first token of the
prediction. Free-text JSONs are scored with DeepEval ``GEval`` using a
Gemini judge.

Generation runs without thinking: what is scored is the answer itself.
Every model prompt appends the sentence: "Base your answer on the supplied
expression profiles." The judge receives the original prompt without it.

    uv run python eval/benchmark_geo_qa.py --workers 4
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

from thalamus_reasoning import batch
from thalamus_reasoning.hub import model_path, release_path

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "600")
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "120")
os.environ.setdefault("DEEPEVAL_RETRY_MAX_ATTEMPTS", "10")
os.environ.setdefault("DEEPEVAL_RETRY_CAP_SECONDS", "60")


HERE = Path(__file__).resolve().parent

GSM_RE = re.compile(r"(?<!\w)GSM\d+(?!\w)")
JUDGE_MODEL = "gemini-3.1-flash-lite"
PROFILE_INSTRUCTION = "Base your answer on the supplied expression profiles."

EVALUATION_STEPS = [
    "Read the input question to understand what biological information is being asked for.",
    "Identify every key biological entity in the expected output: gene names, cell types or "
    "cell lines, tissues, organisms, diseases, treatments, compounds, and signaling pathways. "
    "These are the reference facts.",
    "Check whether the actual output mentions the same key entities or their accepted synonyms "
    "and aliases. An entity counts as matched if it refers to the same biological concept. Note "
    "every reference entity that is missing or replaced by an incorrect entity. Additional "
    "correct details in the actual output that go beyond the expected output should not be "
    "penalized.",
    "For every claim about the direction of a biological change in the expected output — "
    "upregulation vs. downregulation, activation vs. inhibition, increased vs. decreased — "
    "verify that the actual output states the same direction. A contradicted direction is a "
    "critical factual error. If the expected output contains no directional claims, skip this "
    "step.",
    "Evaluate whether the biological interpretation and mechanistic reasoning in the actual "
    "output are consistent with the expected output. The explanation may use different wording, "
    "level of detail, or sentence structure, but the core causal claims and conclusions must not "
    "contradict the reference. If the expected output is a short factual description with no "
    "reasoning, skip this step.",
    "Assess completeness: does the actual output cover the main findings of the expected "
    "output? Missing a minor detail is acceptable; missing the central finding or conclusion is "
    "a significant deficiency. The actual output may include additional correct information not "
    "present in the expected output — this should not lower the score.",
]

RUBRIC = [
    (0, 2, "The actual output is irrelevant to the question, fails to address what was asked, "
           "or is entirely about the wrong biological subject."),
    (3, 4, "The actual output addresses the correct topic but contains major factual errors: "
           "multiple key entities are wrong, the direction of a gene expression change is "
           "contradicted, or the central biological conclusion is incorrect."),
    (5, 6, "The actual output captures the correct biological subject and some key entities, "
           "but is noticeably incomplete or contains minor factual inaccuracies that affect "
           "the overall correctness."),
    (7, 8, "The actual output correctly conveys the main finding and most key entities. Minor "
           "omissions in detail or slight differences in phrasing are present, but no factual "
           "errors."),
    (9, 10, "The actual output accurately captures all key biological entities, expression "
            "directions, and the central conclusion from the expected output. Wording and "
            "structure may differ, but the factual content is fully aligned."),
]


def build(question: str, gsm_to_row: dict[str, int], emb: np.ndarray) -> tuple[str, list[np.ndarray]]:
    """Replace each GSM in *question* with ``<omics>``; collect matching embeddings."""
    parts, embs, prev = [], [], 0
    for m in GSM_RE.finditer(question):
        parts.append(question[prev:m.start()])
        parts.append("<omics>")
        embs.append(emb[gsm_to_row[m.group()]])
        prev = m.end()
    parts.append(question[prev:])
    return "".join(parts), embs


def first_yn(text: str) -> bool | None:
    t = text.strip().lower()
    return True if t.startswith("yes") else False if t.startswith("no") else None


def deepeval_score(prompts: list[str], preds: list[str], answers: list[str]) -> float:
    from deepeval import evaluate as deepeval_evaluate
    from deepeval.evaluate import AsyncConfig, ErrorConfig
    from deepeval.metrics import GEval
    from deepeval.metrics.g_eval.utils import Rubric
    from deepeval.models import GeminiModel
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    geval = GEval(
        name="DeepEval",
        evaluation_steps=EVALUATION_STEPS,
        rubric=[Rubric(score_range=(lo, hi), expected_outcome=desc) for lo, hi, desc in RUBRIC],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT,
                           LLMTestCaseParams.EXPECTED_OUTPUT],
        model=GeminiModel(model=JUDGE_MODEL),
    )
    cases = [LLMTestCase(input=q, actual_output=p, expected_output=a)
             for q, p, a in zip(prompts, preds, answers)]
    scores: list[float | None] = [None] * len(cases)
    pending = list(range(len(cases)))
    for _ in range(4):
        if not pending:
            break
        results = deepeval_evaluate(
            test_cases=[cases[i] for i in pending], metrics=[geval],
            error_config=ErrorConfig(ignore_errors=True),
            async_config=AsyncConfig(run_async=True, max_concurrent=80),
        )
        for result in results.test_results:
            if result.metrics_data:
                [data] = result.metrics_data
                if data.error is None and data.score is not None:
                    scores[pending[result.index]] = data.score
        pending = [i for i in pending if scores[i] is None]
    if pending:
        raise RuntimeError(f"DeepEval failed for {len(pending)} samples after four attempts")
    return float(np.mean(scores))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Local checkpoint (default: $THALAMUS_REASONING_HOME or the Hub)")
    ap.add_argument("--data", type=Path, default=None,
                    help="Folder with the question JSONs and embeddings.npz (default: the model repo)")
    ap.add_argument("--log_dir", type=Path, default=HERE / "logs" / "geo_omics_qa")
    ap.add_argument("--tasks", default="",
                    help="Comma-separated question-file stems to run (default: all).")
    ap.add_argument("--workers", type=int, default=1,
                    help="GPU worker processes, each holding a full model replica (default: 1).")
    args = ap.parse_args()
    selected = [t.strip() for t in args.tasks.split(",") if t.strip()]

    if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        print("Note: free-text splits need your own Gemini credentials (GOOGLE_API_KEY) for the judge.")
    checkpoint = model_path(args.checkpoint)
    data = args.data or release_path("eval/geo_omics_qa")
    npz = np.load(data / "embeddings.npz", allow_pickle=False)
    emb = npz["embeddings"]
    gsm_to_row = {str(g): i for i, g in enumerate(npz["geo_accession"])}

    args.log_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    for qf in sorted(data.glob("*.json")):
        if selected and qf.stem not in selected:
            continue
        rows = json.loads(qf.read_text())
        is_binary = qf.name.startswith("geo_qa_binary")

        prompts, judge_prompts, embs_list = [], [], []
        for r in rows:
            p, e = build(r["question"], gsm_to_row, emb)
            judge_prompt = p
            p += "\n\n" + PROFILE_INSTRUCTION
            if is_binary:
                p += "\nAnswer with Yes or No."
                judge_prompt += "\nAnswer with Yes or No."
            prompts.append(p)
            judge_prompts.append(judge_prompt)
            embs_list.append(e)

        preds = batch.causal_inference(
            checkpoint, prompts, embs_list,
            max_new_tokens=4 if is_binary else 1024,
            workers=args.workers,
        )
        preds = [batch.split_thinking(p)[1] for p in preds]
        answers = [r["answer"] for r in rows]

        if is_binary:
            score = sum(first_yn(p) == first_yn(a) for p, a in zip(preds, answers)) / len(rows)
            print(f"[{qf.name}] accuracy={score:.4f}  n={len(rows)}", flush=True)
            summary[qf.name] = {"metric": "accuracy", "score": score, "n": len(rows)}
        else:
            score = deepeval_score(judge_prompts, preds, answers)
            print(f"[{qf.name}] deepeval={score:.4f}  n={len(rows)}", flush=True)
            summary[qf.name] = {"metric": "deepeval", "score": score,
                                "judge_model": JUDGE_MODEL, "n": len(rows)}

        log_path = args.log_dir / f"{qf.stem}.jsonl"
        with log_path.open("w") as lf:
            for r, p, pred, jp in zip(rows, prompts, preds, judge_prompts):
                record = {"question": r["question"], "answer": r["answer"], "prompt": p, "pred": pred,
                          "judge_prompt": jp}
                lf.write(json.dumps(record) + "\n")
        summary[qf.name]["log"] = str(log_path)

    summary_path = args.log_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n{json.dumps(summary, indent=2)}", flush=True)
    print(f"Summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
