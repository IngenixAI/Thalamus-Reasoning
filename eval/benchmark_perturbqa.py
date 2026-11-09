"""Reproduce Thalamus-Reasoning-9B PerturbQA direction results, or rescore saved answers.

    uv run python eval/benchmark_perturbqa.py run --output runs/pqa --workers 4
    uv run python eval/benchmark_perturbqa.py score --predictions runs/pqa/predictions.parquet --output runs/pqa_rescored

The official PerturbQA labels and evaluator are downloaded into eval/perturbqa_official/ on first use.
"""

import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from perturbqa_scoring import (
    CELLS,
    RUN_IDS,
    asset_file,
    load_inputs,
    sampling_seed,
    save_three_run_metrics,
)

ENGINE = {
    "max_model_len": 8192,
    "max_num_seqs": 128,
    "seed": 0,
    "enable_prefix_caching": False,
    "enforce_eager": True,
    "limit_mm_per_prompt": {"image": 1},
    "mm_processor_cache_gb": 0,
}
SAMPLING = {
    "temperature": 0.6,
    "top_p": 0.9,
    "max_tokens": 4096,
    "skip_special_tokens": False,
    "stop_token_ids": [248046],
}
BATCH_SIZE = 256


def prompt_tokens(tokenizer, prompt: str) -> list[int]:
    ids = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": prompt.replace(
                    "<image>",
                    "<|vision_start|><|image_pad|><|vision_end|>",
                ),
            }
        ],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if len(ids) + SAMPLING["max_tokens"] > ENGINE["max_model_len"]:
        raise ValueError("Prompt exceeds context window; truncation is forbidden")
    return ids


def generate_worker(
    gpu: str, checkpoint: Path, data: Path, frame: pd.DataFrame
) -> list[dict]:
    """One isolated GPU replica running the fixed paper protocol."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    import torch
    from thalamus_reasoning.vllm_plugin import register_thalamus_reasoning
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    register_thalamus_reasoning()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=False)
    prompts = [prompt_tokens(tokenizer, prompt) for prompt in frame.prompt]
    profiles = {
        cell: torch.from_numpy(
            np.load(data / f"profiles/{cell}.npy", allow_pickle=False)
        )
        for cell in frame.cell.unique()
    }
    llm = LLM(
        model=str(checkpoint),
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        enable_mm_embeds=True,
        **ENGINE,
    )
    records = []
    for run_id in RUN_IDS:
        for start in range(0, len(frame), BATCH_SIZE):
            rows = list(frame.iloc[start : start + BATCH_SIZE].itertuples())
            requests = [
                {
                    "prompt_token_ids": ids,
                    "multi_modal_data": {"image": [profiles[row.cell]]},
                }
                for row, ids in zip(rows, prompts[start : start + BATCH_SIZE], strict=True)
            ]
            params = [
                SamplingParams(**SAMPLING, seed=sampling_seed(run_id, row.row_index))
                for row in rows
            ]
            for row, result in zip(rows, llm.generate(requests, params), strict=True):
                output = result.outputs[0]
                records.append(
                    {
                        "row_index": row.row_index,
                        "run_id": run_id,
                        "sampling_seed": sampling_seed(run_id, row.row_index),
                        "response": output.text,
                        "finish_reason": output.finish_reason,
                        "generated_tokens": len(output.token_ids),
                    }
                )
    # The spawned process exits after this shard, releasing the engine and CUDA state.
    return records


def generate(
    checkpoint: Path, data: Path, frame: pd.DataFrame, workers: int
) -> list[dict]:
    import torch

    available = torch.cuda.device_count()
    if not 1 <= workers <= min(available, len(frame)):
        raise ValueError(
            f"Requested {workers} workers for {available} GPUs and {len(frame)} rows"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = (
        visible.split(",") if visible is not None else list(map(str, range(available)))
    )
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        max_tasks_per_child=1,
    ) as pool:
        futures = [
            pool.submit(
                generate_worker,
                devices[rank].strip(),
                checkpoint,
                data,
                frame.iloc[rank::workers],
            )
            for rank in range(workers)
        ]
        records = [record for future in futures for record in future.result()]
    records.sort(key=lambda row: (row["run_id"], row["row_index"]))
    expected = [(run_id, i) for run_id in RUN_IDS for i in frame.row_index]
    if [(row["run_id"], row["row_index"]) for row in records] != expected:
        raise ValueError("Worker outputs do not match input row indices")
    return records


def check_profiles(checkpoint: Path, data: Path) -> None:
    """Fail before inference if a control profile does not fit the checkpoint."""
    dim = json.loads((checkpoint / "config.json").read_text())["omics_input_dim"]
    for cell in CELLS:
        profile = np.load(asset_file(data, f"profiles/{cell}.npy"), allow_pickle=False)
        if profile.shape != (1, dim) or profile.dtype != np.float32 or not np.isfinite(profile).all():
            raise ValueError(f"Invalid control profile: {cell}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["run", "score"])
    parser.add_argument("--data", type=Path, default=None,
                        help="Folder with inputs.parquet and profiles/ (default: the model repository)")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Local checkpoint (default: $THALAMUS_REASONING_HOME or the Hub)")
    parser.add_argument("--predictions", type=Path,
                        help="Required for score: predictions.parquet from a previous run.")
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory.")
    parser.add_argument("--workers", type=int, default=1,
                        help="One model per GPU (default: 1; the reported run used four).")
    args = parser.parse_args()
    if (args.action == "score") != (args.predictions is not None):
        parser.error("--predictions is required for score and only valid for score")
    from thalamus_reasoning.hub import model_path, release_path

    data = args.data or release_path("eval/perturbqa")
    frame = load_inputs(data)
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = {"model": "Thalamus-Reasoning-9B", "checkpoint": "Thalamus-Reasoning-9B release", "thinking": "off",
                  "action": args.action}
    if args.action == "run":
        checkpoint = Path(model_path(args.checkpoint))
        check_profiles(checkpoint, data)
        predictions = pd.DataFrame(generate(checkpoint, data, frame, args.workers)).merge(
            frame, on="row_index", how="left", validate="many_to_one"
        )
        predictions.to_parquet(args.output / "predictions.parquet", index=False)
        provenance.update(
            workers=args.workers,
            engine=ENGINE,
            sampling=SAMPLING,
            request_seed="SHA256(perturbqa:omicslm:run_id:row_index), first 8 bytes modulo 2**63",
            batch_size=BATCH_SIZE,
        )
    else:
        predictions = pd.read_parquet(args.predictions)
    summary = save_three_run_metrics(frame, predictions, data, args.output, provenance)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
