"""Inference utilities for Thalamus-Reasoning.

Two entry points:

    causal_inference(checkpoint_path, questions, embeddings) -> list[str]
        Free-form generation via vLLM. Multi-GPU via tensor parallelism
        or independent GPU workers.

    constrained_inference(checkpoint_path, questions, embeddings, classes)
            -> np.ndarray  # shape (n_samples, n_classes)
        Class probabilities via teacher-forced log-probs, batched over
        samples.

Both accept ``workers`` for one model replica per visible GPU. The default
(``workers=1``) keeps generation's tensor-parallel and scoring's single-GPU paths.

The assistant turn opens with an empty ``<think></think>`` block to disable
thinking, matching the non-thinking chat template.

Inputs:
    checkpoint_path : path to the Thalamus-Reasoning checkpoint directory.
    questions       : list[str], each may contain ``<omics>`` placeholders.
    embeddings      : list[list[np.ndarray]] -- one inner list per question.
                      Each inner element is a 1-D vector of length
                      ``omics_input_dim``. Number of vectors must equal
                      the number of ``<omics>`` placeholders in the prompt.

``<omics>`` is rewritten to ``<|vision_start|><|image_pad|><|vision_end|>``
before tokenisation, the form the model was trained on. The omics projection
writes the projected vectors into the image-pad slots.
"""

import gc
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import softmax
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# split_thinking is re-exported for callers of this module (eval/benchmark_geo_qa.py).
from thalamus_reasoning.model import OMICS, OMICS_TOKENS, THINK_CLOSE, THINK_OPEN, split_thinking  # noqa: F401

# Prompt budget of a vLLM request; max_model_len adds max_new_tokens to it.
_MAX_PROMPT_LEN = 8192
_SAMPLE_BATCH_SIZE = 8


# ---------------------------------------------------------------------------
# Prompt building (shared by both paths)
# ---------------------------------------------------------------------------

def _build_prompt_text(question: str, system: str | None = None) -> str:
    question = question.replace(OMICS, OMICS_TOKENS)
    opening = f"{THINK_OPEN}\n\n{THINK_CLOSE}\n\n"
    head = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
    return f"{head}<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{opening}"


def _run_shard(
    gpu: str,
    method: Literal["causal", "constrained"],
    checkpoint_path: str,
    questions: list[str],
    embeddings: list[list[np.ndarray]],
    kwargs: dict[str, Any],
) -> list[str] | np.ndarray:
    # Mask before either backend initializes CUDA, including vLLM subprocesses.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    if method == "causal":
        return causal_inference(
            checkpoint_path, questions, embeddings, n_gpus=1, **kwargs,
        )
    return constrained_inference(
        checkpoint_path, questions, embeddings, device="cuda:0", **kwargs,
    )


def _parallel_inference(
    method: Literal["causal", "constrained"],
    checkpoint_path: str,
    questions: list[str],
    embeddings: list[list[np.ndarray]],
    workers: int,
    **kwargs: Any,
) -> list[str] | np.ndarray:
    if len(questions) != len(embeddings):
        raise ValueError("Questions and embeddings must have equal lengths")
    available = torch.cuda.device_count()
    if not 1 <= workers <= available:
        raise ValueError(f"Requested {workers} workers for {available} visible GPUs")
    if not questions:
        return [] if method == "causal" else np.empty((0, len(kwargs["classes"])))

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = (
        [gpu.strip() for gpu in visible.split(",")]
        if visible is not None else [str(i) for i in range(available)]
    )
    workers = min(workers, len(questions))
    boundaries = np.linspace(0, len(questions), workers + 1, dtype=int)
    # A process must not handle a second GPU after its CUDA state is initialized.
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=mp.get_context("spawn"), max_tasks_per_child=1,
    ) as pool:
        futures = []
        for rank in range(workers):
            start, stop = boundaries[rank:rank + 2]
            futures.append(pool.submit(
                _run_shard, devices[rank], method, checkpoint_path,
                questions[start:stop], embeddings[start:stop], kwargs,
            ))
        # Submission order preserves the original sample order.
        parts = [future.result() for future in futures]
    if method == "causal":
        return [text for part in parts for text in part]
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Causal inference (vLLM)
# ---------------------------------------------------------------------------

def causal_inference(
    checkpoint_path: str,
    questions: list[str],
    embeddings: list[list[np.ndarray]],
    max_new_tokens: int = 2048,
    n_gpus: int | None = None,
    workers: int = 1,
    system: str | None = None,
) -> list[str]:
    """Free-form generation via vLLM.

    ``workers > 1`` splits samples across independent single-GPU processes.
    Otherwise, ``n_gpus`` controls vLLM's ``tensor_parallel_size``. Default = all
    visible GPUs (``torch.cuda.device_count()``). ``system`` is an optional system message for every
    prompt, e.g. ``thalamus_reasoning.depmap.DEPMAP_SYSTEM`` when the prompts name DepMap cell lines.
    """
    if workers != 1:
        if n_gpus not in (None, 1):
            raise ValueError("GPU workers cannot also use tensor parallelism")
        return _parallel_inference(
            "causal", checkpoint_path, questions, embeddings, workers,
            max_new_tokens=max_new_tokens, system=system,
        )

    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment, destroy_model_parallel,
    )

    if n_gpus is None:
        n_gpus = max(torch.cuda.device_count(), 1)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)

    llm = LLM(
        model=checkpoint_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=n_gpus,
        limit_mm_per_prompt={"image": 64},
        enable_mm_embeds=True,
        disable_log_stats=True,
        gpu_memory_utilization=0.85,
        max_model_len=_MAX_PROMPT_LEN + max_new_tokens,
    )
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        repetition_penalty=1.0,
        max_tokens=max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id],
        skip_special_tokens=True,
    )

    inputs: list[dict] = []
    for q, embs in zip(questions, embeddings, strict=True):
        item: dict = {"prompt": _build_prompt_text(q, system)}
        if len(embs):
            # vLLM detects embeddings via shape: list of 2D tensors
            # (seq_len, hidden). For one omics vector per <|image_pad|> slot,
            # seq_len=1.
            item["multi_modal_data"] = {
                "image": [
                    torch.as_tensor(np.asarray(e).reshape(1, -1), dtype=torch.float32)
                    for e in embs
                ]
            }
        inputs.append(item)

    results = llm.generate(inputs, sampling)
    texts = [r.outputs[0].text for r in results]

    del llm, results
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    torch.cuda.empty_cache()
    return texts


# ---------------------------------------------------------------------------
# Constrained inference (HF)
# ---------------------------------------------------------------------------

def _stack_omics(embs: list[np.ndarray], device, dtype) -> torch.Tensor | None:
    if not embs:
        return None
    arr = np.concatenate([np.atleast_2d(e) for e in embs], axis=0)
    return torch.from_numpy(arr).to(device=device, dtype=dtype)


def _left_pad(seqs: list[list[int]], pad_id: int, device: str):
    """Left-pad token sequences into a batch with attention and position tensors."""
    maxlen = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((len(seqs), maxlen), dtype=torch.long, device=device)
    for i, seq in enumerate(seqs):
        input_ids[i, maxlen - len(seq):] = torch.tensor(seq, dtype=torch.long, device=device)
        attn[i, maxlen - len(seq):] = 1
    return input_ids, attn, attn.cumsum(-1).sub_(1).clamp_(min=0)


def _float32_output_head(model) -> None:
    """Keep close candidate scores distinct without changing backbone precision."""
    head = model.language_model.lm_head
    head.weight = torch.nn.Parameter(head.weight.detach().float(), requires_grad=False)
    if head.bias is not None:
        head.bias = torch.nn.Parameter(head.bias.detach().float(), requires_grad=False)
    head.register_forward_pre_hook(lambda module, args: (args[0].float(),))


@torch.no_grad()
def constrained_inference(
    checkpoint_path: str,
    questions: list[str],
    embeddings: list[list[np.ndarray]],
    classes: list[str],
    device: str = "cuda",
    workers: int = 1,
) -> np.ndarray:
    """Class probabilities from length-normalised teacher-forced log-probs.

    Every candidate is appended to the prompt and scored in one forward pass
    per candidate, batched across samples. The backbone keeps recurrent state,
    so a cached prompt prefix cannot be reused across candidates.
    """
    if workers != 1:
        if device != "cuda":
            raise ValueError("Select worker GPUs with CUDA_VISIBLE_DEVICES, not device")
        return _parallel_inference(
            "constrained", checkpoint_path, questions, embeddings, workers, classes=classes,
        )

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, trust_remote_code=True, dtype=torch.bfloat16,
    ).to(device).eval()
    dtype = next(model.parameters()).dtype
    _float32_output_head(model)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    cand_token_ids = [tokenizer.encode(c, add_special_tokens=False) for c in classes]
    prompts = [
        tokenizer(_build_prompt_text(question), add_special_tokens=False).input_ids
        for question in questions
    ]
    log_scores = np.empty((len(questions), len(classes)), dtype=np.float64)

    for start in tqdm(range(0, len(prompts), _SAMPLE_BATCH_SIZE), desc="score"):
        stop = min(start + _SAMPLE_BATCH_SIZE, len(prompts))
        batch = prompts[start:stop]
        omics = _stack_omics(
            [e for embs in embeddings[start:stop] for e in embs], device, dtype,
        )
        for ci, candidate in enumerate(cand_token_ids):
            k = len(candidate)
            input_ids, attn, positions = _left_pad(
                [prompt + candidate for prompt in batch], pad_id, device,
            )
            out = model(
                input_ids=input_ids, attention_mask=attn, position_ids=positions,
                omics_vectors=omics, use_cache=False, logits_to_keep=k + 1,
            )
            token_lp = F.log_softmax(out.logits[:, :k].float(), dim=-1).gather(
                -1, torch.tensor(candidate, device=device).view(1, k, 1).expand(stop - start, k, 1),
            ).squeeze(-1)
            log_scores[start:stop, ci] = (token_lp.sum(1) / k).cpu().numpy()

    # Release the GPU before returning: the next task starts a vLLM engine,
    # which claims most of the same device. Moving the weights off first frees
    # their storage even if something still references the module.
    model.to("cpu")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return softmax(log_scores, axis=1)
