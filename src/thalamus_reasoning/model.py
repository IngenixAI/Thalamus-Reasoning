"""Load Thalamus-Reasoning with Hugging Face ``transformers`` and run it on prompts with omics vectors.

A prompt marks each profile with ``<omics>``; pass one 20,029-float vector per placeholder, in order of
appearance across the whole conversation::

    from thalamus_reasoning import ThalamusReasoning
    lm = ThalamusReasoning.load()             # Hub or $THALAMUS_REASONING_HOME; or .load("/local/copy")
    lm.chat([{"role": "user", "content": "<omics>\\nWhich tissue is this sample from?"}], [vector])
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from thalamus_reasoning.hub import model_path

OMICS = "<omics>"
OMICS_TOKENS = "<|vision_start|><|image_pad|><|vision_end|>"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


def split_thinking(text: str) -> tuple[str, str]:
    """(reasoning, answer) of a generation; the prompt already opened ``<think>``."""
    reasoning, closed, answer = text.rpartition(THINK_CLOSE)
    if not closed:
        return "", text.rsplit(THINK_OPEN, 1)[-1].strip()
    return reasoning.rsplit(THINK_OPEN, 1)[-1].strip(), answer.strip()


def escape_omics(text: str) -> str:
    """Keep a literal ``<omics>`` typed by a user from becoming a profile slot."""
    return text.replace(OMICS, "&lt;omics&gt;")


@dataclass
class Reply:
    answer: str
    reasoning: str


class ThalamusReasoning:
    def __init__(self, model, tokenizer, device) -> None:
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.dtype = next(model.parameters()).dtype
        self.vector_dim = model.config.omics_input_dim
        image_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if image_id != model.config.image_token_index:
            raise ValueError("tokenizer and checkpoint disagree on the omics token")

    @classmethod
    def load(cls, source: str | None = None, device: str | None = None, dtype=torch.bfloat16) -> "ThalamusReasoning":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = model_path(source)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, dtype=dtype).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        return cls(model, tokenizer, device)

    # ------------------------------------------------------------------ prompts
    def prompt_text(self, messages: list[dict], thinking: bool = False) -> str:
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=thinking)
        return text.replace(OMICS, OMICS_TOKENS)

    def _vectors(self, vectors, expected: int) -> torch.Tensor | None:
        if vectors is None or len(vectors) == 0:
            if expected:
                raise ValueError(f"prompt has {expected} {OMICS} placeholders but no vectors were given")
            return None
        arr = np.asarray(np.stack([np.asarray(v, np.float32).reshape(-1) for v in vectors]))
        if arr.shape[1] != self.vector_dim:
            raise ValueError(f"vectors must have {self.vector_dim} values, got {arr.shape[1]} "
                             "(build them with thalamus_reasoning.preprocess)")
        if arr.shape[0] != expected:
            raise ValueError(f"{expected} {OMICS} placeholders but {arr.shape[0]} vectors")
        return torch.from_numpy(arr).to(self.device, self.dtype)

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def chat(self, messages: list[dict], vectors=None, thinking: bool = False, max_new_tokens: int = 1024,
             temperature: float = 0.0, repetition_penalty: float = 1.0) -> Reply:
        """Answer the last user turn. ``vectors``: one per ``<omics>`` in all messages, in order."""
        n_slots = sum(m["content"].count(OMICS) for m in messages)
        omics = self._vectors(vectors, n_slots)
        text = self.prompt_text(messages, thinking)
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        gen = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0,
               "eos_token_id": self.tokenizer.eos_token_id, "pad_token_id": self.tokenizer.pad_token_id}
        if temperature > 0:
            gen["temperature"] = temperature
        if repetition_penalty != 1.0:
            gen["repetition_penalty"] = repetition_penalty
        out = self.model.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                                  omics_vectors=omics, **gen)[0]
        # Text-only prompts come back with the prompt first; with vectors, generation runs from
        # inputs_embeds and returns only the new tokens.
        if out.shape[0] >= enc.input_ids.shape[1] and torch.equal(out[: enc.input_ids.shape[1]], enc.input_ids[0]):
            out = out[enc.input_ids.shape[1]:]
        decoded = self.tokenizer.decode(out, skip_special_tokens=True).strip()
        reasoning, answer = split_thinking(decoded) if thinking else ("", decoded)
        return Reply(answer=answer, reasoning=reasoning)

    def ask(self, question: str, vectors=None, system: str | None = None, **kwargs) -> str:
        """One-turn convenience wrapper; ``question`` contains the ``<omics>`` placeholders.

        ``system`` is an optional system message, e.g. ``thalamus_reasoning.depmap.DEPMAP_SYSTEM`` for DepMap profiles.
        """
        messages = [{"role": "system", "content": system}] if system else []
        return self.chat(messages + [{"role": "user", "content": question}], vectors, **kwargs).answer

    # ------------------------------------------------------------------ constrained scoring
    def classify(self, question: str, vectors, options: list[str], prefix: str = "Possible answers: "
                 ) -> dict[str, float]:
        """Probability of each option, using the prompt format of the paper's classification benchmarks.

        ``question`` should not contain the options; with no ``<omics>`` in it, one profile slot is put first.
        Listing the candidates in the prompt matters: scoring without them is far less accurate.
        Scores are mean log-probability per answer token, as in the benchmarks; keep labels similar in
        length, since a much longer label is not penalised for its extra tokens. The benchmark path
        (``thalamus_reasoning.batch.constrained_inference``) computes the output layer in fp32, so its scores can differ
        slightly.
        """
        if OMICS not in question:
            question = f"{OMICS}\n{question}"
        listed = ", ".join(f"'{o}'" for o in options)
        probs = self._score_options(f"{question}\n{prefix}{listed}", vectors, options)
        return dict(sorted(zip(options, probs.tolist()), key=lambda kv: -kv[1]))

    @torch.no_grad()
    def _score_options(self, question: str, vectors, options: list[str]) -> np.ndarray:
        """Probability of each option as the (non-thinking) answer, from length-normalised log-likelihood."""
        omics = self._vectors(vectors, question.count(OMICS))
        prompt = self.tokenizer(self.prompt_text([{"role": "user", "content": question}]),
                                add_special_tokens=False).input_ids
        scores = []
        for option in options:
            cand = self.tokenizer.encode(option, add_special_tokens=False)
            ids = torch.tensor([prompt + cand], device=self.device)
            logits = self.model(input_ids=ids, omics_vectors=omics, use_cache=False,
                                logits_to_keep=len(cand) + 1).logits[0, : len(cand)].float()
            lp = F.log_softmax(logits, -1).gather(-1, torch.tensor(cand, device=self.device)[:, None])
            scores.append(lp.mean().item())
        scores = np.array(scores)
        return np.exp(scores - scores.max()) / np.exp(scores - scores.max()).sum()
