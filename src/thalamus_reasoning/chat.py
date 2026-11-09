"""Interactive chat with Thalamus-Reasoning.

Profiles enter a message in two ways:

- **DepMap cell lines** by name (``Compare MCF7 and T-47D``): each first mention gets the line's DepMap 24Q4
  expression vector. Ambiguous names need ``@`` or a following "cells" (``@RD``, ``RD cells``).
- **Your own samples** as ``@sample_id`` after loading vectors made by ``tr-embed``
  (``--profiles my_vectors.npz`` or ``/load my_vectors.npz``).

Run::

    tr-chat                                # model from the Hub (or $THALAMUS_REASONING_HOME)
    tr-chat --profiles my_vectors.npz --thinking

Commands: ``/load FILE.npz``, ``/profiles``, ``/lines [text]``, ``/think on|off``, ``/reset``, ``/exit``.
"""

import argparse
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from thalamus_reasoning.depmap import DEPMAP_SYSTEM, DepMap
from thalamus_reasoning.model import OMICS, ThalamusReasoning, Reply, escape_omics
from thalamus_reasoning.preprocess import load_profiles

log = logging.getLogger(__name__)
_REF = re.compile(r"@([A-Za-z0-9][\w.:\-]*)")


@dataclass
class Turn:
    text: str  # message as sent to the model, with <omics> placeholders
    vectors: list[np.ndarray]
    notes: list[str]


@dataclass
class Conversation:
    """Message history plus the vectors of every ``<omics>`` slot so far."""

    lm: ThalamusReasoning
    depmap: DepMap | None
    profiles: dict[str, np.ndarray] = field(default_factory=dict)
    messages: list[dict] = field(default_factory=list)
    vectors: list[np.ndarray] = field(default_factory=list)
    used_depmap: bool = False

    def load_profiles(self, path: str | Path) -> int:
        profiles = load_profiles(path)
        self.profiles.update(profiles)
        return len(profiles)

    def prepare(self, message: str) -> Turn:
        """Insert placeholders for ``@sample`` references and DepMap names."""
        message = escape_omics(message)
        notes, own, slots = [], [], {}

        def own_profile(m: re.Match) -> str:
            sample = m.group(1).rstrip(".:,;")
            if sample not in self.profiles:
                return m.group(0)
            # A private-use marker that cell-line matching cannot see ("@MCF7_ctrl" must not pull in MCF7).
            key = f"\ue000{chr(0xE100 + len(slots))}\ue001"
            slots[key] = (sample, sample not in own)
            if sample not in own:
                own.append(sample)
            return key + m.group(1)[len(sample):]

        if self.profiles:
            others = [m.group(1).rstrip(".:,;") for m in _REF.finditer(message)
                      if m.group(1).rstrip(".:,;") not in self.profiles]
            if others:
                notes.append("not among your loaded samples: " + ", ".join(f"@{o}" for o in others)
                             + " (checked against DepMap cell lines instead)")
        marked = _REF.sub(own_profile, message)
        depmap_vectors = []
        if self.depmap is not None:
            res, depmap_vectors = self.depmap.resolve(marked)
            marked, notes = res.text, notes + res.notes()
            self.used_depmap = self.used_depmap or bool(res.rows)
        if own:
            notes.append("your profiles: " + ", ".join(own))
        # Vectors follow placeholder order: DepMap slots and own samples interleaved as written.
        text, vectors, depmap_iter = "", [], iter(depmap_vectors)
        for piece in re.split(f"({re.escape(OMICS)}|\ue000.\ue001)", marked):
            if piece == OMICS:
                text += piece
                vectors.append(next(depmap_iter))
            elif piece in slots:
                sample, first = slots[piece]
                text += f"{sample}{OMICS}" if first else sample
                if first:
                    vectors.append(self.profiles[sample])
            else:
                text += piece
        return Turn(text, vectors, notes)

    def send(self, message: str, **gen) -> tuple[Turn, Reply]:
        turn = self.prepare(message)
        messages = self.messages + [{"role": "user", "content": turn.text}]
        prompt = messages
        if self.used_depmap:  # DepMap profiles are in the conversation
            prompt = [{"role": "system", "content": DEPMAP_SYSTEM}] + messages
        reply = self.lm.chat(prompt, self.vectors + turn.vectors, **gen)
        self.messages = messages + [{"role": "assistant", "content": escape_omics(reply.answer)}]
        self.vectors = self.vectors + turn.vectors
        return turn, reply

    def reset(self) -> None:
        self.messages, self.vectors = [], []
        self.used_depmap = False


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="local checkpoint/repository copy (default: $THALAMUS_REASONING_HOME or the Hub)")
    ap.add_argument("--profiles", nargs="*", default=[], help=".npz files from tr-embed")
    ap.add_argument("--thinking", action="store_true", help="let the model reason before answering")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--no-depmap", action="store_true", help="do not resolve cell-line names")
    ap.add_argument("--repetition-penalty", type=float, default=1.05,
                    help="1.05 prevents rare greedy loops in long gene lists; 1.0 = plain greedy decoding")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    lm = ThalamusReasoning.load(args.model)
    conv = Conversation(lm, None if args.no_depmap else DepMap(args.model))
    for path in args.profiles:
        print(f"loaded {conv.load_profiles(path)} profiles from {path}")
    thinking = args.thinking
    dim, bold, off = "\033[2m", "\033[1m", "\033[0m"
    print("=" * 72 + "\nThalamus-Reasoning chat | /load FILE  /profiles  /lines TEXT  /think on|off  /reset  /exit\n"
          + "=" * 72)
    while True:
        try:
            msg = input(f"\n\001{bold}\002you>\001{off}\002 ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not msg:
            continue
        cmd, _, arg = msg.partition(" ")
        if cmd in ("/exit", "/quit"):
            break
        if cmd == "/reset":
            conv.reset()
            print("(conversation reset)")
            continue
        if cmd == "/load":
            try:
                print(f"loaded {conv.load_profiles(arg.strip())} profiles")
            except Exception as exc:  # noqa: BLE001 - report and keep the session
                print(f"could not load {arg}: {exc}")
            continue
        if cmd == "/profiles":
            print(", ".join(sorted(conv.profiles)) or "(none loaded)")
            continue
        if cmd == "/lines":
            if conv.depmap is None:
                print("(DepMap disabled)")
                continue
            table = conv.depmap.available(arg)
            print(table.head(40).to_string(index=False), f"\n({len(table)} lines)")
            continue
        if cmd == "/think":
            thinking = arg.strip() != "off"
            print(f"(thinking {'on' if thinking else 'off'})")
            continue
        try:
            turn, reply = conv.send(msg, thinking=thinking, max_new_tokens=args.max_new_tokens,
                                    repetition_penalty=args.repetition_penalty)
        except Exception as exc:  # noqa: BLE001
            log.exception("message failed: %s", exc)
            continue
        for note in turn.notes:
            print(f"{dim}({note}){off}")
        if reply.reasoning:
            print(f"\n{dim}{reply.reasoning}{off}")
        print(f"\n{bold}tr>{off} {reply.answer}")


if __name__ == "__main__":
    main()
