"""DepMap 24Q4 cell lines as ready-made omics inputs.

The release ships one vector per DepMap cell line (``depmap/depmap_24q4_embeddings.npy``) and a name table
(``depmap/cell_lines.parquet``: Cellosaurus names, synonyms, DepMap / Cellosaurus IDs, tissue, disease).

:class:`CellLineResolver` finds cell-line mentions in free text and inserts an ``<omics>`` placeholder after
the first mention of each line:

- any spelling of a name or synonym works (``HeLa``, ``MCF-7``, ``hct 116``), as do ``ACH-000019`` and
  ``CVCL_0031``;
- names that are also ordinary words, numbers or gene symbols (``CI``, ``SEM``, ``GSS``, ``697``, ``Kelly``)
  only count when forced with ``@`` (``@RD``) or followed by "cell(s)" / "cell line" (``RD cells``);
- each line gets one vector per message; lines known to Cellosaurus but missing from DepMap expression data
  are reported.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from thalamus_reasoning.hub import release_path
from thalamus_reasoning.model import OMICS

WORD_NAMES = frozenset({
    "alexander", "becker", "birch", "fuji", "kelly", "lobe", "patton", "peer",
    "ramos", "sale", "scott", "simpson", "toledo",
})
_SEPARATOR = "[ ._/-]?"
_AT_MENTION = re.compile(r"(?<![\w@])@([A-Za-z0-9][\w.\-]*)")
_CELL_CONTEXT = re.compile(r"[\s-]+cells?\b|[\s-]+cell[\s-]+lines?\b", re.IGNORECASE)

# System message for prompts with DepMap profiles. The chat adds it; pass it as ``system=`` in your own code.
DEPMAP_SYSTEM = ("Cell lines mentioned by name come with their untreated baseline expression profile from the "
                 "DepMap 24Q4 cancer cell line collection. No treatment or genetic perturbation was applied.")


@dataclass(frozen=True)
class CellLine:
    name: str
    depmap_id: str | None
    row: int  # row in the embedding matrix, -1 if DepMap has no expression profile


@dataclass
class Resolution:
    text: str  # message with <omics> after the first mention of each resolved line
    rows: list[int] = field(default_factory=list)
    resolved: list[tuple[str, CellLine]] = field(default_factory=list)  # (as written, line)
    missing: list[tuple[str, CellLine]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)  # @mentions that match no cell line

    def notes(self) -> list[str]:
        notes = []
        if self.resolved:
            notes.append("omics: " + ", ".join(
                f"{w} → {line.name} ({line.depmap_id})" for w, line in self.resolved))
        if self.missing:
            notes.append("no DepMap expression for: " + ", ".join(
                f"{w} ({line.depmap_id})" for w, line in self.missing))
        if self.skipped:
            skipped = list(dict.fromkeys(self.skipped))
            notes.append(f"not treated as cell lines: {', '.join(skipped)} "
                         f"(write @{skipped[0]} or \"{skipped[0]} cells\" to use the cell line)")
        if self.unknown:
            notes.append("unknown cell line or sample (no profile attached): " + ", ".join(f"@{u}" for u in self.unknown)
                         + "; /lines TEXT searches the available lines")
        return notes


@dataclass
class _Entry:
    line: CellLine
    risky: bool
    exact_forms: set[str]
    official: bool


def normalize(name: str) -> str:
    return re.sub(r"[^0-9a-z]", "", name.casefold())


def _strip_separators(name: str) -> str:
    return re.sub(r"[ ._/-]", "", name)


def _is_risky(key: str, gene_symbols: set[str]) -> bool:
    return (len(key) <= 2 or (len(key) <= 3 and key.isalpha()) or key.isdigit()
            or key in WORD_NAMES or key in gene_symbols)


def _surface_pattern(form: str) -> str:
    return _SEPARATOR.join(re.escape(run) for run in re.findall(r"[A-Za-z]+|\d+", form))


class CellLineResolver:
    def __init__(self, table: pd.DataFrame, gene_symbols=()) -> None:
        genes = {normalize(g) for g in gene_symbols}
        candidates: dict[str, list[_Entry]] = defaultdict(list)
        patterns: dict[str, str] = {}
        for r in table.itertuples(index=False):
            line = CellLine(r.CellLine_name, r.DepMap_id, int(r.embedding_row))
            ids = {x for x in (r.DepMap_id, r.CVCL_id) if isinstance(x, str) and x}
            official = {r.CellLine_name, r.CellLine_stripped_name} - {None}
            synonyms = set(r.CellLine_synonyms) if r.CellLine_synonyms is not None else set()
            # Common short forms: "LNCaP clone FGC" -> "LNCaP", "HEL 92.1.7 [HEL]" style suffixes.
            for name in list(official):
                short = re.split(r" clone | \[|, ", name)[0].strip()
                if short and short != name:
                    synonyms.add(short)
            forms_by_key: dict[str, set[str]] = defaultdict(set)
            for form in official | synonyms | ids:
                if key := normalize(form):
                    forms_by_key[key].add(_strip_separators(form))
                    patterns.setdefault(_surface_pattern(form).casefold(), key)
            official_keys = {normalize(f) for f in official | ids}
            for key, forms in forms_by_key.items():
                candidates[key].append(_Entry(line, _is_risky(key, genes), forms, key in official_keys))
        # A key shared by several lines goes to the line it officially names; otherwise it is dropped.
        self._index: dict[str, _Entry] = {}
        for key, entries in candidates.items():
            if len(entries) > 1:
                entries = [e for e in entries if e.official]
            if len(entries) == 1:
                self._index[key] = entries[0]
        alternatives = sorted((p for p, key in patterns.items() if key in self._index),
                              key=lambda p: (len(patterns[p]), len(p)), reverse=True)
        self._pattern = re.compile(rf"(?<!\w)(@?)({'|'.join(alternatives)})(?!\w|[.,]\d)", re.IGNORECASE)

    def resolve(self, text: str) -> Resolution:
        result = Resolution(text="")
        seen: set[str] = set()
        parts: list[str] = []
        prev = 0
        for m in self._pattern.finditer(text):
            forced, written = bool(m.group(1)), m.group(2)
            entry = self._index[normalize(written)]
            if entry.risky:
                exact = _strip_separators(written) in entry.exact_forms
                # "@kelly" in any case; "Kelly cells" / "KELLY cells" but not "my cells" or "b cells".
                explicit = forced or (_CELL_CONTEXT.match(text, m.end()) and (exact or not written.islower())
                                      and len(normalize(written)) > 1)
                if not explicit:
                    # "RD is short" / "my": leave as text, and only note exact-case spellings of a line name.
                    key = normalize(written)
                    if exact and not (key.isdigit() or len(key) == 1):
                        result.skipped.append(written)
                    continue
            parts += [text[prev:m.start()], written]
            prev = m.end()
            line = entry.line
            ident = line.depmap_id or line.name
            if ident in seen:
                continue
            seen.add(ident)
            if line.row < 0:
                result.missing.append((written, line))
                continue
            parts.append(OMICS)
            result.rows.append(line.row)
            result.resolved.append((written, line))
        parts.append(text[prev:])
        result.text = "".join(parts)
        for m in _AT_MENTION.finditer(result.text):
            result.unknown.append(m.group(1))
        return result


class DepMap:
    """DepMap vectors plus the name resolver."""

    def __init__(self, source: str | None = None) -> None:
        self.embeddings = np.load(release_path("depmap/depmap_24q4_embeddings.npy", source), mmap_mode="r")
        self.table = pd.read_parquet(release_path("depmap/cell_lines.parquet", source))
        symbols = pd.read_csv(release_path("preprocessing/gene_panel.tsv", source), sep="\t", keep_default_na=False).gene_symbol
        self.resolver = CellLineResolver(self.table, symbols.astype(str))

    def resolve(self, text: str) -> tuple[Resolution, np.ndarray]:
        res = self.resolver.resolve(text)
        if not res.rows:
            return res, np.zeros((0, self.embeddings.shape[1]), np.float32)
        return res, np.asarray(self.embeddings[res.rows], dtype=np.float32)

    def available(self, query: str = "") -> pd.DataFrame:
        """Cell lines with a DepMap vector; ``query`` keeps rows whose name, IDs, tissue or disease contain it
        (case-insensitive)."""
        cols = ["CellLine_name", "DepMap_id", "CVCL_id", "tissue", "disease"]
        table = self.table[self.table.embedding_row >= 0][cols].sort_values("CellLine_name").reset_index(drop=True)
        if query:
            # Matched against the space-joined row, as the chat /lines command and the app search did.
            query = query.lower()
            hit = table.apply(lambda r: query in " ".join(map(str, r.values)).lower(), axis=1)
            table = table[hit].reset_index(drop=True)
        return table
