"""Turn expression profiles into Thalamus-Reasoning input vectors.

Every profile becomes one 20,029-float vector, built exactly like the training inputs:

====================  =========  ==============================================================
block                 width      computation
====================  =========  ==============================================================
data-type flag        1          ``bulk`` = 0, ``single_cell`` = 1 (cells and pseudobulk), z-scored
expression            19,260     panel genes -> scale to 1e6 total -> log1p -> minus per-gene mean
                                 -> divided by one global std
Geneformer-V2-104M    768        CLS embedding (second-to-last layer) of the rank-value-encoded
                                 profile, z-scored per dimension
====================  =========  ==============================================================

Bulk and single-cell profiles are processed the same way; only the flag differs. The one extra step: bulk
values that are not normalised for gene length (read counts, CPM) are converted to TPM first, because the bulk
training data was TPM (single-cell training data was raw counts). Both are detected from the data
(``detect_kind``, ``detect_values``) unless given. The means and standard deviations come from the training data
(``preprocessing/normalization_stats.npz``). Panel genes absent from the input are set to zero, as in training.

Command line::

    tr-embed profiles.h5ad --out vectors.npz
    tr-embed tpm.tsv --orientation genes_by_samples --out vectors.npz
"""

import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from thalamus_reasoning.hub import GENEFORMER_REPO, GENEFORMER_REVISION, GENEFORMER_SUBFOLDER, release_path

log = logging.getLogger(__name__)

N_GENES = 19260
VECTOR_DIM = 1 + N_GENES + 768
KINDS = {"bulk": 0.0, "single_cell": 1.0}
# Measured on DepMap TPM, GEO bulk counts and 10x Perturb-seq cells:
SINGLE_CELL_MAX_DETECTED = 0.45  # median share of genes detected per profile: 10x cells 8-27%, bulk 70-88%
LENGTH_BIAS_MIN = 0.2  # Spearman(value, gene length): bulk read counts 0.26-0.41, TPM 0.02-0.16, 10x UMIs -0.1
MIN_PANEL_GENES = 5000  # below this the profile is out of distribution (targeted panel, wrong IDs)
_ENSEMBL = re.compile(r"^(ENSG\d{11})(\.\d+)?(_PAR_Y)?$", re.IGNORECASE)
_ENTREZ = re.compile(r"^\d+$")
_OTHER_IDS = {
    "Ensembl transcript (ENST) IDs": re.compile(r"^ENST\d+", re.IGNORECASE),
    "mouse Ensembl (ENSMUSG) IDs": re.compile(r"^ENSMUSG\d+", re.IGNORECASE),
    "microarray probe IDs": re.compile(r"^(\d+_(s_|x_)?at|ILMN_\d+|A_\d+_P\d+)$", re.IGNORECASE),
}


class InputError(ValueError):
    """Input that cannot be turned into in-distribution profiles; the message says how to fix it."""


@dataclass
class AlignmentReport:
    """How the input genes map onto the 19,260-gene panel."""

    input_genes: int
    mapped_input_genes: int
    panel_genes_found: int
    duplicates_summed: int
    id_type: str
    unmapped_examples: list[str] = field(default_factory=list)
    panel_mass_fraction: float | None = None  # share of the input total that lands on panel genes
    kind: str | None = None  # bulk / single_cell
    kind_reason: str | None = None
    values: str | None = None  # bulk: tpm (length-normalised) or counts (converted to TPM)
    values_reason: str | None = None
    panel_genes_detected: int | None = None  # panel genes with a non-zero value in at least one profile
    matrix_source: str | None = None  # X, raw.X or a layer (h5ad input)
    aliases_used: int = 0  # input symbols mapped through HGNC previous symbols / aliases
    renamed_samples: dict = field(default_factory=dict)  # original -> chat-safe sample ID
    _notes: list[str] = field(default_factory=list, repr=False)  # warnings from kind / value detection

    @property
    def panel_coverage(self) -> float:
        return self.panel_genes_found / N_GENES

    def warnings(self) -> list[str]:
        out = []
        if self.panel_coverage < 0.6:
            out.append(
                f"only {self.panel_genes_found:,} of {N_GENES:,} panel genes found "
                f"({self.panel_coverage:.0%}); targeted panels and microarrays are out of distribution"
            )
        elif self.panel_genes_found < 15000:
            out.append(
                f"{self.panel_genes_found:,} of {N_GENES:,} panel genes found; whole-transcriptome tables usually "
                "have over 15,000. Missing genes are read as zero expression; check unmapped_examples"
            )
        out += self._notes
        if self.panel_mass_fraction is not None and self.panel_mass_fraction < 0.6:
            out.append(
                f"panel genes carry only {self.panel_mass_fraction:.0%} of the input total; "
                "check that gene IDs are human Ensembl, Entrez or HGNC symbols"
            )
        return out

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        d["panel_coverage"] = round(self.panel_coverage, 4)
        d["warnings"] = self.warnings()
        return d


class GenePanel:
    """The fixed gene axis: 19,260 human genes, Ensembl IDs without version, sorted."""

    def __init__(self, source: str | None = None) -> None:
        table = pd.read_csv(release_path("preprocessing/gene_panel.tsv", source), sep="\t", keep_default_na=False)
        if len(table) != N_GENES:
            raise ValueError(f"gene panel has {len(table)} rows, expected {N_GENES}")
        self.table = table
        self.ensembl = table["ensembl_gene_id"].to_numpy(str)
        self.symbols = table["gene_symbol"].to_numpy(str)
        self._by_id = {g: i for i, g in enumerate(self.ensembl)}
        self._by_entrez = {str(e): i for i, e in enumerate(table["entrez_id"]) if str(e)}
        self._by_symbol = {s: i for i, s in enumerate(self.symbols)}
        self._by_symbol_upper = {s.upper(): i for i, s in enumerate(self.symbols)}
        # HGNC previous symbols and aliases (unambiguous ones only) for older annotations.
        aliases = pd.read_csv(release_path("preprocessing/gene_aliases.tsv", source), sep="\t", keep_default_na=False)
        self._by_alias = {a.upper(): self._by_id[e] for a, e in zip(aliases.symbol, aliases.ensembl_gene_id)}
        self.lengths = table["gene_length"].to_numpy(float)

    def match_count(self, labels) -> int:
        """How many labels name a panel gene (any supported ID type); used to detect orientation."""
        return int((self.index_of([str(x) for x in labels])[0] >= 0).sum())

    def index_of(self, genes: list[str]) -> tuple[np.ndarray, str]:
        """Panel index for every input gene (-1 if not in the panel) and the ID type used."""
        idx, id_type, _ = self._index(genes)
        return idx, id_type

    def _index(self, genes: list[str]) -> tuple[np.ndarray, str, int]:
        genes = [str(g).strip() for g in genes]
        if not genes:
            return np.array([], dtype=int), "none", 0
        n_ensembl = sum(bool(_ENSEMBL.match(g)) for g in genes)
        n_entrez = sum(bool(_ENTREZ.match(g)) for g in genes)
        if n_entrez >= 0.5 * len(genes):
            return np.array([self._by_entrez.get(g, -1) for g in genes]), "entrez", 0
        if n_ensembl >= 0.5 * len(genes):
            idx = []
            for g in genes:
                m = _ENSEMBL.match(g)
                # GENCODE PAR_Y copies share the stable ID of their X-chromosome gene; drop them.
                idx.append(-1 if m is None or m.group(3) else self._by_id.get(m.group(1).upper(), -1))
            return np.array(idx), "ensembl", 0
        idx = [self._by_symbol.get(g, self._by_symbol_upper.get(g.upper(), -1)) for g in genes]
        # Aliases only fill panel genes no input column names directly (no double counting).
        direct, n_alias = set(idx), 0
        for k, g in enumerate(genes):
            if idx[k] < 0 and (i := self._by_alias.get(g.upper(), -1)) not in direct:
                idx[k] = i
                n_alias += i >= 0
        return np.array(idx), "symbol", n_alias

    def align(self, matrix, genes: list[str], min_genes: int = MIN_PANEL_GENES) -> tuple[np.ndarray, AlignmentReport]:
        """Samples x input genes -> samples x panel genes (float64). Duplicates are summed."""
        idx, id_type, n_alias = self._index(genes)
        keep = idx >= 0
        if keep.sum() < max(min_genes, 1):
            sample = [str(g) for g in genes[:5]]
            hint = next((f" They look like {name}; convert them to gene-level Ensembl IDs or symbols first."
                         for name, pat in _OTHER_IDS.items() if sum(bool(pat.match(str(g))) for g in genes[:200]) > 100), "")
            raise InputError(
                f"only {int(keep.sum())} of {len(genes):,} gene labels match the 19,260-gene panel "
                f"(read as {id_type} IDs; first labels: {sample}).{hint} Gene labels must be human Ensembl gene "
                "IDs, Entrez IDs or HGNC symbols. If the table's rows and columns are swapped (genes along the other "
                "axis), pass --orientation genes_by_samples or samples_by_genes; --gene-column names a column "
                "holding the gene IDs."
            )
        import scipy.sparse as sp

        matrix = matrix.tocsc() if sp.issparse(matrix) else np.asarray(matrix)
        sub = matrix[:, np.flatnonzero(keep)]
        sub = sub.toarray() if sp.issparse(sub) else sub
        sub = np.asarray(sub, dtype=np.float64)
        if not np.isfinite(sub).all() or (sub < 0).any():
            raise InputError("expression values must be non-negative and finite: raw counts or TPM, "
                             "not log-transformed, scaled or z-scored values")
        row_max = sub.max(1)
        logged = (row_max < 50) & ~np.all(np.isclose(sub, np.round(sub)), axis=1)
        if logged.any():
            raise InputError(
                f"{int(logged.sum())} profiles have only small non-integer values (max {row_max[logged].min():.3g}); this looks "
                "log-transformed (e.g. log2(TPM+1) or log1p(CPM)). Undo the log first (TPM = 2**x - 1), or for "
                ".h5ad use the raw counts layer (--layer counts)."
            )
        out = np.zeros((sub.shape[0], N_GENES))
        np.add.at(out.T, idx[keep], sub.T)
        totals = np.asarray(matrix.sum(axis=1)).ravel()
        mass = float(np.median(out.sum(1) / np.maximum(totals, 1e-12))) if totals.size else None
        detected = int((out > 0).any(0).sum())
        report = AlignmentReport(
            input_genes=len(genes),
            mapped_input_genes=int(keep.sum()),
            panel_genes_found=len(set(idx[keep].tolist())),
            duplicates_summed=int(keep.sum() - len(set(idx[keep].tolist()))),
            id_type=id_type,
            unmapped_examples=[g for g, k in zip(genes, keep) if not k][:10],
            panel_mass_fraction=mass,
            panel_genes_detected=detected,
            aliases_used=n_alias,
        )
        report._found = np.unique(idx[keep])  # panel genes present in the input
        return out, report


class GeneformerEncoder:
    """Geneformer-V2-104M CLS embeddings, matching the original ``EmbExtractor`` settings.

    Rank-value encoding: each gene is divided by the profile total, scaled by 10,000 and by its
    Geneformer median; non-zero genes are ranked by that value (at most 4,094), wrapped in
    ``<cls>`` / ``<eos>``. The embedding is the ``<cls>`` state of the second-to-last layer.
    """

    def __init__(self, source: str | None = None, device: str | None = None, batch_size: int = 16) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import BertForMaskedLM

        table = pd.read_csv(release_path("preprocessing/geneformer_v2_tokens.tsv", source), sep="\t")
        special = json.loads(release_path("preprocessing/geneformer_v2_special_tokens.json", source).read_text())
        usable = table["geneformer_token"].to_numpy() >= 0
        self.columns = np.flatnonzero(usable)
        self.tokens = table["geneformer_token"].to_numpy()[usable]
        self.medians = table["geneformer_median"].to_numpy(float)[usable]
        self.cls, self.eos, self.pad = special["<cls>"], special["<eos>"], special["<pad>"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        root = snapshot_download(
            GENEFORMER_REPO, revision=GENEFORMER_REVISION, allow_patterns=[f"{GENEFORMER_SUBFOLDER}/*"],
        )
        model = BertForMaskedLM.from_pretrained(
            Path(root) / GENEFORMER_SUBFOLDER, output_hidden_states=True, attn_implementation="eager",
        )
        self.model = model.to(self.device).eval()
        self.max_len = model.config.max_position_embeddings
        self.layer = model.config.num_hidden_layers - 1  # emb_layer=-1 in Geneformer terms
        self.batch_size = batch_size

    def tokenize(self, profile: np.ndarray) -> np.ndarray | None:
        values = np.asarray(profile, dtype=np.float32)
        total = values.sum(dtype=np.float32)
        if total <= 0:
            return None
        scaled = (values[self.columns] / total * 10_000) / self.medians
        nonzero = np.flatnonzero(scaled)
        if nonzero.size == 0:
            return None
        ranked = self.tokens[nonzero][np.argsort(-scaled[nonzero])][: self.max_len - 2]
        return np.concatenate([[self.cls], ranked, [self.eos]])

    def __call__(self, panel_matrix: np.ndarray) -> np.ndarray:
        import torch

        out = np.zeros((panel_matrix.shape[0], self.model.config.hidden_size), dtype=np.float32)
        tokenized = [(i, t) for i, row in enumerate(panel_matrix) if (t := self.tokenize(row)) is not None]
        if len(tokenized) < panel_matrix.shape[0]:
            log.warning("%d profiles have no Geneformer genes; their Geneformer block is zero",
                        panel_matrix.shape[0] - len(tokenized))
        tokenized.sort(key=lambda it: -len(it[1]))
        for start in range(0, len(tokenized), self.batch_size):
            chunk = tokenized[start:start + self.batch_size]
            width = max(len(t) for _, t in chunk)
            ids = torch.full((len(chunk), width), self.pad, dtype=torch.long)
            mask = torch.zeros((len(chunk), width), dtype=torch.long)
            for j, (_, t) in enumerate(chunk):
                ids[j, : len(t)] = torch.from_numpy(t.astype(np.int64))
                mask[j, : len(t)] = 1
            with torch.no_grad():
                hidden = self.model(
                    input_ids=ids.to(self.device), attention_mask=mask.to(self.device),
                ).hidden_states[self.layer]
            cls = hidden[:, 0].float().cpu().numpy()
            for j, (i, _) in enumerate(chunk):
                out[i] = cls[j]
        return out


class Preprocessor:
    """Expression matrix (samples x genes) -> Thalamus-Reasoning input vectors (samples x 20,029)."""

    def __init__(self, source: str | None = None, device: str | None = None,
                 use_geneformer: bool = True) -> None:
        self.source = source
        self.panel = GenePanel(source)
        stats = np.load(release_path("preprocessing/normalization_stats.npz", source))
        self.stats = {k: stats[k].astype(np.float64) for k in stats.files}
        self.device = device
        self._geneformer = None
        self.use_geneformer = use_geneformer

    @property
    def geneformer(self) -> GeneformerEncoder:
        if self._geneformer is None:
            self._geneformer = GeneformerEncoder(self.source, self.device)
        return self._geneformer

    def counts_to_tpm(self, panel_counts: np.ndarray) -> np.ndarray:
        """Bulk read counts -> TPM using GENCODE union-exon gene lengths (approximate)."""
        rate = panel_counts / self.panel.lengths
        return rate / np.maximum(rate.sum(1, keepdims=True), 1e-12) * 1e6

    def expression_block(self, panel_matrix: np.ndarray) -> np.ndarray:
        totals = panel_matrix.sum(1, keepdims=True)
        scaled = panel_matrix / np.maximum(totals, 1e-6) * 1e6
        return (np.log1p(scaled) - self.stats["expression_center"]) / self.stats["expression_scale"]

    def flag_block(self, kind: str, n: int) -> np.ndarray:
        value = (KINDS[kind] - self.stats["scale_indicator_center"]) / self.stats["scale_indicator_scale"]
        return np.full((n, 1), value[0])

    def geneformer_block(self, panel_matrix: np.ndarray) -> np.ndarray:
        if not self.use_geneformer:
            # Training mean: the most neutral fill when Geneformer cannot be run (degrades accuracy).
            return np.zeros((panel_matrix.shape[0], 768))
        raw = self.geneformer(panel_matrix)
        return (raw - self.stats["geneformer_center"]) / self.stats["geneformer_scale"]

    def embed(self, matrix, genes: list[str], kind: str = "auto", values: str = "auto",
              min_genes: int = MIN_PANEL_GENES) -> tuple[np.ndarray, AlignmentReport]:
        """``matrix``: samples x genes. Bulk: TPM, FPKM, read counts or CPM; single_cell: raw counts (cells or
        pseudobulk). ``kind`` and ``values`` are detected from the data when ``"auto"``.

        Both kinds share one pipeline and differ only in the flag; bulk values that are not normalised for gene
        length are converted to TPM first.
        """
        if kind not in ("auto", *KINDS):
            raise ValueError(f"kind must be auto or one of {sorted(KINDS)}")
        panel_matrix, report = self.panel.align(matrix, genes, min_genes)
        if empty := int((panel_matrix.sum(1) == 0).sum()):
            raise InputError(f"{empty} profiles have no expression on panel genes")
        sample = panel_matrix[:: max(1, len(panel_matrix) // 200)]  # detection looks at up to ~200 profiles
        report.kind, report.kind_reason = (kind, "given") if kind != "auto" else detect_kind(sample, report)
        if report.kind == "bulk":
            report.values, report.values_reason = (
                (values, "given") if values != "auto" else detect_values(sample, self.panel.lengths, report))
            if report.values == "counts":
                panel_matrix = self.counts_to_tpm(panel_matrix)
            elif (kind == "auto" and np.allclose(sample, np.round(sample))  # integers without length bias
                  and not np.all(np.abs(sample.sum(1) / 1e6 - 1) < 0.3)):  # and not rounded TPM
                report._notes.append("integer values without gene-length bias were read as bulk TPM; if these are "
                                     "single cells summed per group (pseudobulk), pass --kind single_cell")
        elif not np.allclose(sample, np.round(sample)):
            totals = sample.sum(1)
            if totals.std() > 0.01 * totals.mean():  # equal totals = normalize_total, same after rescaling
                report._notes.append("single-cell values are not raw counts; the model was trained on raw UMI/read "
                                     "counts (scaled or batch-corrected values shift the profile)")
        log.info("kind: %s (%s)%s", report.kind, report.kind_reason,
                 f"; values: {report.values} ({report.values_reason})" if report.values else "")
        vectors = np.concatenate([
            self.flag_block(report.kind, len(panel_matrix)),
            self.expression_block(panel_matrix),
            self.geneformer_block(panel_matrix),
        ], axis=1).astype(np.float32)
        assert vectors.shape[1] == VECTOR_DIM
        return vectors, report

    def embed_file(self, path: str | Path, kind: str = "auto", values: str = "auto", pseudobulk_by: str | None = None,
                   orientation: str = "auto", layer: str | None = None, gene_column: str | None = None,
                   min_genes: int = MIN_PANEL_GENES, max_profiles: int | None = None,
                   ) -> tuple[np.ndarray, list[str], AlignmentReport]:
        """Read a table or .h5ad, optionally sum cells per ``obs`` group, and embed every profile.

        Returns (vectors, chat-safe sample IDs, report). Problems with the input raise ``InputError``.
        """
        try:
            matrix, samples, genes, obs = read_matrix(path, self.panel, orientation, layer, gene_column)
        except (OSError, KeyError, ValueError) as exc:
            raise InputError(f"could not read {Path(path).name}: {exc}") from None
        if pseudobulk_by:
            if obs is None or pseudobulk_by not in obs:
                raise InputError(f"pseudobulk needs an .h5ad input with an obs column '{pseudobulk_by}'")
            if _looks_logged(matrix):
                raise InputError("the cell matrix looks log-normalised; pseudobulk needs raw counts "
                                 "(a counts layer or raw.X)")
            matrix, samples = pseudobulk(matrix, obs[pseudobulk_by])
            kind = "single_cell"  # summed cells were single-cell data in training
        if max_profiles and len(samples) > max_profiles:
            raise InputError(f"{len(samples)} profiles; at most {max_profiles} can be embedded at a time")
        vectors, report = self.embed(matrix, genes, kind, values, min_genes)
        if pseudobulk_by:
            report.kind_reason = f"pseudobulk of cells by '{pseudobulk_by}'"
        report.matrix_source = obs.attrs.get("tr_matrix_source") if obs is not None else None
        samples, report.renamed_samples = chat_safe_ids(samples)
        return vectors, samples, report


def detect_kind(panel_matrix: np.ndarray, report: AlignmentReport) -> tuple[str, str]:
    """Bulk or single cell, from the share of genes detected: single cells leave most genes at zero."""
    detected = float(np.median((panel_matrix[:, report._found] > 0).mean(1)))
    kind = "single_cell" if detected < SINGLE_CELL_MAX_DETECTED else "bulk"
    if abs(detected - SINGLE_CELL_MAX_DETECTED) < 0.1:
        report._notes.append(f"{detected:.0%} of genes detected per profile is between typical single cells and "
                             "bulk; check the detected kind and pass --kind if it is wrong")
    return kind, f"auto: {detected:.0%} of genes detected per profile (single cells < 45%, bulk above)"


def detect_values(panel_matrix: np.ndarray, lengths: np.ndarray, report: AlignmentReport) -> tuple[str, str]:
    """Bulk values normalised for gene length (TPM, FPKM) or not (read counts, CPM, normalised counts).

    Read counts grow with gene length, so their correlation with length shows whether that was divided out.
    """
    log_len = np.log(lengths)
    bias = []
    for row in panel_matrix:
        on = row > 0
        bias.append(np.corrcoef(_rank(row[on]), _rank(log_len[on]))[0, 1])
    bias = float(np.median(bias))
    values = "counts" if bias >= LENGTH_BIAS_MIN else "tpm"
    if abs(bias - LENGTH_BIAS_MIN) < 0.05:
        report._notes.append(f"unclear whether values are normalised for gene length (correlation {bias:.2f}); "
                             "pass --values tpm (TPM/FPKM) or --values counts (read counts/CPM) if you know")
    what = "not normalised for gene length (read counts or similar)" if values == "counts" else "length-normalised (TPM or FPKM)"
    return values, f"auto: values look {what}; correlation with gene length {bias:.2f}"


def _rank(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)


# --------------------------------------------------------------------------- file input
def read_matrix(path: str | Path, panel: GenePanel, orientation: str = "auto", layer: str | None = None,
                gene_column: str | None = None):
    """Read an expression table and return (samples x genes matrix, sample IDs, gene IDs, obs).

    Tables: the first column is the row index and non-numeric columns are dropped. ``orientation="auto"``
    picks the axis whose labels match more panel genes.
    """
    path = Path(path)
    if path.suffix == ".h5ad":
        import anndata as ad

        adata = ad.read_h5ad(path)
        matrix, source = (adata.layers[layer], f"layers['{layer}']") if layer else (adata.X, "X")
        var = adata.var
        if layer is None and _looks_logged(matrix):
            counts_layer = next((k for k in ("counts", "raw_counts", "count") if k in adata.layers), None)
            if counts_layer:
                log.warning("adata.X looks log-transformed; using adata.layers['%s']", counts_layer)
                matrix, source = adata.layers[counts_layer], f"layers['{counts_layer}']"
            elif adata.raw is not None:
                log.warning("adata.X looks log-transformed; using adata.raw.X")
                matrix, var, source = adata.raw.X, adata.raw.var, "raw.X"
        adata.obs.attrs["tr_matrix_source"] = source
        genes = var[gene_column].astype(str).tolist() if gene_column else var.index.astype(str).tolist()
        if gene_column is None and panel.match_count(genes) < MIN_PANEL_GENES:
            # var_names are often symbols with Ensembl IDs in a column (or vice versa): use the best column.
            best = max(((c, panel.match_count(var[c].astype(str))) for c in var.columns
                        if var[c].dtype == object or str(var[c].dtype) in ("category", "string")),
                       key=lambda t: t[1], default=(None, 0))
            if best[1] > panel.match_count(genes):
                log.info("using var['%s'] for gene IDs (%d panel matches)", best[0], best[1])
                genes = var[best[0]].astype(str).tolist()
        return matrix, adata.obs_names.astype(str).tolist(), genes, adata.obs
    sep = "," if path.suffix == ".csv" or path.name.endswith(".csv.gz") else "\t"
    frame = pd.read_csv(path, sep=sep, index_col=0)
    if gene_column:
        frame = frame.set_index(gene_column)
    frame = frame.select_dtypes("number")
    if orientation == "auto":
        rows_are_genes = panel.match_count(frame.index) > panel.match_count(frame.columns)
        orientation = "genes_by_samples" if rows_are_genes else "samples_by_genes"
        log.info("%s: %d x %d table read as %s", path.name, *frame.shape, orientation)
    if orientation == "genes_by_samples":
        frame = frame.T
    return frame.to_numpy(float), frame.index.astype(str).tolist(), frame.columns.astype(str).tolist(), None


def _looks_logged(matrix) -> bool:
    import scipy.sparse as sp

    sample = matrix[: min(200, matrix.shape[0])]
    data = sample.data if sp.issparse(sample) else np.asarray(sample).ravel()
    data = data[data > 0]
    return data.size > 0 and data.max() < 50 and not np.allclose(data[:5000], np.round(data[:5000]))


def load_profiles(path: str | Path) -> dict[str, np.ndarray]:
    """Vectors written by ``tr-embed`` (or ``np.savez(path, embeddings=..., sample_ids=...)``), by sample ID."""
    data = np.load(path, allow_pickle=False)
    vectors = data["embeddings"]
    if vectors.ndim != 2 or vectors.shape[1] != VECTOR_DIM:
        raise ValueError(f"{path}: vectors have shape {vectors.shape}, expected (n, {VECTOR_DIM})")
    return dict(zip(data["sample_ids"].astype(str), vectors))


def chat_safe_ids(ids: list[str]) -> tuple[list[str], dict[str, str]]:
    """Sample IDs usable as ``@id`` in the chat: other characters become ``_`` ("CD14+ Mono" -> "CD14_Mono").

    Returns the new IDs and a map of the renamed ones; raises if renaming makes IDs collide.
    """
    safe = [re.sub(r"[^\w.:\-]+", "_", str(n)).strip("_.:-") or "sample" for n in ids]
    if len(set(safe)) != len(safe):
        raise InputError("sample IDs collide after replacing spaces/symbols with '_'; rename the samples")
    return safe, {str(a): b for a, b in zip(ids, safe) if str(a) != b}


def pseudobulk(matrix, groups: pd.Series) -> tuple[np.ndarray, list[str]]:
    """Sum raw counts of cells within each group."""
    import scipy.sparse as sp

    labels = groups.astype(str).to_numpy()
    names = sorted(set(labels))
    rows = []
    for name in names:
        part = matrix[np.flatnonzero(labels == name)]
        rows.append(np.asarray(part.sum(axis=0)).ravel() if sp.issparse(part) else part.sum(0))
    return np.vstack(rows), names


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help=".h5ad, .csv or .tsv (optionally .gz)")
    ap.add_argument("--kind", default="auto", choices=["auto", *KINDS],
                    help="bulk RNA-seq, or single_cell (cells or pseudobulk of cells); auto detects it from the "
                         "share of genes detected")
    ap.add_argument("--values", default="auto", choices=["auto", "tpm", "counts"],
                    help="bulk only: tpm = TPM/FPKM; counts = read counts or CPM, converted to TPM with gene "
                         "lengths; auto detects it from the correlation with gene length")
    ap.add_argument("--orientation", default="auto", choices=["auto", "samples_by_genes", "genes_by_samples"])
    ap.add_argument("--layer", help="h5ad layer with raw counts (default: X, or raw.X if X is log-transformed)")
    ap.add_argument("--gene-column", help="var column (h5ad) or table column with gene IDs / symbols")
    ap.add_argument("--pseudobulk-by", help="h5ad obs column; sum cells per group before embedding")
    ap.add_argument("--no-geneformer", action="store_true", help="skip Geneformer (lower accuracy)")
    ap.add_argument("--allow-low-coverage", action="store_true",
                    help=f"embed even if fewer than {MIN_PANEL_GENES:,} panel genes are found (out of distribution)")
    ap.add_argument("--model", help="local copy of the model repository (default: $THALAMUS_REASONING_HOME or the Hub)")
    ap.add_argument("--device")
    ap.add_argument("--out", required=True, help="output .npz with 'embeddings' and 'sample_ids'")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("httpx", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    pre = Preprocessor(args.model, args.device, use_geneformer=not args.no_geneformer)
    try:
        vectors, samples, report = pre.embed_file(
            args.input, args.kind, args.values, args.pseudobulk_by, args.orientation, args.layer, args.gene_column,
            min_genes=1 if args.allow_low_coverage else MIN_PANEL_GENES)
    except InputError as exc:
        raise SystemExit(f"error: {exc}") from None
    if report.renamed_samples:
        log.info("%d sample IDs made chat-safe, e.g. %s", len(report.renamed_samples),
                 next(iter(report.renamed_samples.items())))
    np.savez(args.out, embeddings=vectors, sample_ids=np.array(samples))
    summary = {"samples": len(samples), **report.to_dict()}
    Path(args.out).with_suffix(".report.json").write_text(json.dumps(summary, indent=2) + "\n")
    for warning in report.warnings():
        log.warning(warning)
    log.info("%d vectors (%d dims) -> %s; %s panel genes found",
             len(vectors), vectors.shape[1], args.out, f"{report.panel_genes_found:,}")


if __name__ == "__main__":
    main()
