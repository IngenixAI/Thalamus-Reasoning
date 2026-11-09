---
name: thalamus-reasoning-prepare-profiles
description: Turn a user's own human transcriptomic data (GEO/SRA count tables, TPM tables, Seurat/Scanpy .h5ad, 10x matrices) into Thalamus-Reasoning input vectors with tr-embed, check them, and write prompts that use them. Use when someone wants to ask Thalamus-Reasoning about their own samples.
---

# Preparing your own profiles for Thalamus-Reasoning

Thalamus-Reasoning reads each sample as one 20,029-value vector. `tr-embed` builds it exactly as in training. Your
job is to hand it the right matrix: un-logged expression with gene IDs. It detects the rest from the data.

## 1. What the data is

| Your data | Supply | Detected as |
|---|---|---|
| Bulk RNA-seq (tissue, sorted cells, cell lines) | **TPM, FPKM, raw read counts or CPM** | `bulk` |
| Single cells (10x, Smart-seq…) | **Raw UMI/read counts** per cell; never log-normalised or scaled values | `single_cell` |
| Pseudobulk of single cells (cells summed per sample/cluster) | Summed **raw counts** (`--pseudobulk-by <obs column>` does the summing) | `single_cell` with `--pseudobulk-by`; for a pseudobulk table pass `--kind single_cell` |
| Microarray, proteomics, methylation, mouse data | Not supported: out of distribution. Say so rather than forcing it through. Mixed human/mouse GEO tables: keep only the human columns. | — |

What `tr-embed` detects (the report's `kind_reason` and `values_reason` say what it saw):
- **Bulk or single cell** (`--kind auto`): single cells leave most genes at zero (10x cells: 8–27% of genes
  detected), bulk profiles do not (70–88%). The two kinds get the same normalization; the kind only sets the
  first vector value (bulk = 0, single-cell = 1), which the model saw in training, so it must match the data.
- **Bulk values** (`--values auto`): read counts grow with gene length, TPM and FPKM do not. Values that still
  correlate with gene length (raw counts, CPM, DESeq2-normalised counts) are divided by GENCODE v49 gene
  lengths, i.e. converted to TPM, because the bulk training data was TPM. Each profile is then rescaled to
  10⁶, so FPKM and TPM give the same vector. Single-cell counts are used as they are, as in training.
- Pass `--kind` or `--values` only when the report warns that the call was unclear, or for pseudobulk tables
  (summed cells look like bulk).

**Logged or normalised data.** Values like `log2(TPM+1)`, `log1p(CPM)`, z-scores, or anything negative must be
undone first, e.g. `TPM = 2**x - 1` for log2(TPM+1). `tr-embed` refuses negative values, and profiles whose
values are all small non-integers (max < 50). For `.h5ad`
input it falls back to a counts layer (`counts`, `raw_counts`), then `adata.raw.X`, when `X` looks
log-transformed; pass `--layer` if raw counts sit in a layer with another name.

## 2. Genes: the 19,260-gene panel

- The gene axis is fixed: `preprocessing/gene_panel.tsv` in the model repository, with Ensembl gene IDs (no
  version suffix) in sorted order. It is mainly human protein-coding genes, plus mitochondrial rRNA/tRNA genes.
  19,252 of the 19,260 IDs are in GENCODE v45–v49. The remaining 8 are alt-haplotype genes: HLA-DRB3,
  HLA-DRB4, KIR genes, GSTT1 and others. Symbols are recent HGNC names (column `gene_symbol`). No single
  annotation release defines the panel; the panel file does.
- **Supported IDs:** Ensembl gene IDs, Entrez Gene IDs (common in GEO/NCBI-generated count tables), and HGNC
  symbols. Transcript IDs (`ENST…`), probe IDs and mouse IDs are rejected with a message saying so; aggregate
  them to genes first.
- **Ensembl IDs.** Version suffixes (`ENSG00000141510.18`) are stripped, and `_PAR_Y` rows
  are dropped because GENCODE duplicates them from chrX.
- **Symbols**: exact panel symbols first, then case-insensitive, then unambiguous HGNC previous symbols and
  aliases (`preprocessing/gene_aliases.tsv`). An alias is used only if no column already names that gene.
  Duplicate columns that land on the same gene are **summed**.
- Panel genes missing from your data are set to 0, as in training. The model cannot tell "not measured"
  from "not expressed", so many missing genes means an out-of-distribution input.
- Read the `.report.json` that `tr-embed` writes next to the output:
  - `panel_genes_found`: panel genes that your table has a column for. Expect over 15,000 for any
    whole-transcriptome table; below that the report warns. `panel_genes_detected` counts those with a non-zero value. For a handful of
    single cells it is much lower; that is normal.
  - `panel_mass_fraction`: the share of total expression on panel genes. A healthy value is over 0.6. A low
    value means non-human IDs, transcript IDs (`ENST…`) or probe IDs. Convert those first.
  - `aliases_used`: input symbols mapped through HGNC previous symbols or aliases. Hundreds are normal for
    older GEO tables. Symbols that HGNC gives to two genes (e.g. `QARS`, `ODZ3`) are left unmapped.
  - `renamed_samples`: sample IDs with spaces or symbols, renamed so that `@id` works in the chat.

## 3. Run it

```bash
# GEO-style bulk table: genes in rows, samples in columns, raw counts
tr-embed GSE123_counts.tsv.gz --out gse123.npz

# TPM table, samples in rows
tr-embed tpm.csv --out tpm.npz

# Single-cell h5ad, one vector per cell (keep it to the cells you will ask about)
tr-embed cells.h5ad --layer counts --out cells.npz

# Pseudobulk per donor × cell type
tr-embed atlas.h5ad --pseudobulk-by sample_celltype --out pb.npz
```

The output `.npz` has `embeddings` (n × 20,029 float32) and `sample_ids`. Geneformer-V2-104M is downloaded
from `ctheodoris/Geneformer` at a pinned revision. It takes about 1 s per 50 profiles on a GPU and also runs
on CPU. `--no-geneformer` fills that block with the training mean, which is faster but less accurate; use it
only for quick tests.

Tips:
- **Orientation** is auto-detected: the axis whose labels match more panel genes holds the genes. Force it with
  `--orientation genes_by_samples` if needed.
- **Too few genes:** under 5,000 panel genes is an error, since the input is out of distribution.
  `--allow-low-coverage` overrides it. Use it only if you understand the consequences.
- **First column / extra columns**: the first column is the row index. Non-numeric columns (gene names,
  descriptions) are dropped. If the IDs are in a named column, use `--gene-column`.
- **h5ad gene IDs**: if `var_names` are not gene IDs, the `var` column with the most panel matches is used
  (e.g. `gene_ids`, `feature_name`), or name it with `--gene-column`.
- **10x folders** (`matrix.mtx`, `features.tsv`): `pip install scanpy`, then load them with
  `scanpy.read_10x_mtx(..., var_names="gene_ids")` and save as `.h5ad` first.
- **Choosing cells**: subset before embedding, e.g. `adata[adata.obs.cell_type.isin([...])]`, or
  `sc.pp.subsample`, then `adata.write_h5ad(...)`.
- **Sample and pseudobulk IDs** become chat-safe: spaces and symbols turn into `_`
  (`CD14+ Monocytes` → `CD14_Monocytes`). The ID appears in the prompt. If it names the answer (a cell-type label), rename the groups neutrally
  (`group1`, …) when you test the model.
- **Single cells**: QC first. Drop empty droplets and doublets, and subsample to the cells you actually want
  to discuss.

## 4. Ask questions

```bash
tr-chat --profiles gse123.npz        # then write: "Compare @GSM1001 and @GSM1004 ..."
```

```python
from thalamus_reasoning import ThalamusReasoning, load_profiles
v = load_profiles("gse123.npz")
lm = ThalamusReasoning.load()
lm.ask("Sample A: <omics>\nSample B: <omics>\nWhich sample shows stronger interferon signalling?",
       [v["GSM1001"], v["GSM1004"]])
```

Prompting patterns that match training:
- Put each profile where it is discussed (`Sample A: <omics>`). Up to about 30 profiles per prompt is a safe range.
- Give known metadata in words (tissue, treatment, time point). The model combines text and profile.
- For classification, use `lm.classify(question, vectors, options)`. It lists the candidates in the prompt,
  as in the benchmarks, which is more reliable than open-ended questions. Scores are mean log-probability per answer token (the benchmark protocol), so keep the
  labels similar in length and specificity (`astrocyte` / `oligodendrocyte`, not `astrocyte` /
  `oligodendrocyte precursor cell of the adult cortex`).
- Turn `thinking=True` on for multi-step comparisons. Keep it off for short factual answers; the benchmark
  evaluations ran without thinking.

## 5. Sanity checks before trusting answers

1. In the chat, `@sample_id` refers to your samples. A typo that happens to match a DepMap line name is
   reported ("not among your loaded samples"). Read the notes printed above each answer.
2. Embed a profile whose answer you know, for example a sample with a clear tissue of origin. Ask
   "Which tissue is this sample from?" and check the answer.
3. The `.report.json` warnings list must be empty, or you should understand every warning.
4. If you have both read counts and TPM from your quantifier (Salmon, RSEM), use the TPM: its gene lengths
   are sample-specific, while the counts conversion uses one length per gene.
5. Treat answers as hypotheses. Thalamus-Reasoning is a research model and can be confidently wrong. Within-study
   comparisons (which samples are treated, the direction of marker changes) are its strength. Naming a
   perturbed gene, or the tissue of origin of cultured primary cells, is unreliable: state them in the prompt.
