"""Gene alignment and normalization; uses the released preprocessing files ($THALAMUS_REASONING_HOME or the Hub)."""

import os

import numpy as np
import pytest

from thalamus_reasoning.preprocess import N_GENES, VECTOR_DIM, InputError, Preprocessor

pytestmark = pytest.mark.skipif(not os.environ.get("THALAMUS_REASONING_HOME")
                                and not os.environ.get("THALAMUS_REASONING_ONLINE"),
                                reason="set THALAMUS_REASONING_HOME (local repo copy) or THALAMUS_REASONING_ONLINE=1")


@pytest.fixture(scope="module")
def pre():
    return Preprocessor(use_geneformer=False)


def test_ensembl_versions_and_symbols_agree(pre):
    ids = pre.panel.ensembl[:500].tolist()
    syms = pre.panel.symbols[:500].tolist()
    x = np.random.default_rng(0).poisson(5, size=(3, 500)).astype(float)
    a, ra = pre.panel.align(x, [f"{g}.7" for g in ids], min_genes=1)
    b, rb = pre.panel.align(x, syms, min_genes=1)
    assert ra.id_type == "ensembl" and rb.id_type == "symbol"
    assert np.array_equal(a, b) and ra.panel_genes_found == 500


def test_alias_fills_only_missing_genes(pre):
    # TM4SF6 is a previous symbol of TSPAN6 (panel index 0)
    x = np.array([[3.0, 4.0]])
    one, _ = pre.panel.align(x, ["TM4SF6", "TNMD"], min_genes=1)
    both, _ = pre.panel.align(np.array([[3.0, 4.0, 5.0]]), ["TM4SF6", "TNMD", "TSPAN6"], min_genes=1)
    assert one[0, 0] == 3.0 and both[0, 0] == 5.0


def test_par_y_and_duplicates(pre):
    g = pre.panel.ensembl[10]
    out, rep = pre.panel.align(np.array([[1.0, 2.0, 9.0]]), [g, g + ".3", g + ".3_PAR_Y"], min_genes=1)
    assert out[0, 10] == 3.0 and rep.duplicates_summed == 1


def test_vector_layout(pre):
    x = np.random.default_rng(1).gamma(1.0, 10.0, size=(2, N_GENES))
    v, rep = pre.embed(x, pre.panel.ensembl.tolist(), "bulk")
    assert v.shape == (2, VECTOR_DIM) and v.dtype == np.float32
    assert np.allclose(v[:, 19261:], 0)  # --no-geneformer: training mean
    sc, _ = pre.embed(x, pre.panel.ensembl.tolist(), "single_cell")
    assert v[0, 0] < 0 < sc[0, 0]


def test_rejects_logged_values(pre):
    with pytest.raises(InputError):
        pre.embed(np.array([[-1.0, 2.0]]), pre.panel.ensembl[:2].tolist(), "bulk", min_genes=1)
    x = np.random.default_rng(2).gamma(1.0, 10.0, size=(2, N_GENES))
    with pytest.raises(InputError, match="log"):
        pre.embed(np.log2(x + 1), pre.panel.ensembl.tolist(), "bulk")


def test_entrez_ids_and_orientation(pre, tmp_path):
    import pandas as pd

    from thalamus_reasoning.preprocess import read_matrix

    entrez = pre.panel.table.entrez_id.astype(str)
    keep = entrez != ""
    x = np.random.default_rng(3).poisson(20, size=(int(keep.sum()), 3))
    table = pd.DataFrame(x, index=entrez[keep].tolist(), columns=["GSM1", "GSM2", "GSM3"])
    table.to_csv(tmp_path / "t.tsv", sep="\t")
    m, samples, genes, _ = read_matrix(tmp_path / "t.tsv", panel=pre.panel)
    assert samples == ["GSM1", "GSM2", "GSM3"]
    v, rep = pre.embed(m, genes, "bulk", "counts")
    assert rep.id_type == "entrez" and rep.values == "counts" and rep.panel_genes_found > 19000


def test_too_few_genes_is_an_error(pre):
    with pytest.raises(InputError, match="only"):
        pre.embed(np.ones((1, 100)) * 100, pre.panel.ensembl[:100].tolist(), "bulk")


def test_detects_kind_and_values(pre):
    rng = np.random.default_rng(3)
    genes = pre.panel.ensembl.tolist()
    expr = rng.gamma(0.5, 20.0, size=(4, N_GENES))
    tpm = expr / expr.sum(1, keepdims=True) * 1e6
    reads = rng.poisson(tpm * pre.panel.lengths / 200).astype(float)  # bulk reads grow with gene length
    cells = rng.poisson(tpm / 200).astype(float)  # ~5k UMIs per cell: most genes zero
    cases = {"tpm": (tpm, "bulk", "tpm"), "rounded tpm": (np.round(tpm), "bulk", "tpm"),
             "read counts": (reads, "bulk", "counts"), "cpm": (reads / reads.sum(1, keepdims=True) * 1e6, "bulk", "counts"),
             "cells": (cells, "single_cell", None)}
    for name, (x, kind, values) in cases.items():
        _, rep = pre.embed(x, genes)
        assert (rep.kind, rep.values) == (kind, values), (name, rep.kind_reason, rep.values_reason)
        assert not rep.warnings(), (name, rep.warnings())
    _, rep = pre.embed(reads, genes, "single_cell")  # given kinds are kept
    assert (rep.kind, rep.kind_reason, rep.values) == ("single_cell", "given", None)


def test_old_symbols_and_alias_count(pre):
    # TAZ: previous symbol of TAFAZZIN, alias of WWTR1 -> TAFAZZIN; NOV -> CCN3; mouse-style case is fine
    idx, _ = pre.panel.index_of(["TAZ", "Nov", "Gfap"])
    assert [pre.panel.symbols[i] for i in idx] == ["TAFAZZIN", "CCN3", "GFAP"]
    _, rep = pre.panel.align(np.ones((1, 3)), ["TAZ", "Nov", "Gfap"], min_genes=1)
    assert rep.aliases_used == 2


def test_chat_safe_ids():
    from thalamus_reasoning.preprocess import chat_safe_ids

    ids, renamed = chat_safe_ids(["GSM1", "59yo tumor periphery astro", "CD14+ Monocytes"])
    assert ids == ["GSM1", "59yo_tumor_periphery_astro", "CD14_Monocytes"] and len(renamed) == 2
    with pytest.raises(InputError):
        chat_safe_ids(["a b", "a_b"])


def test_embed_file_pseudobulk_and_errors(pre, tmp_path):
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(4)
    cells = ad.AnnData(rng.poisson(0.3, size=(6, N_GENES)).astype(np.float32),
                       obs=pd.DataFrame({"group": ["T cell"] * 3 + ["B cell"] * 3}, index=[f"c{i}" for i in range(6)]),
                       var=pd.DataFrame(index=pre.panel.ensembl))
    cells.write_h5ad(tmp_path / "cells.h5ad")
    v, ids, rep = pre.embed_file(tmp_path / "cells.h5ad", "bulk", pseudobulk_by="group")
    assert ids == ["B_cell", "T_cell"] and rep.kind == "single_cell" and v.shape == (2, VECTOR_DIM)
    with pytest.raises(InputError, match="obs column"):
        pre.embed_file(tmp_path / "cells.h5ad", pseudobulk_by="donor")
    with pytest.raises(InputError, match="could not read"):
        pre.embed_file(tmp_path / "missing.tsv")
    with pytest.raises(InputError, match="no expression"):
        pre.embed(np.zeros((1, N_GENES)), pre.panel.ensembl.tolist(), "bulk")
