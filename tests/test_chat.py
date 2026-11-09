"""Placeholder handling in chat messages (no model, no downloads)."""

import numpy as np
import pandas as pd

from thalamus_reasoning.chat import Conversation
from thalamus_reasoning.depmap import CellLineResolver, OMICS


class FakeLM:
    vector_dim = 4


class FakeDepMap:
    def __init__(self):
        table = pd.DataFrame({
            "CVCL_id": ["CVCL_0031", "CVCL_0023", "CVCL_0042"],
            "DepMap_id": ["ACH-000019", "ACH-000681", "ACH-000000"],
            "CellLine_name": ["MCF-7", "A549", "RD"],
            "CellLine_stripped_name": ["MCF7", "A549", "RD"],
            "CellLine_synonyms": [["MCF 7"], ["A 549"], []],
            "embedding_row": [0, 1, 2],
        })
        self.resolver = CellLineResolver(table, gene_symbols=["TP53"])
        self.embeddings = np.arange(12, dtype=np.float32).reshape(3, 4)

    def resolve(self, text):
        res = self.resolver.resolve(text)
        return res, self.embeddings[res.rows]


def conv():
    c = Conversation(FakeLM(), FakeDepMap())
    c.profiles = {"GSM1": np.full(4, 100.0), "GSM2": np.full(4, 200.0)}
    return c


def test_depmap_names_any_spelling():
    turn = conv().prepare("Compare mcf 7 with A549 and MCF-7 again")
    assert turn.text.count(OMICS) == 2
    assert [v[0] for v in turn.vectors] == [0.0, 4.0]


def test_risky_name_needs_context():
    turn = conv().prepare("RD is short. But RD cells and @RD are lines.")
    assert turn.text.count(OMICS) == 1
    assert turn.vectors[0][0] == 8.0


def test_own_profiles_and_depmap_interleaved_in_order():
    turn = conv().prepare("Is @GSM2 closer to MCF7 or to @GSM1? Also A549.")
    assert turn.text.startswith(f"Is GSM2{OMICS} closer to MCF7{OMICS} or to GSM1{OMICS}")
    assert [v[0] for v in turn.vectors] == [200.0, 0.0, 100.0, 4.0]


def test_literal_placeholder_is_escaped():
    turn = conv().prepare(f"what does {OMICS} mean?")
    assert OMICS not in turn.text and not turn.vectors


def test_unknown_reference_left_as_text():
    turn = conv().prepare("tell me about @GSM999")
    assert turn.text == "tell me about @GSM999" and not turn.vectors


def test_word_like_names_case_rules():
    c = conv()
    c.depmap.resolver = CellLineResolver(pd.DataFrame({
        "CVCL_id": ["CVCL_2092", "CVCL_0395", "CVCL_X"], "DepMap_id": ["ACH-000259", "ACH-001", "ACH-002"],
        "CellLine_name": ["Kelly", "LNCaP clone FGC", "MY"], "CellLine_stripped_name": ["KELLY", "LNCAPCLONEFGC", "MY"],
        "CellLine_synonyms": [["KELLY"], ["LNCaP"], []], "embedding_row": [0, 1, 2],
    }), gene_symbols=[])
    assert c.prepare("Kelly cells").text.count(OMICS) == 1
    assert c.prepare("@kelly please").text.count(OMICS) == 1
    assert c.prepare("kelly is a name").text.count(OMICS) == 0
    assert c.prepare("compare my cells").text.count(OMICS) == 0
    assert c.prepare("LNCaP vs PC-3").text.count(OMICS) == 1


def test_unknown_mention_is_reported():
    turn = conv().prepare("what about @HEK293?")
    assert not turn.vectors and any("HEK293" in n for n in turn.notes)


def test_own_sample_named_after_a_cell_line():
    c = conv()
    c.profiles["MCF7_ctrl"] = np.full(4, 300.0)
    turn = c.prepare("Compare @MCF7_ctrl with MCF7, then @MCF7_ctrl again.")
    assert turn.text == f"Compare MCF7_ctrl{OMICS} with MCF7{OMICS}, then MCF7_ctrl again."
    assert [v[0] for v in turn.vectors] == [300.0, 0.0]


def test_depmap_system_message_survives_session_rebuild():
    c = conv()
    c.prepare("Tell me about MCF7")
    assert c.used_depmap
    c2 = conv()
    c2.used_depmap = c.used_depmap  # the web app restores it from the session state each turn
    c2.prepare("And what about @GSM1?")
    assert c2.used_depmap
