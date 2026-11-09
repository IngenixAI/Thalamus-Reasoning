"""Gradio web chat for Thalamus-Reasoning (local or as a Hugging Face Space).

    tr-app                            # http://127.0.0.1:7860
    tr-app --share --thinking

Mention DepMap cell lines by name, or upload your own expression table / .h5ad, which is embedded on the
fly and referenced as ``@sample_id``.
"""

import argparse
import logging
import tempfile
from pathlib import Path

import numpy as np

from thalamus_reasoning.chat import Conversation
from thalamus_reasoning.depmap import DepMap
from thalamus_reasoning.model import ThalamusReasoning
from thalamus_reasoning.preprocess import InputError, Preprocessor

log = logging.getLogger(__name__)

def _empty() -> tuple:
    return [], [], {}, False  # session: messages, vectors, own profiles, DepMap used
MAX_UPLOAD = "200mb"
_DOWNLOADS = Path(tempfile.gettempdir()) / "tr_vectors"


def _download_path() -> Path:
    """A fresh file for the vectors download; files older than an hour are deleted."""
    import time
    import uuid

    _DOWNLOADS.mkdir(exist_ok=True)
    for old in _DOWNLOADS.glob("*/tr_vectors.npz"):
        if time.time() - old.stat().st_mtime > 3600:
            old.unlink(missing_ok=True)
            old.parent.rmdir()
    folder = _DOWNLOADS / uuid.uuid4().hex
    folder.mkdir()
    return folder / "tr_vectors.npz"

EXAMPLES = [
    "What tissue and cancer type is MCF7 most likely derived from, based on its expression profile?",
    "Compare the expression profiles of A549 and HCT 116. Which pathways differ most?",
    "How does estrogen receptor signalling differ between T-47D and MDA-MB-231?",
    "Which of these lines looks more mesenchymal: MDA-MB-231 or MCF7?",
]

try:  # ZeroGPU on Hugging Face Spaces
    import spaces

    gpu = spaces.GPU(duration=120)
except ImportError:  # pragma: no cover - local runs
    def gpu(fn):
        return fn


class State:
    def __init__(self, source: str | None) -> None:
        self.lm = ThalamusReasoning.load(source)
        self.depmap = DepMap(source)
        self.pre = Preprocessor(source)

    def conversation(self, session) -> Conversation:
        """Per-browser-session state: (messages, vectors, profiles, used_depmap). Uploads are never shared."""
        messages, vectors, profiles, used_depmap = session or _empty()
        conv = Conversation(self.lm, self.depmap, profiles)
        conv.messages, conv.vectors, conv.used_depmap = list(messages), list(vectors), used_depmap
        return conv


def build(state: State, thinking_default: bool = False):
    import gradio as gr

    @gpu
    def respond(message, chat_history, convo, thinking, max_tokens, temperature):
        conv = state.conversation(convo)
        turn, reply = conv.send(message, thinking=thinking, max_new_tokens=int(max_tokens),
                                temperature=float(temperature), repetition_penalty=1.05)
        shown = reply.answer
        if reply.reasoning:
            shown = f"<details><summary>Reasoning</summary>\n\n{reply.reasoning}\n\n</details>\n\n{shown}"
        if turn.notes:
            shown = "\n".join(f"*{n}*" for n in turn.notes) + "\n\n" + shown
        chat_history = chat_history + [{"role": "user", "content": message},
                                       {"role": "assistant", "content": shown}]
        return "", chat_history, (conv.messages, conv.vectors, conv.profiles, conv.used_depmap)

    @gpu
    def upload(file, kind, values, group_col, session):
        if file is None:
            return "Upload a .csv, .tsv or .h5ad file.", gr.update(), session
        try:
            vectors, samples, report = state.pre.embed_file(file, kind, values, pseudobulk_by=group_col or None,
                                                            max_profiles=200)
        except InputError as exc:
            return f"⚠️ {exc}", gr.update(), session
        renamed = report.renamed_samples
        messages, history_vectors, profiles, used_depmap = session or _empty()
        profiles = {**profiles, **dict(zip(samples, vectors))}
        session = (messages, history_vectors, profiles, used_depmap)
        out = _download_path()
        np.savez(out, embeddings=vectors, sample_ids=np.array(samples))
        warn = "".join(f"\n- ⚠️ {w}" for w in report.warnings())
        if renamed:
            warn += (f"\n- {len(renamed)} sample IDs had spaces or symbols and were renamed, e.g. "
                     f"`{next(iter(renamed))}` → `@{next(iter(renamed.values()))}`")
        ids = ", ".join(f"`@{s}`" for s in samples[:20]) + (" …" if len(samples) > 20 else "")
        read_as = report.kind.replace("_", "-") + (", values converted to TPM" if report.values == "counts" else "")
        return (f"Embedded **{len(samples)}** profiles ({report.panel_genes_found:,} / 19,260 panel genes, "
                f"{report.id_type} IDs, read as {read_as}).{warn}\n\nReference them as {ids}"), gr.update(value=str(out), visible=True), session

    def search(text):
        return state.depmap.available(text).head(300)

    # Gradio's upload cache is cleared hourly; files older than an hour are removed.
    with gr.Blocks(title="Thalamus-Reasoning", delete_cache=(3600, 3600)) as demo:
        gr.Markdown("# Thalamus-Reasoning\nA conversational language model for bulk and single-cell expression "
                    "profiles. Mention a DepMap cell line by name, or upload your own samples and refer to them as "
                    "`@sample_id` (human RNA-seq only: remove mouse or other non-human columns before uploading). "
                    "Do not upload identifiable patient data or controlled-access data. "
                    "Research use only — not for clinical decisions.")
        convo = gr.State(_empty())
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=560)
                box = gr.Textbox(placeholder="Ask about MCF7, A549, … or @your_sample", show_label=False)
                gr.Examples(EXAMPLES, inputs=box)
                with gr.Row():
                    thinking = gr.Checkbox(value=thinking_default, label="Reasoning (thinking)")
                    max_tokens = gr.Slider(64, 4096, value=1024, step=64, label="Max new tokens")
                    temperature = gr.Slider(0, 1.0, value=0.0, step=0.05, label="Temperature")
                    clear = gr.Button("Reset")
            with gr.Column(scale=2):
                with gr.Tab("Your profiles"):
                    file = gr.File(label="Expression table (.csv/.tsv, samples × genes or genes × samples) or .h5ad")
                    kind = gr.Radio(["auto", "bulk", "single_cell"], value="auto", label="Data type",
                                    info="auto: detected from the share of genes detected")
                    values = gr.Radio(["auto", "tpm", "counts"], value="auto", label="Bulk values",
                                      info="auto: read counts / CPM are detected and converted to TPM")
                    group = gr.Textbox(label="Pseudobulk by obs column (h5ad, optional)")
                    go = gr.Button("Embed")
                    status = gr.Markdown()
                    download = gr.File(label="Vectors (.npz)", visible=False)
                with gr.Tab("DepMap cell lines"):
                    query = gr.Textbox(label="Search name, tissue or disease")
                    table = gr.Dataframe(search(""), interactive=False, wrap=True)
        box.submit(respond, [box, chatbot, convo, thinking, max_tokens, temperature], [box, chatbot, convo])
        clear.click(lambda s: ([], ([], [], (s or _empty())[2], False)), convo, [chatbot, convo])
        go.click(upload, [file, kind, values, group, convo], [status, download, convo])
        query.change(search, query, table)
    return demo


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="local copy of the model repository (default: $THALAMUS_REASONING_HOME or the Hub)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true",
                    help="also create a public gradio.live link; anyone with the link can use the app on your GPU")
    ap.add_argument("--thinking", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if args.share:
        log.warning("--share: anyone with the public link can use the app on your GPU and upload data to this "
                    "machine. Share the link only with people you trust.")
    build(State(args.model), args.thinking).launch(server_name=args.host, server_port=args.port, share=args.share,
                                                   max_file_size=MAX_UPLOAD)


if __name__ == "__main__":
    main()
