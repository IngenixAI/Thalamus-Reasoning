"""Locate the released files: a local copy (``THALAMUS_REASONING_HOME`` or an explicit path) or the Hugging Face Hub."""

import os
from pathlib import Path

REPO_ID = "ingenix/Thalamus-Reasoning-9B"

# Hugging Face repository holding Geneformer-V2-104M. Pinned so profile embeddings match the training inputs.
GENEFORMER_REPO = "ctheodoris/Geneformer"
GENEFORMER_REVISION = "1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5"
GENEFORMER_SUBFOLDER = "Geneformer-V2-104M"

# Checkpoint files only; the repository also holds DepMap and evaluation data.
MODEL_FILES = [
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "chat_template.jinja", "*.safetensors", "model.safetensors.index.json", "*.py",
]


def _local_or_hub(source: str | os.PathLike | None, patterns: list[str] | None) -> Path:
    """``source`` (or ``$THALAMUS_REASONING_HOME``) if it is a local directory, else the Hub snapshot of ``source``
    (a repo id) or ``REPO_ID`` restricted to ``patterns``. A path that is not a folder raises ``FileNotFoundError``."""
    if not source and os.environ.get("THALAMUS_REASONING_HOME"):
        source = os.environ["THALAMUS_REASONING_HOME"]
        if not Path(source).expanduser().is_dir():
            raise FileNotFoundError(f"THALAMUS_REASONING_HOME={source} is not a folder")
    if source:
        local = Path(source).expanduser()
        if local.is_dir():
            return local
        # A Hub repo id is "owner/name"; anything else that is not a folder is a mistyped path.
        if local.exists() or str(source).startswith((".", "/", "~")) or str(source).count("/") != 1:
            raise FileNotFoundError(f"{source} is not a folder; give a local copy of the model repository "
                                    "or a Hugging Face repo id (owner/name)")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(source or REPO_ID), allow_patterns=patterns))


def release_path(relative: str = "", source: str | os.PathLike | None = None) -> Path:
    """Path of ``relative`` inside the release, downloading it from the Hub if needed.

    ``source`` (or ``$THALAMUS_REASONING_HOME``) may point at a local copy of the repository; otherwise
    ``REPO_ID`` is fetched. Directories are downloaded with all their files.
    """
    patterns = [relative, f"{relative.rstrip('/')}/*"] if relative else None
    path = _local_or_hub(source, patterns) / relative
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def model_path(source: str | os.PathLike | None = None) -> str:
    """Checkpoint directory: a local path, ``$THALAMUS_REASONING_HOME``, or the downloaded Hub snapshot."""
    return str(_local_or_hub(source, MODEL_FILES))
