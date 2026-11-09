"""Thalamus-Reasoning: a multimodal language model for transcriptomic profiles."""

from thalamus_reasoning.model import OMICS, ThalamusReasoning, Reply
from thalamus_reasoning.preprocess import load_profiles

__all__ = ["OMICS", "ThalamusReasoning", "Reply", "load_profiles"]
__version__ = "0.1.0"
