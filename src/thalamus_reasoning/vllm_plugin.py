"""Thalamus-Reasoning model for vLLM inference.

Self-contained vLLM model class for Thalamus-Reasoning (Qwen3.5 backbone + linear omics
projection). Registered with vLLM's ``ModelRegistry`` via the
``vllm.general_plugins`` entry point declared in ``pyproject.toml``, which
dispatches to ``register_thalamus_reasoning`` below.
"""

from collections.abc import Iterable, Mapping, Sequence

import torch
import torch.nn as nn
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.inputs import MultiModalDataDict
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MultiModalEmbeddings,
    SupportsMRoPE,
    SupportsMultiModal,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    init_vllm_registered_model,
    maybe_prefix,
)


class OmicsProjection(nn.Module):
    """Linear projection from omics-vector space to LLM embedding space."""

    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.to(self.linear.weight.dtype))


class ThalamusReasoningProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_data_parser(self) -> MultiModalDataParser:
        hf_config = self.get_hf_config()
        return MultiModalDataParser(expected_hidden_size=hf_config.omics_input_dim)


class ThalamusReasoningDummyInputsBuilder(BaseDummyInputsBuilder["ThalamusReasoningProcessingInfo"]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        tokenizer = self.info.get_tokenizer()
        image_token = tokenizer.decode([self.info.get_hf_config().image_token_index])
        return image_token * mm_counts.get("image", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, object] | None = None,
    ) -> MultiModalDataDict:
        n = mm_counts.get("image", 0)
        omics_dim = self.info.get_hf_config().omics_input_dim
        return {"image": [torch.zeros(1, omics_dim) for _ in range(n)]}


class ThalamusReasoningMultiModalProcessor(BaseMultiModalProcessor["ThalamusReasoningProcessingInfo"]):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        text_outputs = tokenizer(
            prompt, return_tensors="pt",
            **{k: v for k, v in tok_kwargs.items() if k != "return_tensors"},
        )
        result: dict[str, object] = dict(text_outputs)

        images = mm_data.get("image", [])
        tensors = [img.view(-1) if img.dim() > 1 else img
                   for img in images if isinstance(img, torch.Tensor)]
        if tensors:
            result["omics_vectors"] = torch.stack(tensors)
        return BatchFeature(result)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # vLLM stores precomputed-embedding passthrough data as
        # "{modality}_embeds"; declare it here so build_elems picks it up.
        return dict(
            omics_vectors=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        image_token_id = self.info.get_hf_config().image_token_index
        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=[image_token_id],
            ),
        ]


@MULTIMODAL_REGISTRY.register_processor(
    ThalamusReasoningMultiModalProcessor,
    info=ThalamusReasoningProcessingInfo,
    dummy_inputs=ThalamusReasoningDummyInputsBuilder,
)
class ThalamusReasoningQwen3_5ForConditionalGenerationVLLM(
    nn.Module, SupportsMultiModal, HasInnerState, IsHybrid, SupportsMRoPE,
):
    """vLLM-native Thalamus-Reasoning: Qwen3.5 backbone (hybrid Mamba state, MRoPE) + linear omics projection."""

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        return "<image>" if modality.startswith("image") else None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config

        with self._mark_tower_model(vllm_config, "image"):
            self.omics_projection = OmicsProjection(
                input_dim=config.omics_input_dim,
                output_dim=config.llm_hidden_size,
                bias=config.projection_bias,
            )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        omics = kwargs.pop("omics_vectors", None)
        if omics is None:
            omics = kwargs.pop("image_embeds", None)
        if omics is None:
            return []
        projected = self.omics_projection(omics)
        return projected.unsqueeze(1) if projected.dim() == 2 else projected

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)

    def get_mm_mapping(self):
        from vllm.model_executor.models.module_mapping import MultiModelKeys
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="omics_projection",
        )

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM
        return Qwen3_5ForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM
        return Qwen3_5ForCausalLM.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM
        return Qwen3_5ForCausalLM.get_mamba_state_copy_func()

    def get_mrope_input_positions(
        self, input_tokens: list[int], mm_features: list[object],
    ) -> tuple[torch.Tensor, int]:
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.unsqueeze(0).expand(3, -1), 0


def register_thalamus_reasoning():
    """vLLM plugin entry point (``vllm.general_plugins``)."""
    from vllm import ModelRegistry
    ModelRegistry.register_model(
        "ThalamusReasoningQwen3_5ForConditionalGeneration",
        "thalamus_reasoning.vllm_plugin:ThalamusReasoningQwen3_5ForConditionalGenerationVLLM",
    )
