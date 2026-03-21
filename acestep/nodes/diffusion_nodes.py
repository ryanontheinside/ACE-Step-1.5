"""Diffusion nodes: configuration and generation."""

from __future__ import annotations

import torch
from typing import Any, ClassVar, Optional

from acestep.engine.conditions import ConditionSet, PreparedCondition
from acestep.engine.diffusion import DiffusionConfig

from .base import BaseNode, NodeDefinition, NodePort, NodeRegistry
from .types import (
    Conditioning,
    Config,
    Curve,
    Latent,
    ModelHandle,
)


@NodeRegistry.register
class DiffusionConfigNode(BaseNode):
    """Create a diffusion loop configuration.

    Node parameters:
        steps: Number of diffusion steps (default 8 for turbo).
        method: Solver type, "ode" or "sde".
        shift: Timestep shift (default 3.0 for turbo).
        seed: Random seed.
        denoise: Denoising strength 0.0-1.0 (1.0 = full generation).
        use_cache: Enable KV caching (default False).
        noise_on_cpu: Generate noise on CPU for ComfyUI parity (default True).
    """

    node_type_id: ClassVar[str] = "acestep.DiffusionConfig"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Diffusion Config",
            category="diffusion",
            description="Configure the diffusion sampling loop.",
            inputs=(),
            outputs=(
                NodePort(name="config", type="CONFIG"),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        config = DiffusionConfig(
            infer_steps=kwargs.get("steps", 8),
            infer_method=kwargs.get("method", "ode"),
            shift=kwargs.get("shift", 3.0),
            seed=kwargs.get("seed", None),
            use_cache=kwargs.get("use_cache", False),
            noise_on_cpu=kwargs.get("noise_on_cpu", True),
            denoise=kwargs.get("denoise", 1.0),
        )
        return {"config": Config(config=config)}


@NodeRegistry.register
class Generate(BaseNode):
    """Run the diffusion generation loop.

    This is the central node. It takes a model, conditioning, config,
    and optional modulation curves, builds the ConditionSet, and runs
    DiffusionEngine.generate().

    Positive conditioning is required. Negative conditioning is optional
    (for CFG with guidance_scale > 1.0, future base model support).
    Source latent is required when denoise < 1.0.
    """

    node_type_id: ClassVar[str] = "acestep.Generate"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Generate",
            category="diffusion",
            description="Run the ACE-Step diffusion loop.",
            inputs=(
                NodePort(name="model", type="MODEL"),
                NodePort(name="config", type="CONFIG"),
                NodePort(name="positive", type="CONDITIONING"),
                NodePort(
                    name="negative",
                    type="CONDITIONING",
                    required=False,
                    description="Negative conditioning for CFG (optional, ignored at guidance_scale=1.0).",
                ),
                NodePort(
                    name="source_latent",
                    type="LATENT",
                    required=False,
                    description="Source audio latent for partial denoising (denoise < 1.0).",
                ),
                NodePort(
                    name="velocity_scale",
                    type="CURVE",
                    required=False,
                    description="Per-frame velocity scaling curve.",
                ),
                NodePort(
                    name="sde_denoise_curve",
                    type="CURVE",
                    required=False,
                    description="Per-frame SDE re-noise modulation (requires method='sde').",
                ),
                NodePort(
                    name="initial_noise_curve",
                    type="CURVE",
                    required=False,
                    description="Per-frame initial noise/source mixing curve.",
                ),
                NodePort(
                    name="x0_target",
                    type="LATENT",
                    required=False,
                    description="Target latent for x0 blending.",
                ),
                NodePort(
                    name="x0_target_curve",
                    type="CURVE",
                    required=False,
                    description="Per-frame blend strength toward x0 target.",
                ),
                NodePort(
                    name="guidance_curve",
                    type="CURVE",
                    required=False,
                    description="Per-frame CFG guidance scale (requires negative conditioning).",
                ),
                NodePort(
                    name="ode_noise_curve",
                    type="CURVE",
                    required=False,
                    description="Per-frame ODE noise injection curve (skipped on final step).",
                ),
            ),
            outputs=(
                NodePort(name="latent", type="LATENT"),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        model_handle: ModelHandle = kwargs["model"]
        config_payload: Config = kwargs["config"]
        positive: Conditioning = kwargs["positive"]

        negative: Optional[Conditioning] = kwargs.get("negative")
        source_latent: Optional[Latent] = kwargs.get("source_latent")
        velocity_scale: Optional[Curve] = kwargs.get("velocity_scale")
        sde_denoise_curve: Optional[Curve] = kwargs.get("sde_denoise_curve")
        initial_noise_curve: Optional[Curve] = kwargs.get("initial_noise_curve")
        x0_target: Optional[Latent] = kwargs.get("x0_target")
        x0_target_curve: Optional[Curve] = kwargs.get("x0_target_curve")
        guidance_curve_input: Optional[Curve] = kwargs.get("guidance_curve")
        ode_noise_curve: Optional[Curve] = kwargs.get("ode_noise_curve")

        handler = model_handle.handler
        config = config_payload.config

        # Build PreparedConditions from the Conditioning payload
        conditions = []
        for entry in positive.to_entries():
            conditions.append(
                PreparedCondition(
                    encoder_hidden_states=entry.encoder_hidden_states,
                    encoder_attention_mask=entry.encoder_attention_mask,
                    context_latents=entry.context_latents,
                    temporal_weight=entry.temporal_weight,
                    step_range=entry.step_range,
                    hook_ref=entry.hook_ref,
                )
            )

        # Build the ConditionSet
        guidance_scale = kwargs.get("guidance_scale", 1.0)
        condition_set = ConditionSet(
            conditions=conditions,
            guidance_scale=guidance_scale,
        )

        # Extract source latents and mask
        source_latents = None
        latent_mask = None
        if source_latent is not None:
            source_latents = source_latent.tensor
            latent_mask = source_latent.mask

        # Build negative condition set for CFG
        negative_condition_set = None
        if negative is not None and guidance_curve_input is not None:
            neg_conditions = []
            for entry in negative.to_entries():
                neg_conditions.append(
                    PreparedCondition(
                        encoder_hidden_states=entry.encoder_hidden_states,
                        encoder_attention_mask=entry.encoder_attention_mask,
                        context_latents=entry.context_latents,
                        temporal_weight=entry.temporal_weight,
                        step_range=entry.step_range,
                        hook_ref=entry.hook_ref,
                    )
                )
            negative_condition_set = ConditionSet(conditions=neg_conditions)

        # Build engine kwargs for optional curves (move to model device)
        device = handler.device
        dtype = handler.dtype
        engine_kwargs: dict[str, Any] = {}
        if velocity_scale is not None:
            engine_kwargs["velocity_scale"] = velocity_scale.tensor.to(device=device, dtype=dtype)
        if sde_denoise_curve is not None:
            engine_kwargs["sde_denoise_curve"] = sde_denoise_curve.tensor.to(device=device, dtype=dtype)
        if initial_noise_curve is not None:
            engine_kwargs["initial_noise_curve"] = initial_noise_curve.tensor.to(device=device, dtype=dtype)
        if x0_target is not None:
            engine_kwargs["x0_target"] = x0_target.tensor.to(device=device, dtype=dtype)
        if x0_target_curve is not None:
            engine_kwargs["x0_target_curve"] = x0_target_curve.tensor.to(device=device, dtype=dtype)
        if negative_condition_set is not None:
            engine_kwargs["negative_condition_set"] = negative_condition_set
        if guidance_curve_input is not None:
            engine_kwargs["guidance_curve"] = guidance_curve_input.tensor.to(device=device, dtype=dtype)
        if ode_noise_curve is not None:
            engine_kwargs["ode_noise_curve"] = ode_noise_curve.tensor.to(device=device, dtype=dtype)

        # When LoRA is active, bypass the compiled fast path.
        # torch.compile can produce different numerical results in bf16,
        # and LoRA-modified weights amplify these differences.
        apply_hooks_fn = None
        if getattr(handler, '_active_lora_deltas', None):
            apply_hooks_fn = lambda _: None

        # Run the engine
        result = handler.engine_generate(
            condition_set=condition_set,
            config=config,
            latent_mask=latent_mask,
            source_latents=source_latents,
            apply_hooks_fn=apply_hooks_fn,
            **engine_kwargs,
        )

        return {
            "latent": Latent(tensor=result["target_latents"]),
        }
