"""Conditioning nodes: text encoding, zeroing, averaging, combining."""

from __future__ import annotations

import torch
from typing import Any, ClassVar, Optional

from .base import BaseNode, NodeDefinition, NodePort, NodeRegistry
from .types import (
    Audio,
    CLIPHandle,
    Conditioning,
    ConditioningEntry,
    Latent,
    Mask,
    ModelHandle,
    SemanticHints,
)


@NodeRegistry.register
class TextEncode(BaseNode):
    """Encode text prompt and build a conditioning for the decoder.

    Directly tokenizes text/lyrics, encodes embeddings, and calls
    model.prepare_condition(). Based on the proven pattern from the
    test scripts (not the handler's batch pipeline).

    Node parameters:
        tags: Genre/style tags string.
        lyrics: Song lyrics (empty string for instrumental).
        task: "generate" or "cover".
        bpm: Beats per minute.
        duration: Duration in seconds.
        key: Musical key (e.g. "G# minor").
        time_signature: Time signature (e.g. "4").
        language: Language code (e.g. "en").
    """

    node_type_id: ClassVar[str] = "acestep.TextEncode"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="ACE-Step Text Encode",
            category="conditioning",
            description="Encode tags, lyrics, and source audio into conditioning.",
            inputs=(
                NodePort(name="clip", type="CLIP"),
                NodePort(name="model", type="MODEL"),
                NodePort(
                    name="source_latent",
                    type="LATENT",
                    required=False,
                    description="Source audio latent (required for cover tasks).",
                ),
                NodePort(
                    name="semantic_hints",
                    type="SEMANTIC_HINTS",
                    required=False,
                    description="Pre-extracted semantic hints.",
                ),
                NodePort(
                    name="refer_audio",
                    type="AUDIO",
                    required=False,
                    description="Reference audio for timbre (uses source latent if not provided).",
                ),
            ),
            outputs=(
                NodePort(name="conditioning", type="CONDITIONING"),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        clip: CLIPHandle = kwargs["clip"]
        model_handle: ModelHandle = kwargs["model"]
        handler = clip.handler
        device = handler.device
        dtype = handler.dtype

        source_latent: Optional[Latent] = kwargs.get("source_latent")
        semantic_hints: Optional[SemanticHints] = kwargs.get("semantic_hints")
        refer_audio: Optional[Audio] = kwargs.get("refer_audio")

        tags = kwargs.get("tags", "")
        lyrics = kwargs.get("lyrics", "")
        task = kwargs.get("task", "generate")
        bpm = kwargs.get("bpm", 120)
        duration = kwargs.get("duration", 60.0)
        key = kwargs.get("key", "C major")
        time_signature = kwargs.get("time_signature", "4")
        language = kwargs.get("language", "en")

        is_cover = task in ("cover", "edit", "repaint")

        # --- Build text prompt (matches test script format) ---
        instruction = "Generate audio semantic tokens based on the given conditions:"
        meta_cap = (
            f"- bpm: {bpm}\n"
            f"- timesignature: {time_signature}\n"
            f"- keyscale: {key}\n"
            f"- duration: {duration}\n"
        )
        text_prompt = (
            f"# Instruction\n{instruction}\n\n"
            f"# Caption\n{tags}\n\n"
            f"# Metas\n{meta_cap}"
            f"<|endoftext|>\n"
        )

        # --- Build lyrics prompt ---
        if lyrics:
            lyrics_prompt = f"# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|><|endoftext|>"
        else:
            lyrics_prompt = f"# Languages\n{language}\n\n# Lyric\n<|endoftext|><|endoftext|>"

        # --- Tokenize and encode ---
        with handler._load_model_context("text_encoder"):
            tokens = handler.text_tokenizer(
                text_prompt, return_tensors="pt", add_special_tokens=False
            )
            text_hidden = handler.infer_text_embeddings(
                tokens["input_ids"].to(device)
            )
            text_mask = tokens["attention_mask"].to(device).bool()

            lyric_tokens = handler.text_tokenizer(
                lyrics_prompt, return_tensors="pt", add_special_tokens=False
            )
            lyric_hidden = handler.infer_lyric_embeddings(
                lyric_tokens["input_ids"].to(device)
            )
            lyric_mask = torch.ones(
                lyric_hidden.shape[:2], device=device, dtype=torch.bool
            )

        # --- Source latents ---
        if source_latent is not None:
            src_lat = source_latent.tensor.to(device=device, dtype=dtype)
        else:
            # Generate from silence
            handler._ensure_silence_latent_on_device()
            T = int(duration * 25)  # 25 fps latent rate
            src_lat = (
                handler.silence_latent
                .unsqueeze(0)
                .expand(1, T, -1)
                .clone()
                .to(device=device, dtype=dtype)
            )

        T = src_lat.shape[1]
        D = src_lat.shape[2]

        # --- Reference audio (timbre) ---
        if refer_audio is not None:
            with handler._load_model_context("vae"):
                refer_packed, refer_order_mask = handler.encode_reference_from_audio(
                    refer_audio.waveform[0]
                    if refer_audio.waveform.dim() == 3
                    else refer_audio.waveform
                )
        else:
            # Default: use source latent as reference (matches test scripts)
            refer_packed = src_lat.clone()
            refer_order_mask = torch.zeros(1, device=device, dtype=torch.long)

        # --- Semantic hints ---
        precomputed_hints = None
        if semantic_hints is not None:
            precomputed_hints = semantic_hints.tensor.to(device=device, dtype=dtype)

        # --- Build condition via model.prepare_condition ---
        chunk_masks = torch.ones(1, T, D, device=device, dtype=dtype)
        is_covers = torch.tensor([is_cover], dtype=torch.bool, device=device)
        handler._ensure_silence_latent_on_device()

        with handler._load_model_context("model"):
            enc_hidden, enc_mask, context_latents = handler.model.prepare_condition(
                text_hidden_states=text_hidden.to(dtype),
                text_attention_mask=text_mask,
                lyric_hidden_states=lyric_hidden.to(dtype),
                lyric_attention_mask=lyric_mask,
                refer_audio_acoustic_hidden_states_packed=refer_packed.to(device),
                refer_audio_order_mask=refer_order_mask,
                hidden_states=src_lat,
                attention_mask=torch.ones(1, T, device=device, dtype=dtype),
                silence_latent=handler.silence_latent,
                src_latents=src_lat,
                chunk_masks=chunk_masks,
                is_covers=is_covers,
                precomputed_lm_hints_25Hz=precomputed_hints,
            )

        return {
            "conditioning": Conditioning(
                encoder_hidden_states=enc_hidden,
                encoder_attention_mask=enc_mask,
                context_latents=context_latents,
            )
        }


@NodeRegistry.register
class ConditioningZeroOut(BaseNode):
    """Zero out the encoder hidden states to produce an unconditional embedding.

    Used as the negative/uncond input for CFG when guidance_scale > 1.0.
    With the turbo model (guidance_scale=1.0), this is effectively ignored.
    """

    node_type_id: ClassVar[str] = "acestep.ConditioningZeroOut"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Conditioning Zero Out",
            category="conditioning",
            description="Zero out encoder hidden states for unconditional embedding.",
            inputs=(
                NodePort(name="conditioning", type="CONDITIONING"),
            ),
            outputs=(
                NodePort(name="conditioning", type="CONDITIONING"),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        cond: Conditioning = kwargs["conditioning"]
        entries = cond.to_entries()
        if not entries:
            return {"conditioning": cond}

        entry = entries[0]
        return {
            "conditioning": Conditioning(
                encoder_hidden_states=torch.zeros_like(entry.encoder_hidden_states),
                encoder_attention_mask=entry.encoder_attention_mask,
                context_latents=entry.context_latents,
            )
        }


@NodeRegistry.register
class ConditioningAverage(BaseNode):
    """Blend two conditionings by weighted average.

    Produces a single fused conditioning by interpolating the
    encoder hidden states and context latents. Attention mask
    is taken from conditioning_a.

    Node parameters:
        weight: Blend weight. 0.0 = all A, 1.0 = all B.
    """

    node_type_id: ClassVar[str] = "acestep.ConditioningAverage"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Conditioning Average",
            category="conditioning",
            description="Weighted average of two conditionings.",
            inputs=(
                NodePort(name="conditioning_a", type="CONDITIONING"),
                NodePort(name="conditioning_b", type="CONDITIONING"),
            ),
            outputs=(
                NodePort(name="conditioning", type="CONDITIONING"),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        cond_a: Conditioning = kwargs["conditioning_a"]
        cond_b: Conditioning = kwargs["conditioning_b"]
        weight = kwargs.get("weight", 0.5)

        entries_a = cond_a.to_entries()
        entries_b = cond_b.to_entries()
        if not entries_a or not entries_b:
            return {"conditioning": cond_a}

        a = entries_a[0]
        b = entries_b[0]

        w = float(weight)
        blended_enc = (1.0 - w) * a.encoder_hidden_states + w * b.encoder_hidden_states
        blended_ctx = (1.0 - w) * a.context_latents + w * b.context_latents

        return {
            "conditioning": Conditioning(
                encoder_hidden_states=blended_enc,
                encoder_attention_mask=a.encoder_attention_mask,
                context_latents=blended_ctx,
            )
        }


@NodeRegistry.register
class ConditioningCombine(BaseNode):
    """Combine two conditionings into a multi-condition set.

    Unlike ConditioningAverage (which fuses into one), this preserves
    both conditions as separate entries with optional compositing
    metadata. The Generate node will run separate decoder calls
    and blend velocities per-frame.

    Node parameters:
        step_range_start_b: Diffusion step fraction where B activates.
        step_range_end_b: Diffusion step fraction where B deactivates.
    """

    node_type_id: ClassVar[str] = "acestep.ConditioningCombine"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Conditioning Combine",
            category="conditioning",
            description="Combine two conditionings for multi-condition generation.",
            inputs=(
                NodePort(name="conditioning_a", type="CONDITIONING"),
                NodePort(name="conditioning_b", type="CONDITIONING"),
                NodePort(
                    name="temporal_weight_b",
                    type="MASK",
                    required=False,
                    description="Per-frame blend weight for condition B.",
                ),
            ),
            outputs=(
                NodePort(name="conditioning", type="CONDITIONING"),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        cond_a: Conditioning = kwargs["conditioning_a"]
        cond_b: Conditioning = kwargs["conditioning_b"]
        temporal_mask: Optional[Mask] = kwargs.get("temporal_weight_b")

        step_start = kwargs.get("step_range_start_b")
        step_end = kwargs.get("step_range_end_b")

        entries_a = cond_a.to_entries()
        entries_b = cond_b.to_entries()

        # A entries have no compositing metadata (uniform weight)
        combined = list(entries_a)

        # B entries get compositing metadata
        step_range = None
        if step_start is not None and step_end is not None:
            step_range = (float(step_start), float(step_end))

        temporal_weight = None
        if temporal_mask is not None:
            temporal_weight = temporal_mask.tensor

        for entry in entries_b:
            combined.append(
                ConditioningEntry(
                    encoder_hidden_states=entry.encoder_hidden_states,
                    encoder_attention_mask=entry.encoder_attention_mask,
                    context_latents=entry.context_latents,
                    temporal_weight=temporal_weight,
                    step_range=step_range,
                    hook_ref=entry.hook_ref,
                )
            )

        return {
            "conditioning": Conditioning(entries=combined)
        }
