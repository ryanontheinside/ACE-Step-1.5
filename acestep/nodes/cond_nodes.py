"""Conditioning nodes: text encoding, zeroing, averaging, combining."""

from __future__ import annotations

import torch
from typing import Any, ClassVar, Optional

from .base import BaseNode, NodeDefinition, NodePort, NodeRegistry
from .types import (
    CLIPHandle,
    Conditioning,
    ConditioningEntry,
    Latent,
    Mask,
    ModelHandle,
)
from ..constants import TASK_INSTRUCTIONS


@NodeRegistry.register
class TextEncode(BaseNode):
    """Encode text prompt into cross-attention conditioning.

    Tokenizes text/lyrics, encodes timbre from a reference latent, and
    packs everything via model.encoder() into encoder_hidden_states.

    Does NOT build context_latents; that is handled by Generate from
    explicit source_latent + chunk_mask inputs.

    Node parameters:
        tags: Genre/style tags string.
        lyrics: Song lyrics (empty string for instrumental).
        instruction: Instruction text for the model. Standard options:
            - "Fill the audio semantic mask based on the given conditions:" (text2music)
            - "Generate audio semantic tokens based on the given conditions:" (cover)
            - "Repaint the mask area based on the given conditions:" (repaint)
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
            description="Encode tags, lyrics, and timbre into cross-attention conditioning.",
            inputs=(
                NodePort(name="clip", type="CLIP"),
                NodePort(name="model", type="MODEL"),
                NodePort(
                    name="refer_latent",
                    type="LATENT",
                    required=False,
                    description="Timbre reference latent. Defaults to silence.",
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

        refer_latent: Optional[Latent] = kwargs.get("refer_latent")

        tags = kwargs.get("tags", "")
        lyrics = kwargs.get("lyrics", "")
        instruction = kwargs.get(
            "instruction", TASK_INSTRUCTIONS["text2music"]
        )
        bpm = kwargs.get("bpm", 120)
        duration = kwargs.get("duration", 60.0)
        key = kwargs.get("key", "C major")
        time_signature = kwargs.get("time_signature", "4")
        language = kwargs.get("language", "en")

        # --- Build text prompt ---
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

        # --- Timbre reference ---
        if refer_latent is not None:
            refer_packed = refer_latent.tensor.to(device=device, dtype=dtype)
            refer_order_mask = torch.zeros(1, device=device, dtype=torch.long)
        else:
            handler._ensure_silence_latent_on_device()
            refer_packed = handler.silence_latent[:, :750, :].to(
                device=device, dtype=dtype
            )
            refer_order_mask = torch.zeros(1, device=device, dtype=torch.long)

        # --- Encode via model.encoder ---
        with handler._load_model_context("model"):
            enc_hidden, enc_mask = handler.model.encoder(
                text_hidden_states=text_hidden.to(dtype),
                text_attention_mask=text_mask,
                lyric_hidden_states=lyric_hidden.to(dtype),
                lyric_attention_mask=lyric_mask,
                refer_audio_acoustic_hidden_states_packed=refer_packed,
                refer_audio_order_mask=refer_order_mask,
            )

        return {
            "conditioning": Conditioning(
                encoder_hidden_states=enc_hidden,
                encoder_attention_mask=enc_mask,
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
            )
        }


@NodeRegistry.register
class ConditioningAverage(BaseNode):
    """Blend two conditionings by weighted average.

    Interpolates the encoder hidden states. Attention mask
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

        # Match encoder_hidden_states lengths
        enc_a = a.encoder_hidden_states
        enc_b = b.encoder_hidden_states
        len_a = enc_a.shape[1]
        len_b = enc_b.shape[1]
        if len_b > len_a:
            enc_b = enc_b[:, :len_a]
        elif len_b < len_a:
            enc_b = torch.nn.functional.pad(enc_b, (0, 0, 0, len_a - len_b))

        blended_enc = (1.0 - w) * enc_a + w * enc_b

        return {
            "conditioning": Conditioning(
                encoder_hidden_states=blended_enc,
                encoder_attention_mask=a.encoder_attention_mask,
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

        # B entries get compositing metadata
        step_range = None
        if step_start is not None and step_end is not None:
            step_range = (float(step_start), float(step_end))

        temporal_weight_b = None
        temporal_weight_a = None
        if temporal_mask is not None:
            ref = entries_b[0].encoder_hidden_states if entries_b else (
                entries_a[0].encoder_hidden_states if entries_a else None
            )
            if ref is not None:
                temporal_weight_b = temporal_mask.tensor.to(device=ref.device, dtype=ref.dtype)
            else:
                temporal_weight_b = temporal_mask.tensor
            temporal_weight_a = 1.0 - temporal_weight_b

        combined = []
        for entry in entries_a:
            combined.append(
                ConditioningEntry(
                    encoder_hidden_states=entry.encoder_hidden_states,
                    encoder_attention_mask=entry.encoder_attention_mask,
                    temporal_weight=temporal_weight_a,
                    step_range=entry.step_range,
                    hook_ref=entry.hook_ref,
                )
            )

        for entry in entries_b:
            combined.append(
                ConditioningEntry(
                    encoder_hidden_states=entry.encoder_hidden_states,
                    encoder_attention_mask=entry.encoder_attention_mask,
                    temporal_weight=temporal_weight_b,
                    step_range=step_range,
                    hook_ref=entry.hook_ref,
                )
            )

        return {
            "conditioning": Conditioning(entries=combined)
        }
