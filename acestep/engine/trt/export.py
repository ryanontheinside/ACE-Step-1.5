"""ONNX export and TensorRT engine build for the ACE-Step decoder.

Export flow:
  1. Wrap decoder in DecoderForExport (fixes Lambda, forces SDPA, no cache)
  2. Export to ONNX with dynamic B / T / L_enc axes
  3. Build TRT engine with FP16 and optimization profiles

Precision strategy:
  - Export weights in fp32 (preserves full precision in ONNX graph)
  - TRT builder converts to fp16 internally with its own kernel selection
  - This avoids the bf16-to-fp16 silent truncation that causes wrong output
    when exporting directly in half precision
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Traceable replacement for the Lambda(transpose) modules
# ------------------------------------------------------------------

class _Transpose12(nn.Module):
    """Transpose dims 1 and 2.  Drop-in for Lambda(lambda x: x.transpose(1, 2))."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2)


# ------------------------------------------------------------------
# Export wrapper
# ------------------------------------------------------------------

class DecoderForExport(nn.Module):
    """Thin wrapper that makes AceStepDiTModel safe for ONNX tracing.

    Changes vs. the raw decoder forward():
      - Lambda modules replaced with _Transpose12 for traceability
      - Attention implementation forced to SDPA (no flash_attn CUDA kernels)
      - KV cache disabled (use_cache=False, past_key_values=None)
      - output_attentions=False (no extra tuple elements)
      - timestep_r set equal to timestep (inference convention)
      - Returns the velocity tensor directly, not a tuple

    The input attention_mask and encoder_attention_mask parameters are
    intentionally set to None because the decoder's forward() shadows
    them immediately with local None assignments (lines 1378-1382 in
    the modeling file) and constructs full bidirectional masks from
    scratch via create_4d_mask().  Passing None here is therefore
    identical to passing torch.ones and avoids two unnecessary dynamic
    inputs in the TRT engine.
    """

    def __init__(self, decoder: nn.Module, mixed_precision: bool = False):
        super().__init__()
        self.decoder = decoder

        # Replace Lambda with traceable transpose
        self._replace_lambdas()

        # Force SDPA so the graph contains only standard ops
        self.decoder.config._attn_implementation = "sdpa"

        # Patch the decoder forward to be ONNX-trace-safe
        self._patch_decoder_for_trace()

        if mixed_precision:
            self._setup_mixed_precision()

    # ---- internal helpers ----

    def _replace_lambdas(self) -> None:
        for seq in (self.decoder.proj_in, self.decoder.proj_out):
            for i, mod in enumerate(seq):
                if type(mod).__name__ == "Lambda":
                    seq[i] = _Transpose12()

    def _setup_mixed_precision(self) -> None:
        """Convert bulk of model to fp16, keep precision-critical ops in fp32.

        The AdaLN pattern (scale_shift_table + temb -> scale/shift/gate)
        and RMSNorm are numerically sensitive. In pure fp16, the gate
        values get slightly wrong, and over 24 layers the error compounds
        multiplicatively (0.92^24 ~ 7x dampening). Keeping these ops in
        fp32 while running attention/MLP in fp16 gives near-full accuracy
        with most of the fp16 speedup.
        """
        decoder = self.decoder

        # Convert everything to fp16 first
        decoder.half()

        # Force fp32 for precision-critical paths:

        # 1. Timestep embedding (sinusoidal encoding + projection)
        decoder.time_embed.float()
        decoder.time_embed_r.float()

        # 2. Output AdaLN: scale_shift_table, norm_out
        decoder.scale_shift_table = nn.Parameter(
            decoder.scale_shift_table.data.float()
        )
        decoder.norm_out.float()

        # 3. Per-layer AdaLN: scale_shift_table + RMSNorm (all 3 norms)
        # Note: condition_embedder stays fp16 so encoder_hidden_states
        # match Q dtype in cross-attention (SDPA requires same dtype).
        for layer in decoder.layers:
            layer.scale_shift_table = nn.Parameter(
                layer.scale_shift_table.data.float()
            )
            layer.self_attn_norm.float()
            layer.mlp_norm.float()
            if hasattr(layer, "cross_attn_norm"):
                layer.cross_attn_norm.float()

    def _patch_decoder_for_trace(self) -> None:
        """Monkey-patch the decoder forward to be ONNX-trace-safe.

        Fixes three trace-hostile patterns in the stock forward():

          1. GQA in SDPA: transformers passes ``enable_gqa=True`` to
             ``F.scaled_dot_product_attention`` when num_key_value_groups > 1
             and attention_mask is None.  The ONNX exporter cannot convert
             this.  We monkey-patch ``use_gqa_in_sdpa`` to return False so
             the SDPA path falls back to ``repeat_kv`` (head expansion via
             ``repeat_interleave``), which is fully traceable.

          2. Shape-dependent Python branches: the original forward captures
             ``original_seq_len = shape[1]`` as a Python int (baked constant
             in ONNX) and uses ``if pad_length > 0`` (baked branch).  We
             remove padding/cropping entirely; the caller must ensure
             seq_len is a multiple of patch_size (=2, i.e. even).

          3. ``create_4d_mask()`` builds shape-dependent masks that bake
             traced dimensions.  Replaced with inline tensor ops for the
             sliding window mask (bidirectional, ``|i-j| <= window``).
             Full attention layers get ``None`` (is_causal=False on the
             module means SDPA treats None as bidirectional).
        """
        import types

        # --- Fix GQA: disable enable_gqa in SDPA for ONNX traceability ---
        # When use_gqa_in_sdpa returns False, the transformers SDPA function
        # manually expands K/V heads via repeat_kv (repeat_interleave) instead
        # of passing enable_gqa=True.  repeat_interleave traces cleanly.
        import transformers.integrations.sdpa_attention as _sdpa_mod
        _sdpa_mod.use_gqa_in_sdpa = lambda *args, **kwargs: False

        decoder = self.decoder
        sliding_window = decoder.config.sliding_window  # 128
        layer_types = decoder.config.layer_types  # list of "full_attention"/"sliding_attention"

        def _export_forward(
            self_dec,
            hidden_states,
            timestep,
            timestep_r,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            context_latents,
            use_cache=None,
            past_key_values=None,
            cache_position=None,
            position_ids=None,
            output_attentions=False,
            return_hidden_states=None,
            custom_layers_config=None,
            enable_early_exit=False,
            **flash_attn_kwargs,
        ):
            # Timestep embeddings
            temb_t, timestep_proj_t = self_dec.time_embed(timestep)
            temb_r, timestep_proj_r = self_dec.time_embed_r(timestep - timestep_r)
            temb = temb_t + temb_r
            timestep_proj = timestep_proj_t + timestep_proj_r

            # Concatenate context
            hidden_states = torch.cat([context_latents, hidden_states], dim=-1)

            # No padding or cropping.  seq_len must be a multiple of
            # patch_size (=2).  This avoids shape-dependent Python branches
            # that bake constants into the ONNX graph.

            # proj_in (patch embedding: Conv1d stride=2 halves seq_len)
            hidden_states = self_dec.proj_in(hidden_states)
            encoder_hidden_states = self_dec.condition_embedder(encoder_hidden_states)

            # Position IDs / embeddings
            seq_len_pat = hidden_states.shape[1]
            cache_position = torch.arange(seq_len_pat, device=hidden_states.device)
            position_ids = cache_position.unsqueeze(0)
            position_embeddings = self_dec.rotary_emb(hidden_states, position_ids)

            # Sliding window mask: bidirectional, |i-j| <= window.
            # Uses tensor ops (arange, abs, where) so ONNX can trace them.
            # Full attention layers get None (is_causal=False on the module
            # means SDPA treats None as fully bidirectional).
            indices = cache_position  # [seq_len_pat]
            diff = indices.unsqueeze(0) - indices.unsqueeze(1)  # [S, S]
            sw_mask = torch.where(
                torch.abs(diff) <= sliding_window,
                torch.zeros(1, device=hidden_states.device, dtype=hidden_states.dtype),
                torch.full((1,), torch.finfo(hidden_states.dtype).min, device=hidden_states.device, dtype=hidden_states.dtype),
            )
            sw_mask = sw_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]

            # Layer loop: static branching on layer_types (config, not runtime)
            for i, layer_module in enumerate(self_dec.layers):
                attn_mask = sw_mask if layer_types[i] == "sliding_attention" else None
                layer_outputs = layer_module(
                    hidden_states,
                    position_embeddings,
                    timestep_proj,
                    attn_mask,
                    position_ids,
                    None,   # past_key_values
                    False,  # output_attentions
                    False,  # use_cache
                    cache_position,
                    encoder_hidden_states,
                    None,   # encoder_attention_mask
                )
                hidden_states = layer_outputs[0]

            # Output AdaLN + proj_out (ConvTranspose1d stride=2 doubles seq_len)
            shift, scale = (self_dec.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
            hidden_states = (self_dec.norm_out(hidden_states) * (1 + scale) + shift).type_as(hidden_states)
            hidden_states = self_dec.proj_out(hidden_states)

            return (hidden_states, None)

        decoder.forward = types.MethodType(_export_forward, decoder)

    # ---- forward ----

    def forward(
        self,
        hidden_states: torch.Tensor,       # [B, T, 64]
        timestep: torch.Tensor,            # [B]
        encoder_hidden_states: torch.Tensor,  # [B, L_enc, 2048]
        context_latents: torch.Tensor,     # [B, T, 128]
    ) -> torch.Tensor:
        outputs = self.decoder(
            hidden_states=hidden_states,
            timestep=timestep,
            timestep_r=timestep,
            attention_mask=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=None,
            context_latents=context_latents,
            use_cache=False,
            past_key_values=None,
            output_attentions=False,
        )
        return outputs[0]  # velocity [B, T, 64]


# ------------------------------------------------------------------
# ONNX export
# ------------------------------------------------------------------

@dataclass
class OnnxExportConfig:
    """Configuration for ONNX export."""

    # Trace input sizes (should be "typical" values)
    batch_size: int = 1
    seq_len: int = 750       # 30s at 25 Hz, must be even
    enc_len: int = 200       # typical encoder seq len

    opset_version: int = 17
    do_constant_folding: bool = True

    # Mixed precision: export with fp16 bulk + fp32 for AdaLN/timestep/norm.
    # Use with TRTBuildConfig.strongly_typed=True for best FP16 accuracy.
    mixed_precision: bool = False

    # When True, disables ONNX constant folding to preserve PyTorch
    # parameter names as ONNX initializer names.  Required for TRT
    # REFIT so the refitter can address weights by their original names.
    # Without this, nn.Linear weights get auto-generated names like
    # "onnx__MatMul_12882" that can't be mapped back to LoRA targets.
    for_refit: bool = False


def export_decoder_onnx(
    model,
    onnx_path: Union[str, Path],
    device: str = "cuda",
    config: Optional[OnnxExportConfig] = None,
) -> Path:
    """Export the decoder to ONNX with dynamic shapes.

    Args:
        model: AceStepConditionGenerationModel (the full model, we extract .decoder).
        onnx_path: Where to write the .onnx file.
        device: Device for tracing ("cuda" or "cpu").
        config: Export configuration.  Defaults are fine for most cases.

    Returns:
        Path to the written ONNX file.
    """
    if config is None:
        config = OnnxExportConfig()

    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    decoder = model.decoder
    wrapper = DecoderForExport(decoder, mixed_precision=config.mixed_precision).eval()

    if config.mixed_precision:
        # Mixed precision: model already has fp16/fp32 regions set up.
        # Move to device without changing dtypes.
        wrapper = wrapper.to(device)
        trace_dtype = torch.float16
        logger.info("Exporting with mixed precision (fp16 bulk + fp32 critical ops)")
    else:
        # Full fp32 export
        wrapper = wrapper.float().to(device)
        trace_dtype = torch.float32

    B = config.batch_size
    T = config.seq_len
    L = config.enc_len

    example_inputs = (
        torch.randn(B, T, 64, device=device, dtype=trace_dtype),
        torch.full((B,), 0.5, device=device, dtype=torch.float32),  # timestep always fp32
        torch.randn(B, L, 2048, device=device, dtype=trace_dtype),
        torch.randn(B, T, 128, device=device, dtype=trace_dtype),
    )

    input_names = [
        "hidden_states",
        "timestep",
        "encoder_hidden_states",
        "context_latents",
    ]
    output_names = ["velocity"]

    dynamic_axes = {
        "hidden_states":          {0: "batch", 1: "seq_len"},
        "timestep":               {0: "batch"},
        "encoder_hidden_states":  {0: "batch", 1: "enc_len"},
        "context_latents":        {0: "batch", 1: "seq_len"},
        "velocity":               {0: "batch", 1: "seq_len"},
    }

    # For refit-enabled builds, disable constant folding to preserve
    # weight names as ONNX initializer names.  TRT does its own constant
    # folding internally, so this has no effect on engine quality.
    do_constant_folding = config.do_constant_folding
    if config.for_refit:
        do_constant_folding = False
        logger.info("REFIT mode: constant folding disabled to preserve weight names")

    logger.info("Tracing decoder for ONNX export (T=%d, L=%d) ...", T, L)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            example_inputs,
            str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=config.opset_version,
            do_constant_folding=do_constant_folding,
            dynamo=False,
        )

    # The ONNX file may exceed the 2GB protobuf limit since the decoder
    # is ~6GB.  This is fine:
    #   - OnnxRuntime uses its own parser (not protobuf) and handles it
    #   - TRT's OnnxParser.parse_from_file also handles large inline ONNX
    # The onnx Python library's load() cannot read >2GB files, which is
    # why the previous external_data conversion produced 0-byte files.
    # We skip it and rely on the native parsers.

    size_mb = onnx_path.stat().st_size / (1 << 20)
    logger.info("ONNX saved to %s (%.1f MB)", onnx_path, size_mb)
    return onnx_path


# ------------------------------------------------------------------
# TensorRT engine build
# ------------------------------------------------------------------

@dataclass
class TRTBuildConfig:
    """Configuration for TensorRT engine build."""

    fp16: bool = True
    bf16: bool = False          # TRT 9.0+ on Ampere/Hopper
    tf32: bool = True           # TF32 for fp32 accumulation kernels

    workspace_gb: float = 4.0

    # Dynamic shape profiles: (min, optimal, max) per axis
    batch_min: int = 1
    batch_opt: int = 1
    batch_max: int = 4

    seq_min: int = 126          # ~5s, even
    seq_opt: int = 750          # 30s
    seq_max: int = 1500         # 60s

    enc_min: int = 32
    enc_opt: int = 200
    enc_max: int = 512

    # Builder optimization level (0-5, higher = slower build, faster engine)
    builder_optimization_level: int = 3

    # When True, TRT respects the dtypes in the ONNX graph exactly.
    # Use with mixed-precision ONNX export to ensure fp32 regions
    # (timestep embedding, AdaLN, norms) stay in fp32 while
    # attention/MLP run in fp16.
    strongly_typed: bool = False

    # Enable weight refitting.  Allows updating engine weights at runtime
    # via trt.Refitter without rebuilding.  Required for dynamic LoRA.
    # Slight engine size increase; negligible performance impact.
    refit: bool = False

    def engine_filename(self) -> str:
        """Generate a standardized engine filename from build config.

        Format: decoder_{precision}_b{batch_max}_s{seq_max}.engine
        """
        if self.strongly_typed:
            prec = "mixed"
        elif self.bf16:
            prec = "bf16"
        elif self.fp16:
            prec = "fp16"
        else:
            prec = "fp32"
        refit_tag = "_refit" if self.refit else ""
        return f"decoder_{prec}{refit_tag}_b{self.batch_max}_s{self.seq_max}.engine"


def build_trt_engine(
    onnx_path: Union[str, Path],
    engine_path: Union[str, Path],
    config: Optional[TRTBuildConfig] = None,
) -> Path:
    """Parse ONNX and build a TensorRT engine with dynamic shapes.

    Args:
        onnx_path: Path to the ONNX model.
        engine_path: Where to write the serialized TRT engine.
        config: Build configuration.

    Returns:
        Path to the written engine file.
    """
    import tensorrt as trt

    if config is None:
        config = TRTBuildConfig()

    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)

    # Network creation flags
    net_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if config.strongly_typed and hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
        net_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        logger.info("Using STRONGLY_TYPED network (precision from ONNX graph)")

    network = builder.create_network(net_flags)
    parser = trt.OnnxParser(network, trt_logger)

    logger.info("Parsing ONNX from %s ...", onnx_path)
    # Use parse_from_file so TRT resolves external data relative to the ONNX path
    onnx_abs = str(onnx_path.resolve())
    if not parser.parse_from_file(onnx_abs):
        for i in range(parser.num_errors):
            logger.error("ONNX parse error: %s", parser.get_error(i))
        raise RuntimeError("ONNX parsing failed")

    logger.info(
        "Network: %d inputs, %d outputs, %d layers",
        network.num_inputs, network.num_outputs, network.num_layers,
    )

    # Builder config
    build_config = builder.create_builder_config()
    build_config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(config.workspace_gb * (1 << 30)),
    )

    if not config.strongly_typed:
        # Standard mode: set precision flags for TRT to use
        if config.fp16:
            build_config.set_flag(trt.BuilderFlag.FP16)
    # STRONGLY_TYPED mode: precision is baked into the ONNX graph types,
    # so we don't set FP16 flag (TRT would ignore it anyway)

    if config.refit:
        build_config.set_flag(trt.BuilderFlag.REFIT)
        logger.info("REFIT enabled: engine weights can be updated at runtime")

    if config.bf16 and hasattr(trt.BuilderFlag, "BF16"):
        build_config.set_flag(trt.BuilderFlag.BF16)
    if config.tf32:
        build_config.set_flag(trt.BuilderFlag.TF32)

    if hasattr(build_config, "builder_optimization_level"):
        build_config.builder_optimization_level = config.builder_optimization_level

    # Optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()

    Bmin, Bopt, Bmax = config.batch_min, config.batch_opt, config.batch_max
    Smin, Sopt, Smax = config.seq_min, config.seq_opt, config.seq_max
    Emin, Eopt, Emax = config.enc_min, config.enc_opt, config.enc_max

    profile.set_shape(
        "hidden_states",
        min=(Bmin, Smin, 64), opt=(Bopt, Sopt, 64), max=(Bmax, Smax, 64),
    )
    profile.set_shape(
        "timestep",
        min=(Bmin,), opt=(Bopt,), max=(Bmax,),
    )
    profile.set_shape(
        "encoder_hidden_states",
        min=(Bmin, Emin, 2048), opt=(Bopt, Eopt, 2048), max=(Bmax, Emax, 2048),
    )
    profile.set_shape(
        "context_latents",
        min=(Bmin, Smin, 128), opt=(Bopt, Sopt, 128), max=(Bmax, Smax, 128),
    )

    build_config.add_optimization_profile(profile)

    logger.info(
        "Building TRT engine (fp16=%s, bf16=%s, opt_level=%d) ...",
        config.fp16, config.bf16, config.builder_optimization_level,
    )
    logger.info(
        "  Profiles: B=[%d,%d,%d]  T=[%d,%d,%d]  L_enc=[%d,%d,%d]",
        Bmin, Bopt, Bmax, Smin, Sopt, Smax, Emin, Eopt, Emax,
    )

    serialized = builder.build_serialized_network(network, build_config)
    if serialized is None:
        raise RuntimeError("TRT engine build failed")

    with open(engine_path, "wb") as f:
        f.write(serialized)

    size_mb = engine_path.stat().st_size / (1 << 20)
    logger.info("Engine saved to %s (%.1f MB)", engine_path, size_mb)
    return engine_path


# ------------------------------------------------------------------
# Validation helper
# ------------------------------------------------------------------

@torch.no_grad()
def validate_trt_vs_pytorch(
    model,
    engine_path: Union[str, Path],
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    seq_len: int = 750,
    enc_len: int = 200,
    seed: int = 42,
) -> dict:
    """Compare TRT decoder output against PyTorch decoder output.

    Returns a dict with per-element statistics so you can gauge accuracy.
    """
    from .runtime import TRTDecoder

    torch.manual_seed(seed)
    B = 1

    hidden_states = torch.randn(B, seq_len, 64, device=device, dtype=dtype)
    timestep = torch.tensor([0.75], device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(B, enc_len, 2048, device=device, dtype=dtype)
    context_latents = torch.randn(B, seq_len, 128, device=device, dtype=dtype)

    # PyTorch reference
    model.decoder.eval()
    with torch.no_grad():
        pt_out = model.decoder(
            hidden_states=hidden_states,
            timestep=timestep,
            timestep_r=timestep,
            attention_mask=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=None,
            context_latents=context_latents,
            use_cache=False,
        )[0]

    # TRT
    trt_decoder = TRTDecoder(engine_path)
    trt_out = trt_decoder(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        context_latents=context_latents,
    )

    # Compare
    diff = (pt_out.float() - trt_out.float()).abs()
    rel_diff = diff / (pt_out.float().abs() + 1e-8)

    results = {
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "max_rel_diff": rel_diff.max().item(),
        "mean_rel_diff": rel_diff.mean().item(),
        "pt_mean": pt_out.float().mean().item(),
        "trt_mean": trt_out.float().mean().item(),
        "pt_std": pt_out.float().std().item(),
        "trt_std": trt_out.float().std().item(),
        "cosine_sim": torch.nn.functional.cosine_similarity(
            pt_out.float().flatten().unsqueeze(0),
            trt_out.float().flatten().unsqueeze(0),
        ).item(),
    }

    logger.info("Validation results:")
    for k, v in results.items():
        logger.info("  %-20s: %.6f", k, v)

    return results
