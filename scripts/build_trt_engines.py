#!/usr/bin/env python3
"""Build VAE TensorRT engines from scratch.

This script is the single reproducible entry point for TRT engine creation.
It loads the ACE-Step model, exports VAE ONNX files, and builds TRT engines.
The DiT decoder uses torch.compile (faster than TRT for this model).

Usage:
    python scripts/build_trt_engines.py

    # Custom output directory:
    python scripts/build_trt_engines.py --output-dir trt_engines

    # Skip ONNX export (reuse existing):
    python scripts/build_trt_engines.py --skip-onnx

    # Custom max duration (default 4 minutes):
    python scripts/build_trt_engines.py --max-duration 600

Requirements:
    - tensorrt-cu12 (uv pip install tensorrt-cu12)
    - ACE-Step model checkpoint at checkpoints/acestep-v15-turbo
"""

import argparse
import logging
import os
import sys
import time

# Suppress flash_attn import (not needed for export)
import importlib, importlib.util
_orig = importlib.util.find_spec
def _patch(name, *a, **k):
    if "flash_attn" in str(name):
        return None
    return _orig(name, *a, **k)
importlib.util.find_spec = _patch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build ACE-Step TRT engines")
    parser.add_argument("--output-dir", default=os.path.join(project_root, "trt_engines"),
                        help="Directory for ONNX and engine files")
    parser.add_argument("--checkpoint", default="acestep-v15-turbo",
                        help="Model checkpoint directory name")
    parser.add_argument("--skip-onnx", action="store_true",
                        help="Skip ONNX export, reuse existing files")
    parser.add_argument("--max-duration", type=int, default=240,
                        help="Maximum audio duration in seconds (default: 240 = 4min)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workspace-gb", type=float, default=8.0,
                        help="TRT builder workspace in GB")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    max_latent_frames = int(args.max_duration * 25)  # 25 Hz
    max_audio_samples = int(args.max_duration * 48000)

    onnx_dir = args.output_dir
    vae_enc_onnx = os.path.join(onnx_dir, "vae_encode.onnx")
    vae_dec_onnx = os.path.join(onnx_dir, "vae_decode.onnx")

    vae_enc_engine = os.path.join(args.output_dir, "vae_encode_fp16.engine")
    vae_dec_engine = os.path.join(args.output_dir, "vae_decode_fp16.engine")

    # ================================================================
    # Step 1: Load model
    # ================================================================
    logger.info("Loading model from checkpoints/%s...", args.checkpoint)
    from acestep.handler import AceStepHandler

    handler = AceStepHandler()
    handler.initialize_service(
        project_root=project_root,
        config_path=args.checkpoint,
        device=args.device,
        use_flash_attention=False,  # SDPA for export
        compile_model=False,
    )
    logger.info("Model loaded.")

    # ================================================================
    # Step 2: Export ONNX
    # ================================================================
    if not args.skip_onnx:
        logger.info("=" * 60)
        logger.info("ONNX EXPORT")
        logger.info("=" * 60)

        # VAE encoder
        logger.info("Exporting VAE encoder...")
        from acestep.engine.trt.vae_export import (
            export_vae_encoder_onnx,
            export_vae_decoder_onnx,
            VAEExportConfig,
        )

        with handler._load_model_context("vae"):
            t0 = time.time()
            export_vae_encoder_onnx(
                handler.vae, vae_enc_onnx, device=args.device,
                config=VAEExportConfig(trace_audio_samples=48000 * 30),
            )
            logger.info("VAE encoder exported in %.1fs", time.time() - t0)

            # VAE decoder
            logger.info("Exporting VAE decoder...")
            t0 = time.time()
            export_vae_decoder_onnx(
                handler.vae, vae_dec_onnx, device=args.device,
                config=VAEExportConfig(trace_latent_frames=750),
            )
            logger.info("VAE decoder exported in %.1fs", time.time() - t0)

        logger.info("All ONNX exports complete.")
    else:
        logger.info("Skipping ONNX export (--skip-onnx)")
        for f in [vae_enc_onnx, vae_dec_onnx]:
            if not os.path.exists(f):
                logger.error("Missing ONNX file: %s", f)
                sys.exit(1)

    # Free model memory before TRT build
    del handler
    torch.cuda.empty_cache()

    # ================================================================
    # Step 3: Build TRT engines
    # ================================================================
    logger.info("=" * 60)
    logger.info("TRT ENGINE BUILD (max_duration=%ds, max_frames=%d)",
                args.max_duration, max_latent_frames)
    logger.info("=" * 60)

    from acestep.engine.trt.vae_export import (
        build_vae_decode_engine,
        build_vae_encode_engine,
        VAETRTBuildConfig,
    )

    vae_config = VAETRTBuildConfig(
        workspace_gb=args.workspace_gb,
        decode_max_frames=max_latent_frames,
        encode_max_samples=max_audio_samples,
    )

    # VAE decoder
    logger.info("Building VAE decode engine...")
    t0 = time.time()
    build_vae_decode_engine(vae_dec_onnx, vae_dec_engine, config=vae_config)
    logger.info("VAE decode engine built in %.0fs", time.time() - t0)

    # VAE encoder
    logger.info("Building VAE encode engine...")
    t0 = time.time()
    build_vae_encode_engine(vae_enc_onnx, vae_enc_engine, config=vae_config)
    logger.info("VAE encode engine built in %.0fs", time.time() - t0)

    # ================================================================
    # Step 4: Verify
    # ================================================================
    logger.info("=" * 60)
    logger.info("VERIFICATION")
    logger.info("=" * 60)

    import tensorrt as trt

    rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    for name, path in [
        ("VAE encode", vae_enc_engine),
        ("VAE decode", vae_dec_engine),
    ]:
        with open(path, "rb") as f:
            engine = rt.deserialize_cuda_engine(f.read())
        if engine is None:
            logger.error("FAILED to load %s: %s", name, path)
            continue

        io_info = []
        for i in range(engine.num_io_tensors):
            tname = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(tname)
            shape = engine.get_tensor_shape(tname)
            label = "IN" if mode == trt.TensorIOMode.INPUT else "OUT"
            io_info.append(f"{label}: {tname} {shape}")

        profiles = []
        for i in range(engine.num_io_tensors):
            tname = engine.get_tensor_name(i)
            if engine.get_tensor_mode(tname) == trt.TensorIOMode.INPUT:
                shapes = engine.get_tensor_profile_shape(tname, 0)
                profiles.append(f"{tname}: min={shapes[0]} opt={shapes[1]} max={shapes[2]}")

        size_mb = os.path.getsize(path) / 1e6
        logger.info("  %s: OK (%.1f MB)", name, size_mb)
        for s in io_info:
            logger.info("    %s", s)
        for s in profiles:
            logger.info("    Profile: %s", s)

    logger.info("=" * 60)
    logger.info("All engines built and verified.")
    logger.info("Output directory: %s", args.output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
