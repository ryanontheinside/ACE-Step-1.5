"""TensorRT runtime for the ACE-Step decoder.

Provides TRTDecoder, a drop-in replacement for model.decoder that runs
inference through a pre-built TensorRT engine.  Designed to slot directly
into DiffusionEngine._decoder_call().

Buffer management:
  - Input tensors are passed by pointer (zero-copy from PyTorch CUDA tensors)
  - Output tensor is pre-allocated at the max profile size and sliced per call
  - All execution uses the current PyTorch CUDA stream to avoid sync overhead
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)

# TRT tensor dtype -> torch dtype mapping
_TRT_TO_TORCH = None

def _get_trt_to_torch_map():
    global _TRT_TO_TORCH
    if _TRT_TO_TORCH is None:
        import tensorrt as trt
        _TRT_TO_TORCH = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32: torch.int32,
            trt.int8: torch.int8,
            trt.bool: torch.bool,
        }
        if hasattr(trt, "bfloat16"):
            _TRT_TO_TORCH[trt.bfloat16] = torch.bfloat16
    return _TRT_TO_TORCH


class TRTDecoder:
    """TensorRT decoder engine with the same call signature as the PyTorch decoder.

    Usage::

        trt_dec = TRTDecoder("decoder_fp16.engine")
        velocity = trt_dec(
            hidden_states=xt,          # [B, T, 64]
            timestep=t_tensor,         # [B]
            encoder_hidden_states=enc, # [B, L_enc, 2048]
            context_latents=ctx,       # [B, T, 128]
        )
    """

    INPUT_NAMES = ("hidden_states", "timestep", "encoder_hidden_states", "context_latents")
    OUTPUT_NAME = "velocity"

    def __init__(
        self,
        engine_path: Union[str, Path],
        device: Union[str, torch.device] = "cuda",
    ):
        import tensorrt as trt
        from polygraphy.backend.common import bytes_from_path
        from polygraphy.backend.trt import engine_from_bytes

        self._trt = trt
        engine_path = Path(engine_path)
        if not engine_path.exists():
            raise FileNotFoundError(f"TRT engine not found: {engine_path}")

        self.device = torch.device(device)

        logger.info("Loading TRT engine from %s ...", engine_path)
        self.engine = engine_from_bytes(bytes_from_path(str(engine_path)))

        self.context = self.engine.create_execution_context()

        # Dedicated non-default stream for TRT execution.
        # TRT on the default stream triggers extra cudaStreamSynchronize
        # calls internally and degrades performance.
        self._stream = torch.cuda.Stream(device=self.device)

        # Determine output dtype from engine
        dtype_map = _get_trt_to_torch_map()
        out_trt_dtype = self.engine.get_tensor_dtype(self.OUTPUT_NAME)
        self._output_dtype = dtype_map.get(out_trt_dtype, torch.float32)

        # Cached output buffer (allocated on first call or shape change)
        self._output_buffer: Optional[torch.Tensor] = None

        logger.info("TRT decoder ready (output_dtype=%s)", self._output_dtype)

    def _ensure_contiguous(self, t: torch.Tensor, name: str) -> torch.Tensor:
        """Ensure tensor is contiguous and on the right device.

        TRT needs contiguous memory; the dtype is left as-is because the
        engine was built with fp32 inputs and TRT handles internal conversion.
        """
        if not t.is_contiguous():
            t = t.contiguous()
        if t.device != self.device:
            t = t.to(self.device)
        # Cast to fp32 for the TRT engine (exported in fp32)
        if t.dtype != torch.float32:
            t = t.float()
        return t

    def __call__(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        context_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Run one decoder step through TensorRT.

        All inputs should be CUDA tensors.  Any dtype is accepted; they are
        cast to fp32 to match the ONNX export.  The output is returned in
        the engine's native output dtype (typically fp16 when built with FP16).

        Returns:
            velocity: [B, T, 64] tensor.
        """
        trt = self._trt
        ctx = self.context

        # Prepare inputs
        hs = self._ensure_contiguous(hidden_states, "hidden_states")
        ts = self._ensure_contiguous(timestep, "timestep")
        enc = self._ensure_contiguous(encoder_hidden_states, "encoder_hidden_states")
        cl = self._ensure_contiguous(context_latents, "context_latents")

        # The ONNX graph requires seq_len to be even (patch_size=2).
        # Pad by one frame if odd; crop back after execution.
        orig_T = hs.shape[1]
        if orig_T % 2 == 1:
            hs = torch.nn.functional.pad(hs, (0, 0, 0, 1))
            cl = torch.nn.functional.pad(cl, (0, 0, 0, 1))

        inputs = {
            "hidden_states": hs,
            "timestep": ts,
            "encoder_hidden_states": enc,
            "context_latents": cl,
        }

        # Set input shapes and addresses
        for name, tensor in inputs.items():
            ctx.set_input_shape(name, tuple(tensor.shape))
            ctx.set_tensor_address(name, tensor.data_ptr())

        # Allocate output based on inferred shape
        out_shape = tuple(ctx.get_tensor_shape(self.OUTPUT_NAME))
        if self._output_buffer is None or self._output_buffer.shape != out_shape:
            self._output_buffer = torch.empty(
                out_shape, dtype=self._output_dtype, device=self.device,
            )
        output = self._output_buffer
        ctx.set_tensor_address(self.OUTPUT_NAME, output.data_ptr())

        # Execute on dedicated stream, synced with PyTorch's current stream
        stream = self._stream
        stream.wait_stream(torch.cuda.current_stream(self.device))
        if not ctx.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TRT execute_async_v3 failed")
        torch.cuda.current_stream(self.device).wait_stream(stream)

        output = output.clone()
        if orig_T % 2 == 1:
            output = output[:, :orig_T, :]
        return output

    def benchmark(
        self,
        seq_len: int = 750,
        enc_len: int = 200,
        batch_size: int = 1,
        warmup: int = 5,
        iterations: int = 20,
    ) -> dict:
        """Benchmark TRT decoder throughput.

        Returns dict with mean/min/max step time in ms and steps/sec.
        """
        import time

        B, T, L = batch_size, seq_len, enc_len

        hs = torch.randn(B, T, 64, device=self.device, dtype=torch.float32)
        ts = torch.full((B,), 0.5, device=self.device, dtype=torch.float32)
        enc = torch.randn(B, L, 2048, device=self.device, dtype=torch.float32)
        ctx = torch.randn(B, T, 128, device=self.device, dtype=torch.float32)

        # Warmup
        for _ in range(warmup):
            self(hs, ts, enc, ctx)

        torch.cuda.synchronize()
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self(hs, ts, enc, ctx)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

        results = {
            "mean_ms": sum(times) / len(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "steps_per_sec": 1000.0 / (sum(times) / len(times)),
            "seq_len": T,
            "enc_len": L,
            "batch_size": B,
        }
        logger.info("TRT benchmark (T=%d, L=%d, B=%d):", T, L, B)
        logger.info(
            "  mean=%.1fms  min=%.1fms  max=%.1fms  (%.1f steps/sec)",
            results["mean_ms"], results["min_ms"], results["max_ms"],
            results["steps_per_sec"],
        )
        return results
