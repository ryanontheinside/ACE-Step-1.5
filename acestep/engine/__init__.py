from .conditions import PreparedCondition, ConditionSet, ConditionBuilder
from .diffusion import DiffusionEngine, DiffusionConfig
from .masking import LatentNoiseMask
from .model_context import ModelContext
from .ops import average_conditions, blend_semantic_hints, extract_semantic_hints

__all__ = [
    "PreparedCondition",
    "ConditionSet",
    "ConditionBuilder",
    "DiffusionEngine",
    "DiffusionConfig",
    "LatentNoiseMask",
    "ModelContext",
    "average_conditions",
    "blend_semantic_hints",
    "extract_semantic_hints",
]

# TRT imports are deferred to avoid hard dependency on tensorrt
def get_trt_decoder(engine_path, **kwargs):
    """Convenience factory for TRTDecoder (lazy import)."""
    from .trt.runtime import TRTDecoder
    return TRTDecoder(engine_path, **kwargs)
