"""Curve generation nodes for per-frame modulation."""

from __future__ import annotations

import math
import torch
from typing import Any, ClassVar

from .base import BaseNode, NodeDefinition, NodePort, NodeRegistry
from .types import Curve


@NodeRegistry.register
class CurveConstant(BaseNode):
    """Create a constant-value curve.

    Node parameters:
        value: The constant value.
        length: Number of frames.
    """

    node_type_id: ClassVar[str] = "acestep.CurveConstant"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Constant Curve",
            category="curve",
            description="Create a constant-value per-frame curve.",
            inputs=(),
            outputs=(NodePort(name="curve", type="CURVE"),),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        value = kwargs.get("value", 1.0)
        length = kwargs.get("length", 1500)
        return {"curve": Curve(tensor=torch.full((int(length),), float(value)))}


@NodeRegistry.register
class CurveRamp(BaseNode):
    """Create a linear ramp curve.

    Node parameters:
        start: Value at frame 0.
        end: Value at final frame.
        length: Number of frames.
    """

    node_type_id: ClassVar[str] = "acestep.CurveRamp"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Ramp Curve",
            category="curve",
            description="Create a linear ramp between two values.",
            inputs=(),
            outputs=(NodePort(name="curve", type="CURVE"),),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        start = kwargs.get("start", 0.0)
        end = kwargs.get("end", 1.0)
        length = int(kwargs.get("length", 1500))
        return {
            "curve": Curve(
                tensor=torch.linspace(float(start), float(end), length)
            )
        }


@NodeRegistry.register
class CurveWave(BaseNode):
    """Create a periodic wave curve.

    Node parameters:
        wave_type: "sine", "pulse", or "square".
        frames_per_cycle: Period in frames.
        amplitude: Peak amplitude (default 1.0).
        offset: Vertical offset (default 0.0).
        length: Number of frames.
    """

    node_type_id: ClassVar[str] = "acestep.CurveWave"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Wave Curve",
            category="curve",
            description="Create a periodic wave curve (sine, pulse, square).",
            inputs=(),
            outputs=(NodePort(name="curve", type="CURVE"),),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        wave_type = kwargs.get("wave_type", "sine")
        frames_per_cycle = int(kwargs.get("frames_per_cycle", 150))
        amplitude = float(kwargs.get("amplitude", 1.0))
        offset = float(kwargs.get("offset", 0.0))
        length = int(kwargs.get("length", 1500))

        t = torch.arange(length, dtype=torch.float32)
        phase = (t % frames_per_cycle) / frames_per_cycle  # [0, 1)

        if wave_type == "sine":
            wave = torch.sin(2 * math.pi * phase)
        elif wave_type == "square":
            wave = torch.where(phase < 0.5, torch.ones_like(phase), -torch.ones_like(phase))
        elif wave_type == "pulse":
            wave = torch.where(phase < 0.5, phase * 2, (1.0 - phase) * 2)
        else:
            wave = torch.sin(2 * math.pi * phase)

        result = wave * amplitude + offset
        return {"curve": Curve(tensor=result)}


@NodeRegistry.register
class CurveMath(BaseNode):
    """Combine two curves with a math operation.

    Node parameters:
        operation: "add", "multiply", "min", "max", "lerp".
        scalar: Scalar value used when only curve_a is connected
                (e.g. curve_a * scalar).
    """

    node_type_id: ClassVar[str] = "acestep.CurveMath"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Curve Math",
            category="curve",
            description="Combine two curves with a math operation.",
            inputs=(
                NodePort(name="curve_a", type="CURVE"),
                NodePort(name="curve_b", type="CURVE", required=False),
            ),
            outputs=(NodePort(name="curve", type="CURVE"),),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        curve_a: Curve = kwargs["curve_a"]
        curve_b_payload = kwargs.get("curve_b")
        operation = kwargs.get("operation", "add")
        scalar = float(kwargs.get("scalar", 1.0))

        a = curve_a.tensor
        if curve_b_payload is not None:
            b = curve_b_payload.tensor
        else:
            b = torch.full_like(a, scalar)

        if operation == "add":
            result = a + b
        elif operation == "multiply":
            result = a * b
        elif operation == "min":
            result = torch.min(a, b)
        elif operation == "max":
            result = torch.max(a, b)
        elif operation == "lerp":
            t = float(kwargs.get("lerp_t", 0.5))
            result = a * (1.0 - t) + b * t
        else:
            result = a + b

        return {"curve": Curve(tensor=result)}
