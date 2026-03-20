"""Node framework: BaseNode, NodePort, NodeDefinition, NodeRegistry.

Every node is a class that:
  1. Declares its ports via a static NodeDefinition (get_definition)
  2. Implements execute(**inputs) -> dict[str, payload]

The NodeRegistry provides node discovery and connection validation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

from .types import types_compatible


@dataclass(frozen=True)
class NodePort:
    """Descriptor for a single input or output port."""
    name: str
    type: str  # TYPE_NAME from types.py (e.g. "LATENT", "CONDITIONING")
    required: bool = True
    description: str = ""


@dataclass(frozen=True)
class NodeDefinition:
    """Static metadata describing a node type."""
    node_type_id: str  # e.g. "acestep.VAEEncode"
    display_name: str  # e.g. "VAE Encode Audio"
    category: str  # e.g. "vae", "conditioning", "diffusion"
    description: str = ""
    inputs: Tuple[NodePort, ...] = ()
    outputs: Tuple[NodePort, ...] = ()

    def input_port(self, name: str) -> Optional[NodePort]:
        """Look up an input port by name."""
        for p in self.inputs:
            if p.name == name:
                return p
        return None

    def output_port(self, name: str) -> Optional[NodePort]:
        """Look up an output port by name."""
        for p in self.outputs:
            if p.name == name:
                return p
        return None


class BaseNode(ABC):
    """Abstract base class for all graph nodes.

    Subclasses set node_type_id as a ClassVar and implement
    get_definition() and execute().
    """

    node_type_id: ClassVar[str]

    @classmethod
    @abstractmethod
    def get_definition(cls) -> NodeDefinition:
        """Return the static definition for this node type."""
        ...

    @abstractmethod
    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """Execute the node.

        Args:
            **kwargs: Input payloads keyed by input port name.
                Required ports are guaranteed present and type-checked.
                Optional ports may be absent (not in kwargs).

        Returns:
            Dict mapping output port name to payload value.
            Every declared output port should have a key.
        """
        ...


class NodeRegistry:
    """Registry for node type discovery and connection validation."""

    _nodes: Dict[str, Type[BaseNode]] = {}

    @classmethod
    def register(cls, node_class: Type[BaseNode]) -> Type[BaseNode]:
        """Register a node class. Can be used as a decorator."""
        defn = node_class.get_definition()
        cls._nodes[defn.node_type_id] = node_class
        return node_class

    @classmethod
    def get(cls, node_type_id: str) -> Optional[Type[BaseNode]]:
        """Look up a node class by type ID."""
        return cls._nodes.get(node_type_id)

    @classmethod
    def all_definitions(cls) -> List[NodeDefinition]:
        """Return definitions for all registered nodes."""
        return [nc.get_definition() for nc in cls._nodes.values()]

    @classmethod
    def list_node_types(cls) -> List[str]:
        """Return all registered node type IDs."""
        return list(cls._nodes.keys())

    @classmethod
    def validate_connection(
        cls,
        source_node_type: str,
        source_port_name: str,
        target_node_type: str,
        target_port_name: str,
    ) -> Tuple[bool, str]:
        """Validate that a connection between two ports is type-safe.

        Returns:
            (valid, reason) tuple. If valid is False, reason explains why.
        """
        src_cls = cls._nodes.get(source_node_type)
        dst_cls = cls._nodes.get(target_node_type)

        if src_cls is None:
            return False, f"Unknown source node type: {source_node_type}"
        if dst_cls is None:
            return False, f"Unknown target node type: {target_node_type}"

        src_defn = src_cls.get_definition()
        dst_defn = dst_cls.get_definition()

        src_port = src_defn.output_port(source_port_name)
        dst_port = dst_defn.input_port(target_port_name)

        if src_port is None:
            return False, f"{source_node_type} has no output port '{source_port_name}'"
        if dst_port is None:
            return False, f"{target_node_type} has no input port '{target_port_name}'"

        if not types_compatible(src_port.type, dst_port.type):
            return (
                False,
                f"Type mismatch: {src_port.type} -> {dst_port.type} "
                f"({source_port_name} -> {target_port_name})",
            )

        return True, ""

    @classmethod
    def clear(cls) -> None:
        """Remove all registered nodes. Primarily for testing."""
        cls._nodes.clear()
