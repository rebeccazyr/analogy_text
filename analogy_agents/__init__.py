"""Six-agent text analogy evaluation pipeline."""

from __future__ import annotations

from typing import Any


__all__ = ["PipelineConfig", "SixAgentPipeline", "load_split"]


def __getattr__(name: str) -> Any:
    """Load the API pipeline lazily so pure scoring helpers stay lightweight."""
    if name in __all__:
        from .pipeline import PipelineConfig, SixAgentPipeline, load_split

        return {
            "PipelineConfig": PipelineConfig,
            "SixAgentPipeline": SixAgentPipeline,
            "load_split": load_split,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
