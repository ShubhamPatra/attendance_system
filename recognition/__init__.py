"""Recognition package compatibility layer for migrated project structure."""

from . import aligner, config, detector, embedder, matcher, pipeline, tracker

__all__ = [
    "aligner",
    "config",
    "detector",
    "embedder",
    "matcher",
    "pipeline",
    "tracker",
]
