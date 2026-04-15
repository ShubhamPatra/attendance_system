"""Face embedder compatibility module."""

from app_vision.face_engine import (
	encoding_cache,
	generate_encoding,
	get_arcface_backend,
	get_embedding_backend_name,
)

__all__ = [
	"encoding_cache",
	"generate_encoding",
	"get_arcface_backend",
	"get_embedding_backend_name",
]
