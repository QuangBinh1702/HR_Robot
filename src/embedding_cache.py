"""
Embedding Cache - In-memory NumPy matrix for fast face matching.
Loads all embeddings from SQLite → NumPy matrix, matches via dot product.
"""

import numpy as np
from typing import Optional

from config.settings import EMBEDDING_SIZE
from src.database.models import session_scope
from src.database.repository import FaceRepository, normalize_embedding


class EmbeddingCache:
    """
    In-memory cache of all face embeddings for fast matching.
    Source of truth is SQLite — this is a read-only cache that rebuilds on demand.
    """

    def __init__(self, repository: FaceRepository, model_name: Optional[str] = None):
        self.repository = repository
        self.model_name = model_name
        self._clear()

    def _clear(self):
        """Reset cache to empty state."""
        self.matrix = np.empty((0, EMBEDDING_SIZE), dtype=np.float32)
        self.member_ids = np.empty((0,), dtype=np.int32)
        self.member_names: list[str] = []
        self.embedding_ids = np.empty((0,), dtype=np.int32)

    def rebuild(self) -> int:
        """
        Reload all embeddings from DB into memory.
        Returns number of embeddings loaded.
        """
        with session_scope() as session:
            rows = self.repository.fetch_all_embeddings(
                session, model_name=self.model_name, active_only=True,
            )

        if not rows:
            self._clear()
            return 0

        valid_rows = []
        for row in rows:
            try:
                embedding = normalize_embedding(row["embedding"])
            except ValueError as exc:
                print(
                    f"[EmbeddingCache] Skipping invalid embedding "
                    f"id={row['embedding_id']} member_id={row['member_id']}: {exc}"
                )
                continue
            valid_rows.append({**row, "embedding": embedding})

        if not valid_rows:
            self._clear()
            return 0

        self.matrix = np.vstack([r["embedding"] for r in valid_rows]).astype(np.float32)
        self.member_ids = np.array([r["member_id"] for r in valid_rows], dtype=np.int32)
        self.member_names = [r["full_name"] for r in valid_rows]
        self.embedding_ids = np.array([r["embedding_id"] for r in valid_rows], dtype=np.int32)

        return len(valid_rows)

    @property
    def is_empty(self) -> bool:
        return self.matrix.shape[0] == 0

    @property
    def size(self) -> int:
        return self.matrix.shape[0]

    def match(self, query_embedding: np.ndarray, threshold: float) -> tuple[str, float, Optional[int]]:
        """
        Match query embedding against cache using dot product.
        (Embeddings are L2-normalized, so dot product == cosine similarity.)

        Returns:
            (name, score, member_id) or ("Người lạ", best_score, None)
        """
        if self.is_empty:
            return ("Người lạ", 0.0, None)

        try:
            query = normalize_embedding(query_embedding)
        except ValueError:
            return ("Người lạ", 0.0, None)

        scores = self.matrix @ query
        if not np.all(np.isfinite(scores)):
            return ("Người lạ", 0.0, None)

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score < threshold:
            return ("Người lạ", best_score, None)

        return (
            self.member_names[best_idx],
            best_score,
            int(self.member_ids[best_idx]),
        )
