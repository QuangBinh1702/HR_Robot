"""
Face Repository - CRUD operations for Members & Embeddings.
Single source of truth for all face database operations.
"""

import numpy as np
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session, selectinload

from config.settings import EMBEDDING_SIZE
from src.database.models import Member, MemberEmbedding


# ========== Serialization Helpers ==========

def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """L2 normalize embedding vector."""
    vec = embedding.astype(np.float32).flatten()
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def serialize_embedding(embedding: np.ndarray) -> bytes:
    """Convert numpy embedding → bytes for DB storage."""
    vec = normalize_embedding(embedding)
    if vec.shape[0] != EMBEDDING_SIZE:
        raise ValueError(f"Expected {EMBEDDING_SIZE}-d embedding, got {vec.shape[0]}-d")
    return vec.tobytes()


def deserialize_embedding(blob: bytes) -> np.ndarray:
    """Convert bytes from DB → numpy embedding."""
    return np.frombuffer(blob, dtype=np.float32).copy()


# ========== Repository ==========

class FaceRepository:
    """CRUD operations for Members and MemberEmbeddings."""

    # --- Member ---

    def get_member_by_name(self, session: Session, full_name: str) -> Optional[Member]:
        """Find member by exact name."""
        return session.query(Member).filter(
            Member.full_name == full_name,
            Member.is_active == True,
        ).first()

    def get_or_create_member(
        self, session: Session, full_name: str,
        role: str = "member", contact_info: Optional[str] = None,
    ) -> Member:
        """Get existing member or create new one. Reactivates deactivated members."""
        # Check active first
        member = self.get_member_by_name(session, full_name)
        if member:
            return member

        # Check if deactivated member exists → reactivate
        inactive = session.query(Member).filter(
            Member.full_name == full_name,
            Member.is_active == False,
        ).first()
        if inactive:
            inactive.is_active = True
            inactive.updated_at = datetime.utcnow()
            session.flush()
            return inactive

        member = Member(
            full_name=full_name,
            role=role,
            contact_info=contact_info,
        )
        session.add(member)
        session.flush()
        return member

    def list_members(self, session: Session, active_only: bool = True) -> list[Member]:
        """List all members."""
        q = session.query(Member)
        if active_only:
            q = q.filter(Member.is_active == True)
        return q.order_by(Member.full_name).all()

    # --- Embedding ---

    def add_embedding(
        self, session: Session, member_id: int,
        embedding: np.ndarray, model_name: str = "buffalo_l",
    ) -> MemberEmbedding:
        """Add a face embedding for a member."""
        blob = serialize_embedding(embedding)
        emb = MemberEmbedding(
            member_id=member_id,
            embedding=blob,
            model_name=model_name,
        )
        session.add(emb)
        session.flush()
        return emb

    def count_embeddings(
        self, session: Session, member_id: int,
        model_name: Optional[str] = None,
    ) -> int:
        """Count embeddings for a member."""
        q = session.query(MemberEmbedding).filter(
            MemberEmbedding.member_id == member_id,
        )
        if model_name:
            q = q.filter(MemberEmbedding.model_name == model_name)
        return q.count()

    def fetch_all_embeddings(
        self, session: Session,
        model_name: Optional[str] = None,
        active_only: bool = True,
    ) -> list[dict]:
        """
        Fetch all embeddings (for cache rebuild).
        Returns list of {embedding_id, member_id, full_name, embedding, model_name}.
        """
        q = (
            session.query(MemberEmbedding, Member.full_name)
            .join(Member, MemberEmbedding.member_id == Member.id)
        )
        if active_only:
            q = q.filter(Member.is_active == True)
        if model_name:
            q = q.filter(MemberEmbedding.model_name == model_name)

        results = []
        for emb, name in q.all():
            results.append({
                "embedding_id": emb.id,
                "member_id": emb.member_id,
                "full_name": name,
                "embedding": deserialize_embedding(emb.embedding),
                "model_name": emb.model_name,
            })
        return results

    def list_registered_faces(
        self, session: Session,
        model_name: Optional[str] = None,
    ) -> dict:
        """
        List registered faces with embedding counts.
        Returns {name: {member_id, num_embeddings, registered_at, last_updated}}.
        """
        members = (
            session.query(Member)
            .options(selectinload(Member.embeddings))
            .filter(Member.is_active == True)
            .order_by(Member.full_name)
            .all()
        )

        result = {}
        for m in members:
            embs = m.embeddings
            if model_name:
                embs = [e for e in embs if e.model_name == model_name]
            if not embs:
                continue

            last_created = max(e.created_at for e in embs)
            result[m.full_name] = {
                "member_id": m.id,
                "num_embeddings": len(embs),
                "registered_at": m.created_at.isoformat() if m.created_at else None,
                "last_updated": last_created.isoformat() if last_created else None,
            }
        return result

    def delete_embeddings_by_name(
        self, session: Session, full_name: str,
        model_name: Optional[str] = None,
    ) -> int:
        """Delete all embeddings for a member (keeps the member row)."""
        member = self.get_member_by_name(session, full_name)
        if not member:
            return 0

        q = session.query(MemberEmbedding).filter(
            MemberEmbedding.member_id == member.id,
        )
        if model_name:
            q = q.filter(MemberEmbedding.model_name == model_name)

        count = q.count()
        q.delete(synchronize_session="fetch")
        return count
