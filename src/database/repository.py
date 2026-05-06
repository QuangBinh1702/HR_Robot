"""
Face Repository - CRUD operations for Members & Embeddings.
Single source of truth for all face database operations.
"""

import numpy as np
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session, selectinload

from config.settings import EMBEDDING_SIZE
from src.database.models import Member, MemberEmbedding, AttendanceLog, SpaceStatus


# ========== Serialization Helpers ==========

def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """L2 normalize embedding vector."""
    vec = embedding.astype(np.float32).flatten()
    if not np.all(np.isfinite(vec)):
        raise ValueError("Embedding contains NaN or infinite values")

    norm = np.linalg.norm(vec)
    if norm <= 0:
        raise ValueError("Embedding norm must be positive")

    vec = vec / norm
    return vec


def serialize_embedding(embedding: np.ndarray) -> bytes:
    """Convert numpy embedding → bytes for DB storage."""
    vec = embedding.astype(np.float32).flatten()
    if vec.shape[0] != EMBEDDING_SIZE:
        raise ValueError(f"Expected {EMBEDDING_SIZE}-d embedding, got {vec.shape[0]}-d")
    vec = normalize_embedding(vec)
    return vec.tobytes()


def deserialize_embedding(blob: bytes) -> np.ndarray:
    """Convert bytes from DB → numpy embedding."""
    return np.frombuffer(blob, dtype=np.float32).copy()


# ========== Repository ==========

class FaceRepository:
    """CRUD operations for Members and MemberEmbeddings."""

    # --- Member ---

    def get_member_by_id(self, session: Session, member_id: int) -> Optional[Member]:
        """Find member by id."""
        return session.query(Member).filter(
            Member.id == member_id,
            Member.is_active == True,
        ).first()

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

    # --- Attendance ---

    def get_open_attendance_log(self, session: Session, member_id: int) -> Optional[AttendanceLog]:
        """Get open (not checked-out) attendance log for a member."""
        return session.query(AttendanceLog).filter(
            AttendanceLog.member_id == member_id,
            AttendanceLog.check_out_time == None,
        ).first()

    def list_open_attendance_logs(self, session: Session) -> list[AttendanceLog]:
        """List all open attendance logs (for startup restore)."""
        return session.query(AttendanceLog).filter(
            AttendanceLog.check_out_time == None,
        ).all()

    def log_checkin(
        self, session: Session, member_id: int,
        confidence_score: float,
        check_in_time: Optional[datetime] = None,
    ) -> AttendanceLog:
        """
        Create a check-in record. Idempotent: returns existing open log if present.
        """
        existing = self.get_open_attendance_log(session, member_id)
        if existing:
            return existing

        log = AttendanceLog(
            member_id=member_id,
            check_in_time=check_in_time or datetime.utcnow(),
            confidence_score=confidence_score,
        )
        session.add(log)
        session.flush()
        return log

    def log_checkout(
        self, session: Session, attendance_log_id: int,
        check_out_time: Optional[datetime] = None,
    ) -> Optional[AttendanceLog]:
        """
        Close an attendance log. Computes duration_minutes.
        Returns None if log not found or already closed.
        """
        log = session.query(AttendanceLog).filter(
            AttendanceLog.id == attendance_log_id,
        ).first()

        if not log or log.check_out_time is not None:
            return None

        out_time = check_out_time or datetime.utcnow()
        log.check_out_time = out_time
        log.duration_minutes = (out_time - log.check_in_time).total_seconds() / 60.0
        session.flush()
        return log

    # --- Space Status ---

    def update_headcount(
        self, session: Session, current_headcount: int,
        max_capacity: int,
        timestamp: Optional[datetime] = None,
    ) -> SpaceStatus:
        """Insert a new SpaceStatus record (append-only)."""
        status = SpaceStatus(
            timestamp=timestamp or datetime.utcnow(),
            current_headcount=current_headcount,
            max_capacity=max_capacity,
            is_overloaded=current_headcount > max_capacity,
        )
        session.add(status)
        session.flush()
        return status

    def get_latest_space_status(self, session: Session) -> Optional[SpaceStatus]:
        """Get the most recent space status record."""
        return session.query(SpaceStatus).order_by(
            SpaceStatus.timestamp.desc()
        ).first()

    # --- API Queries ---

    def list_attendance_logs(
        self, session: Session,
        start_dt: Optional[datetime] = None,
        end_dt: Optional[datetime] = None,
        member_id: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        """
        Query attendance logs with filters. Returns (items, total_count).
        Each item includes member full_name via join.
        """
        q = (
            session.query(AttendanceLog, Member.full_name)
            .join(Member, AttendanceLog.member_id == Member.id)
        )
        if start_dt:
            q = q.filter(AttendanceLog.check_in_time >= start_dt)
        if end_dt:
            q = q.filter(AttendanceLog.check_in_time < end_dt)
        if member_id:
            q = q.filter(AttendanceLog.member_id == member_id)

        total = q.count()
        rows = q.order_by(AttendanceLog.check_in_time.desc()).offset(offset).limit(limit).all()

        items = []
        for log, full_name in rows:
            items.append({
                "id": log.id,
                "member_id": log.member_id,
                "full_name": full_name,
                "check_in_time": log.check_in_time.isoformat() if log.check_in_time else None,
                "check_out_time": log.check_out_time.isoformat() if log.check_out_time else None,
                "duration_minutes": round(log.duration_minutes, 1) if log.duration_minutes else None,
                "confidence_score": round(log.confidence_score, 2) if log.confidence_score else None,
                "status": "present" if log.check_out_time is None else "checked_out",
            })
        return items, total

    def get_attendance_stats(
        self, session: Session,
        start_dt: datetime,
        end_dt: datetime,
    ) -> dict:
        """
        Compute attendance KPI stats for a date range.
        Returns: avg_duration, attendance_rate, peak_hour, daily_attendance, peak_hours
        """
        from sqlalchemy import func, extract

        # All logs in range
        logs = (
            session.query(AttendanceLog)
            .filter(AttendanceLog.check_in_time >= start_dt)
            .filter(AttendanceLog.check_in_time < end_dt)
            .all()
        )

        if not logs:
            return {
                "avg_duration_minutes": 0,
                "attendance_rate_pct": 0,
                "total_checkins": 0,
                "peak_hour": None,
                "daily_attendance": [],
                "peak_hours": [],
            }

        # Avg duration (only completed sessions)
        durations = [l.duration_minutes for l in logs if l.duration_minutes is not None]
        avg_duration = round(sum(durations) / len(durations), 1) if durations else 0

        # Attendance rate
        unique_members = len(set(l.member_id for l in logs))
        total_active = session.query(Member).filter(Member.is_active == True).count()
        rate = round((unique_members / total_active) * 100, 1) if total_active > 0 else 0

        # Peak hours histogram
        hour_counts: dict[int, int] = {}
        for l in logs:
            h = l.check_in_time.hour
            hour_counts[h] = hour_counts.get(h, 0) + 1

        peak_hours = [{"hour": h, "checkins": c} for h, c in sorted(hour_counts.items())]
        peak_hour = max(peak_hours, key=lambda x: x["checkins"]) if peak_hours else None

        # Daily attendance (unique members per day)
        day_members: dict[str, set] = {}
        for l in logs:
            day_str = l.check_in_time.strftime("%Y-%m-%d")
            if day_str not in day_members:
                day_members[day_str] = set()
            day_members[day_str].add(l.member_id)

        daily_attendance = [
            {"date": d, "unique_members": len(members)}
            for d, members in sorted(day_members.items())
        ]

        return {
            "avg_duration_minutes": avg_duration,
            "attendance_rate_pct": rate,
            "total_checkins": len(logs),
            "peak_hour": peak_hour,
            "daily_attendance": daily_attendance,
            "peak_hours": peak_hours,
        }
