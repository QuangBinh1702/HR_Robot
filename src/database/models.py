"""
Database Models - SQLAlchemy ORM
Core tables: Members, MemberEmbedding, AttendanceLogs, SpaceStatus.
"""

import sys
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean,
    DateTime, Text, ForeignKey, LargeBinary, event
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATABASE_URL

Base = declarative_base()


class Member(Base):
    """
    Bảng Members (Quản lý hồ sơ)
    Lưu trữ thông tin thành viên.
    """
    __tablename__ = 'members'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    full_name = Column(String(255), nullable=False)
    role = Column(String(100), default='member')  # member, admin, leader
    contact_info = Column(String(255), nullable=True)
    # DEPRECATED: dùng bảng member_embeddings thay thế. Giữ lại để không break schema cũ.
    face_embedding = Column(LargeBinary, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    attendance_logs = relationship('AttendanceLog', back_populates='member')
    embeddings = relationship('MemberEmbedding', back_populates='member', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f"<Member(id={self.id}, name='{self.full_name}', role='{self.role}')>"


class MemberEmbedding(Base):
    """
    Bảng MemberEmbeddings (Face embeddings)
    Mỗi member có thể có nhiều embedding (multi-angle capture).
    Mỗi embedding là 512-d float32 vector = 2048 bytes.
    """
    __tablename__ = 'member_embeddings'

    id = Column(Integer, primary_key=True, autoincrement=True)
    member_id = Column(Integer, ForeignKey('members.id', ondelete='CASCADE'), nullable=False, index=True)
    embedding = Column(LargeBinary, nullable=False)  # 512 x float32 = 2048 bytes
    model_name = Column(String(50), nullable=False, default='buffalo_l')
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    member = relationship('Member', back_populates='embeddings')

    def __repr__(self):
        return f"<MemberEmbedding(id={self.id}, member_id={self.member_id}, model='{self.model_name}')>"


class AttendanceLog(Base):
    """
    Bảng AttendanceLogs (Lịch sử ra vào)
    Ghi nhận check-in, check-out và thời gian lưu lại.
    """
    __tablename__ = 'attendance_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    member_id = Column(Integer, ForeignKey('members.id'), nullable=False)
    
    check_in_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    check_out_time = Column(DateTime, nullable=True)
    duration_minutes = Column(Float, nullable=True)
    # Auto-calculated: (check_out - check_in) in minutes
    
    confidence_score = Column(Float, nullable=True)
    # Recognition confidence at check-in
    
    # Relationships
    member = relationship('Member', back_populates='attendance_logs')
    
    def __repr__(self):
        return (f"<AttendanceLog(id={self.id}, member={self.member_id}, "
                f"in={self.check_in_time}, out={self.check_out_time})>")


class SpaceStatus(Base):
    """
    Bảng SpaceStatus (Theo dõi lưu lượng)
    Ghi nhận số người có mặt theo thời gian thực.
    """
    __tablename__ = 'space_status'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    current_headcount = Column(Integer, nullable=False, default=0)
    is_overloaded = Column(Boolean, default=False)
    max_capacity = Column(Integer, default=20)
    
    def __repr__(self):
        return (f"<SpaceStatus(id={self.id}, count={self.current_headcount}, "
                f"overloaded={self.is_overloaded})>")


# ========== Database Engine Setup (Singleton + WAL) ==========

def _build_engine():
    """Create database engine with SQLite optimizations."""
    connect_args = {}
    if DATABASE_URL.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    eng = create_engine(DATABASE_URL, echo=False, connect_args=connect_args)

    if DATABASE_URL.startswith("sqlite"):
        @event.listens_for(eng, "connect")
        def _set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON;")
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
            cursor.close()

    return eng


engine = _build_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def get_engine():
    """Return the singleton engine."""
    return engine


def get_session():
    """Create a new database session."""
    return SessionLocal()


@contextmanager
def session_scope():
    """Context manager: auto commit/rollback/close."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db():
    """Create all tables."""
    Base.metadata.create_all(engine)
    print(f"[DB] Database initialized: {DATABASE_URL}")
    print(f"[DB] Tables: {list(Base.metadata.tables.keys())}")


if __name__ == "__main__":
    init_db()
    print("\n✓ Database setup complete!")
