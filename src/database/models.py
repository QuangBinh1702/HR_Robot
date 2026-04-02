"""
Database Models - SQLAlchemy ORM
Implements the 3 core tables: Members, AttendanceLogs, SpaceStatus.

Schema matches the design in cac_cong_viec.md.
"""

import sys
from pathlib import Path
from datetime import datetime

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean,
    DateTime, Text, ForeignKey, LargeBinary
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATABASE_URL

Base = declarative_base()


class Member(Base):
    """
    Bảng Members (Quản lý hồ sơ)
    Lưu trữ thông tin thành viên và vector đặc trưng khuôn mặt.
    """
    __tablename__ = 'members'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    full_name = Column(String(255), nullable=False)
    role = Column(String(100), default='member')  # member, admin, leader
    contact_info = Column(String(255), nullable=True)
    face_embedding = Column(LargeBinary, nullable=True)
    # Stores serialized numpy array of 512-d vector(s)
    # Do NOT store original photos for privacy
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    attendance_logs = relationship('AttendanceLog', back_populates='member')
    
    def __repr__(self):
        return f"<Member(id={self.id}, name='{self.full_name}', role='{self.role}')>"


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


# ========== Database Engine Setup ==========

def get_engine():
    """Create database engine."""
    return create_engine(DATABASE_URL, echo=False)


def get_session():
    """Create a new database session."""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def init_db():
    """Create all tables."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    print(f"[DB] Database initialized: {DATABASE_URL}")
    print(f"[DB] Tables: {list(Base.metadata.tables.keys())}")


if __name__ == "__main__":
    init_db()
    print("\n✓ Database setup complete!")
