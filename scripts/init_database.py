"""
HR Robot - Database Initialization Script
Creates tables and optionally seeds test data.

Usage:
    python scripts/init_database.py
    python scripts/init_database.py --seed    # With test data
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATABASE_URL
from src.database.models import Base, Member, AttendanceLog, SpaceStatus, get_engine, get_session


def init_tables():
    """Create all database tables."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    
    print(f"[DB] Database URL: {DATABASE_URL}")
    print(f"[DB] Tables created:")
    for table_name in Base.metadata.tables:
        print(f"  ✓ {table_name}")


def seed_test_data():
    """Insert sample data for testing."""
    session = get_session()
    
    # Check if data already exists
    existing = session.query(Member).count()
    if existing > 0:
        print(f"[DB] Database already has {existing} members, skipping seed")
        session.close()
        return
    
    print("[DB] Seeding test data...")
    
    # Add sample members
    members = [
        Member(full_name="Nguyễn Quang Bình", role="leader", contact_info="binh@example.com"),
        Member(full_name="Trương Bùi Diễn", role="member", contact_info="dien@example.com"),
        Member(full_name="Lê Quang Thái", role="member", contact_info="thai@example.com"),
        Member(full_name="Nguyễn Văn A", role="member", contact_info="a@example.com"),
        Member(full_name="Trần Thị B", role="member", contact_info="b@example.com"),
    ]
    
    session.add_all(members)
    session.commit()
    print(f"  ✓ Added {len(members)} members")
    
    # Add sample attendance logs (last 7 days)
    now = datetime.utcnow()
    logs = []
    for member in members:
        for day_offset in range(7):
            # Random attendance: 70% chance of showing up
            if random.random() < 0.7:
                check_in = now - timedelta(days=day_offset, hours=random.randint(8, 10))
                duration = random.randint(60, 240)  # 1-4 hours
                check_out = check_in + timedelta(minutes=duration)
                
                log = AttendanceLog(
                    member_id=member.id,
                    check_in_time=check_in,
                    check_out_time=check_out,
                    duration_minutes=duration,
                    confidence_score=random.uniform(0.7, 0.99),
                )
                logs.append(log)
    
    session.add_all(logs)
    session.commit()
    print(f"  ✓ Added {len(logs)} attendance logs")
    
    # Add sample space status
    statuses = []
    for hour_offset in range(24):
        status = SpaceStatus(
            timestamp=now - timedelta(hours=hour_offset),
            current_headcount=random.randint(0, 8),
            is_overloaded=False,
            max_capacity=20,
        )
        statuses.append(status)
    
    session.add_all(statuses)
    session.commit()
    print(f"  ✓ Added {len(statuses)} space status records")
    
    session.close()
    print("[DB] Seed complete ✓")


def verify_database():
    """Verify database is working."""
    session = get_session()
    
    members = session.query(Member).count()
    logs = session.query(AttendanceLog).count()
    statuses = session.query(SpaceStatus).count()
    
    print(f"\n[DB] Database contents:")
    print(f"  Members:         {members}")
    print(f"  Attendance Logs: {logs}")
    print(f"  Space Statuses:  {statuses}")
    
    session.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Initialize HR Robot database")
    parser.add_argument("--seed", action="store_true", help="Seed with test data")
    args = parser.parse_args()
    
    print("=" * 50)
    print("HR Robot - Database Initialization")
    print("=" * 50)
    
    init_tables()
    
    if args.seed:
        seed_test_data()
    
    verify_database()
    print("\n✓ Database ready!")


if __name__ == "__main__":
    main()
