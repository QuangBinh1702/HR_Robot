"""
Migrate face database from .npz + .json files to SQLite.
One-time migration script.

Usage:
    python scripts/migrate_npz_to_db.py              # Run migration
    python scripts/migrate_npz_to_db.py --dry-run    # Preview only
    python scripts/migrate_npz_to_db.py --force      # Overwrite existing embeddings
    python scripts/migrate_npz_to_db.py --model buffalo_l  # Specify model name
"""

import sys
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import FACE_DB_DIR, EMBEDDING_SIZE
from src.database.models import session_scope, MemberEmbedding, init_db
from src.database.repository import FaceRepository

repo = FaceRepository()


def backup_legacy_files() -> Path:
    """Backup .npz and .json before migration."""
    backup_dir = FACE_DB_DIR / "backup_before_migration"
    backup_dir.mkdir(parents=True, exist_ok=True)

    for fname in ["face_database.json", "embeddings.npz"]:
        src = FACE_DB_DIR / fname
        if src.exists():
            dst = backup_dir / f"{fname}.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(src, dst)
            print(f"  Backup: {src.name} -> {dst.name}")

    return backup_dir


def load_legacy_database() -> tuple[dict, dict]:
    """
    Load legacy face_database.json + embeddings.npz.
    Returns (face_info, face_embeddings) where:
        face_info = {name: {registered_at, num_embeddings, ...}}
        face_embeddings = {name: np.ndarray shape (N, 512)}
    """
    json_path = FACE_DB_DIR / "face_database.json"
    npz_path = FACE_DB_DIR / "embeddings.npz"

    if not json_path.exists():
        raise FileNotFoundError(f"Not found: {json_path}")
    if not npz_path.exists():
        raise FileNotFoundError(f"Not found: {npz_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        face_info = json.load(f)

    data = np.load(npz_path, allow_pickle=True)

    face_embeddings = {}
    for name in face_info:
        key = f"emb_{name}"
        if key in data:
            embs = data[key]
            if embs.ndim == 1:
                embs = embs.reshape(1, -1)
            face_embeddings[name] = embs
        else:
            print(f"  [WARN] Key '{key}' not found in npz, skipping '{name}'")

    return face_info, face_embeddings


def migrate(force: bool = False, dry_run: bool = False, model_name: str = "buffalo_l"):
    """Run the migration."""
    print("=" * 50)
    print("Migration: .npz + .json -> SQLite")
    print("=" * 50)

    # 1. Check if DB already has embeddings
    with session_scope() as s:
        existing_count = s.query(MemberEmbedding).count()

    if existing_count > 0 and not force:
        print(f"\n[ABORT] DB already has {existing_count} embeddings.")
        print("Use --force to overwrite.")
        return

    if existing_count > 0 and force:
        print(f"\n[FORCE] Clearing {existing_count} existing embeddings...")
        if not dry_run:
            with session_scope() as s:
                s.query(MemberEmbedding).delete()

    # 2. Load legacy data
    print("\n--- Loading legacy files ---")
    try:
        face_info, face_embeddings = load_legacy_database()
    except FileNotFoundError as e:
        print(f"[ABORT] {e}")
        return

    print(f"  Persons: {len(face_info)}")
    for name, info in face_info.items():
        n_embs = face_embeddings.get(name, np.empty((0,))).shape[0] if name in face_embeddings else 0
        print(f"    - {name}: {n_embs} embeddings (json says {info.get('num_embeddings', '?')})")

    total_embeddings = sum(e.shape[0] for e in face_embeddings.values())
    print(f"  Total embeddings to migrate: {total_embeddings}")

    if dry_run:
        print("\n[DRY RUN] No changes made.")
        return

    # 3. Backup
    print("\n--- Backing up legacy files ---")
    backup_legacy_files()

    # 4. Migrate
    print("\n--- Migrating to SQLite ---")
    members_created = 0
    members_reused = 0
    embeddings_imported = 0
    skipped = 0

    with session_scope() as s:
        for name, embs_array in face_embeddings.items():
            member = repo.get_member_by_name(s, name)
            if member:
                members_reused += 1
                print(f"  [REUSE] Member '{name}' (id={member.id})")
            else:
                member = repo.get_or_create_member(s, name)
                members_created += 1
                print(f"  [NEW]   Member '{name}' (id={member.id})")

            for i in range(embs_array.shape[0]):
                emb = embs_array[i]
                if emb.shape[0] != EMBEDDING_SIZE:
                    print(f"    [SKIP] Embedding #{i} has wrong dim: {emb.shape[0]}")
                    skipped += 1
                    continue

                repo.add_embedding(s, member.id, emb, model_name=model_name)
                embeddings_imported += 1

    # 5. Verify
    print("\n--- Verification ---")
    with session_scope() as s:
        db_count = s.query(MemberEmbedding).count()
        faces = repo.list_registered_faces(s, model_name=model_name)

    print(f"  Embeddings in DB: {db_count}")
    print(f"  Registered faces: {len(faces)}")
    for name, info in faces.items():
        print(f"    - {name}: {info['num_embeddings']} samples")

    assert db_count == embeddings_imported, f"Mismatch: DB={db_count}, imported={embeddings_imported}"

    # 6. Summary
    print(f"\n{'=' * 50}")
    print(f"Migration complete!")
    print(f"  Members created: {members_created}")
    print(f"  Members reused:  {members_reused}")
    print(f"  Embeddings imported: {embeddings_imported}")
    print(f"  Skipped: {skipped}")
    print(f"{'=' * 50}")


def main():
    parser = argparse.ArgumentParser(description="Migrate face DB from npz to SQLite")
    parser.add_argument("--dry-run", action="store_true", help="Preview without changes")
    parser.add_argument("--force", action="store_true", help="Overwrite existing embeddings")
    parser.add_argument("--model", type=str, default="buffalo_l", help="Model name tag")
    args = parser.parse_args()

    init_db()
    migrate(force=args.force, dry_run=args.dry_run, model_name=args.model)


if __name__ == "__main__":
    main()
