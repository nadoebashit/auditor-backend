from __future__ import annotations

import argparse
import asyncio
import io
import sys
import uuid
from pathlib import Path

from arq import create_pool
from arq.connections import RedisSettings
from sqlalchemy.orm import Session

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.core.config import settings
from app.core.db import Base, SessionLocal, _import_models, engine
from app.core.logging import configure_logging, get_logger
from app.modules.auth.models import User
from app.modules.files.models import FileIndexStatus, FileScope, StoredFile
from app.modules.files.storage import FileStorage, S3Config

logger = get_logger(__name__)


def _get_storage() -> FileStorage:
    return FileStorage(
        cfg=S3Config(
            endpoint_url=settings.S3_ENDPOINT_URL,
            access_key=settings.S3_ACCESS_KEY,
            secret_key=settings.S3_SECRET_KEY,
            region=settings.S3_REGION,
            bucket_admin_laws=settings.S3_BUCKET_ADMIN_LAWS,
            bucket_customer_docs=settings.S3_BUCKET_CUSTOMER_DOCS,
        )
    )


def _resolve_admin_user_id(db: Session, admin_user_id: str | None) -> uuid.UUID:
    if admin_user_id:
        try:
            return uuid.UUID(admin_user_id)
        except Exception as exc:
            raise ValueError("admin_user_id must be a valid UUID") from exc

    user = (
        db.query(User)
        .filter(User.is_admin.is_(True), User.is_active.is_(True))
        .order_by(User.created_at.asc())
        .first()
    )
    if user is None:
        raise RuntimeError("No active admin user found. Pass --admin-user-id")
    return user.id


def _iter_kb_files(prompts_dir: Path) -> list[Path]:
    out: list[Path] = []
    for pattern in ["B*_*.txt", "C*_*.txt", "D*_*.txt", "F*_*.txt"]:
        out.extend(sorted(prompts_dir.glob(pattern)))
    return [p for p in out if p.is_file()]


async def seed_kb(
    *,
    admin_user_id: str | None,
    force: bool,
    dry_run: bool,
    prompts_dir: str | None,
) -> int:
    configure_logging()

    _import_models()
    Base.metadata.create_all(bind=engine)

    repo_root = Path(__file__).resolve().parent.parent
    resolved_prompts_dir = Path(prompts_dir) if prompts_dir else (repo_root / "app" / "prompts" / "knowledge")
    if not resolved_prompts_dir.exists():
        raise RuntimeError(f"Prompts directory not found: {resolved_prompts_dir}")

    files = _iter_kb_files(resolved_prompts_dir)
    if not files:
        logger.warning("No KB files found", extra={"prompts_dir": str(resolved_prompts_dir)})
        return 0

    db = SessionLocal()
    try:
        owner_id = _resolve_admin_user_id(db, admin_user_id)
        storage = _get_storage()

        redis = await create_pool(RedisSettings.from_dsn(str(settings.REDIS_URL)))

        created = 0
        enqueued = 0
        skipped = 0

        for path in files:
            filename = path.name

            existing = (
                db.query(StoredFile)
                .filter(
                    StoredFile.scope == FileScope.ADMIN_LAW,
                    StoredFile.original_filename == filename,
                )
                .order_by(StoredFile.uploaded_at.desc())
                .first()
            )

            if existing is not None and not force:
                skipped += 1
                continue

            content = path.read_bytes()
            object_key = f"{uuid.uuid4()}_{filename}"

            if not dry_run:
                storage.upload_admin_file(
                    file_obj=io.BytesIO(content),
                    object_key=object_key,
                    content_type="text/plain",
                )

                stored_file = StoredFile(
                    owner_id=owner_id,
                    customer_id=None,
                    scope=FileScope.ADMIN_LAW,
                    bucket=storage._cfg.bucket_admin_laws,
                    object_key=object_key,
                    original_filename=filename,
                    content_type="text/plain",
                    size_bytes=len(content),
                    is_indexed=False,
                    index_error=None,
                    index_status=FileIndexStatus.QUEUED,
                )

                db.add(stored_file)
                db.commit()
                db.refresh(stored_file)
                created += 1

                await redis.enqueue_job("index_file_task", str(stored_file.id))
                enqueued += 1

        try:
            logger.info(
                "KB seed completed",
                extra={
                    "kb_created": created,
                    "kb_enqueued": enqueued,
                    "kb_skipped": skipped,
                    "dry_run": dry_run,
                    "force": force,
                    "prompts_dir": str(resolved_prompts_dir),
                },
            )
            return 0
        finally:
            try:
                if hasattr(redis, "aclose"):
                    await redis.aclose()  # type: ignore[attr-defined]
                else:
                    res = redis.close()
                    if asyncio.iscoroutine(res):
                        await res
            except Exception:
                pass
            try:
                await redis.wait_closed()  # type: ignore[attr-defined]
            except Exception:
                pass
    finally:
        try:
            db.close()
        except Exception:
            pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--admin-user-id", dest="admin_user_id", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prompts-dir", dest="prompts_dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    asyncio.run(
        seed_kb(
            admin_user_id=args.admin_user_id,
            force=bool(args.force),
            dry_run=bool(args.dry_run),
            prompts_dir=args.prompts_dir,
        )
    )


if __name__ == "__main__":
    main()
