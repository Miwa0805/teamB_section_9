import argparse
import asyncio
import uuid
from dataclasses import dataclass

from app.core.logging import logger, setup_logging
from app.db.session import SessionLocal
from app.domain.outfits.orchestration import DEFAULT_BATCH_TPO, generate_outfit_for_user
from app.domain.users import crud as users_crud

DEFAULT_PAGE_SIZE = 100


@dataclass(frozen=True, slots=True)
class UserBatchTarget:
    id: uuid.UUID
    default_region_code: str | None


@dataclass(frozen=True, slots=True)
class BatchJobStats:
    processed: int
    succeeded: int
    failed: int
    image_skipped: int

    @property
    def image_generated(self) -> int:
        return self.succeeded - self.image_skipped


def should_fail_job(stats: BatchJobStats) -> bool:
    return stats.processed > 0 and stats.image_generated == 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and save outfit images for all users."
    )
    parser.add_argument(
        "--tpo",
        default=DEFAULT_BATCH_TPO,
        help="Scene prompt passed to outfit generation. Defaults to business.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
        help="Number of users fetched per page.",
    )
    return parser


async def fetch_user_batch(*, limit: int, offset: int) -> list[UserBatchTarget]:
    logger.info(
        "Fetching users. limit=%d offset=%d",
        limit,
        offset,
    )

    async with SessionLocal() as db:
        users = await users_crud.list_users(db, limit=limit, offset=offset)

    logger.info("Fetched %d users", len(users))

    return [
        UserBatchTarget(
            id=user.id,
            default_region_code=user.default_region_code,
        )
        for user in users
    ]


async def process_user(*, user: UserBatchTarget, tpo: str):
    logger.info(
        "Start processing user. user=%s region=%s tpo=%s",
        user.id,
        user.default_region_code,
        tpo,
    )

    async with SessionLocal() as db:
        outfit = await generate_outfit_for_user(
            db,
            user_id=user.id,
            default_region_code=user.default_region_code,
            tpo=tpo,
            closet_mode="free",
        )

    logger.info(
        "batch outfit generated (user=%s, outfit=%s, image=%s)",
        user.id,
        outfit.id,
        outfit.coordinate_image_url,
    )

    return outfit


async def run_job(
    *,
    tpo: str = DEFAULT_BATCH_TPO,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> BatchJobStats:
    logger.info(
        "Batch job started. tpo=%s page_size=%d",
        tpo,
        page_size,
    )

    processed = 0
    succeeded = 0
    failed = 0
    image_skipped = 0
    offset = 0

    while True:
        users = await fetch_user_batch(limit=page_size, offset=offset)
        if not users:
            logger.info("No more users to process.")
            break

        for user in users:
            processed += 1

            try:
                outfit = await process_user(user=user, tpo=tpo)

                succeeded += 1

                if outfit.coordinate_image_url is None:
                    image_skipped += 1
                    logger.info(
                        "Image generation skipped. user=%s outfit=%s",
                        user.id,
                        outfit.id,
                    )

            except Exception as exc:  # noqa: BLE001 - ユーザー単位で継続する
                failed += 1

                logger.exception(
                    "batch outfit generation failed (user=%s, tpo=%s): %s",
                    user.id,
                    tpo,
                    exc,
                )

        offset += len(users)

    logger.info(
        (
            "Batch job completed. "
            "processed=%d succeeded=%d failed=%d "
            "image_generated=%d image_skipped=%d"
        ),
        processed,
        succeeded,
        failed,
        succeeded - image_skipped,
        image_skipped,
    )

    return BatchJobStats(
        processed=processed,
        succeeded=succeeded,
        failed=failed,
        image_skipped=image_skipped,
    )


async def main() -> int:
    setup_logging()
    args = build_parser().parse_args()
    stats = await run_job(tpo=args.tpo, page_size=args.page_size)
    logger.info(
        "batch outfit generation finished "
        "(processed=%d, succeeded=%d, failed=%d, image_generated=%d, image_skipped=%d)",
        stats.processed,
        stats.succeeded,
        stats.failed,
        stats.image_generated,
        stats.image_skipped,
    )
    return 1 if should_fail_job(stats) else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
