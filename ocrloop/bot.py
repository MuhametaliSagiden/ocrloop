"""Telegram OCR bot entry point.

Run with ``python -m ocrloop.bot`` (after exporting ``BOT_TOKEN``).

Behavior:
    * A single photo → OCR it and reply with the recognised text only.
    * An album of up to 20 photos → OCR each in the order they were sent,
      concatenate the per-image results separated by a blank line, and reply
      once. If the combined text exceeds Telegram's 4096-char message limit,
      it's split across multiple messages on whitespace boundaries.
    * Documents whose mime type is an image are also accepted.
"""

from __future__ import annotations

import asyncio
import logging
import os
from io import BytesIO

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiohttp import web
from dotenv import load_dotenv

from .album import AlbumMiddleware
from .ocr import OCRConfig, extract_text

logger = logging.getLogger("ocrloop")

TELEGRAM_MESSAGE_LIMIT = 4096
MAX_ALBUM_SIZE = 20  # Telegram's own album cap


def _split_for_telegram(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """Split ``text`` into chunks that each fit in a single Telegram message.

    We split on the latest newline before the limit so we never break a line
    of recognised text in half.
    """
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind("\n", 0, limit)
        if cut == -1:
            cut = remaining.rfind(" ", 0, limit)
        if cut == -1:
            cut = limit
        chunks.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return chunks


async def _download_photo(bot: Bot, message: Message) -> bytes | None:
    """Download the highest-resolution photo or image document from a message."""
    file_id: str | None = None
    if message.photo:
        # Largest size is the last one in the list.
        file_id = message.photo[-1].file_id
    elif message.document and (message.document.mime_type or "").startswith("image/"):
        file_id = message.document.file_id
    if file_id is None:
        return None
    buf = BytesIO()
    await bot.download(file_id, destination=buf)
    return buf.getvalue()


async def handle_photos(message: Message, bot: Bot, album: list[Message]) -> None:
    cfg = OCRConfig(
        langs=os.getenv("OCR_LANGS", "rus+eng"),
    )
    # Cap at MAX_ALBUM_SIZE photos defensively (Telegram already enforces this).
    album = album[:MAX_ALBUM_SIZE]

    pieces: list[str] = []
    for msg in album:
        data = await _download_photo(bot, msg)
        if data is None:
            continue
        try:
            # Run blocking OCR in a worker thread so the event loop stays
            # responsive when several albums arrive in parallel.
            text = await asyncio.to_thread(extract_text, data, cfg)
        except Exception:  # pragma: no cover - defensive
            logger.exception("OCR failed for message %s", msg.message_id)
            continue
        if text:
            pieces.append(text)

    if not pieces:
        # The user asked for no conversational filler, but we still need to
        # signal "nothing was found" otherwise the message looks lost.
        await message.reply("—")
        return

    combined = "\n\n".join(pieces)
    for chunk in _split_for_telegram(combined):
        await message.reply(chunk, disable_web_page_preview=True)


async def handle_start(message: Message) -> None:
    await message.answer(
        "Send one or more photos (up to 20 in one album) — "
        "I'll reply with the extracted text."
    )


def build_dispatcher(latency: float) -> Dispatcher:
    dp = Dispatcher()
    album_mw = AlbumMiddleware(latency=latency)
    dp.message.middleware(album_mw)

    dp.message.register(handle_start, CommandStart())
    photo_filter = F.photo | (
        F.document & F.document.mime_type.startswith("image/")
    )
    dp.message.register(handle_photos, photo_filter)
    return dp


async def _start_health_server(port: int) -> web.AppRunner:
    """Run a tiny health-check HTTP server alongside the bot.

    Some hosts (Koyeb, Render Web Service, Fly.io with public IP, …) only
    expose **Web Service** plans on their free tier, which require the app
    to listen on a TCP port. This bot itself uses long polling and has no
    HTTP surface, so we add a no-op ``/`` and ``/health`` endpoint here.
    Activated only when ``PORT`` is set, so plain Worker / VM deploys
    aren't affected.
    """
    app = web.Application()

    async def ok(_request: web.Request) -> web.Response:
        return web.Response(text="ok")

    app.router.add_get("/", ok)
    app.router.add_get("/health", ok)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=port)  # noqa: S104
    await site.start()
    logger.info("Health server listening on 0.0.0.0:%d", port)
    return runner


async def main() -> None:
    load_dotenv()
    token = os.environ.get("BOT_TOKEN")
    if not token:
        raise SystemExit("BOT_TOKEN environment variable is required.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    latency = float(os.environ.get("ALBUM_LATENCY", "1.2"))
    bot = Bot(token=token, default=DefaultBotProperties(parse_mode=None))
    dp = build_dispatcher(latency=latency)

    health_runner: web.AppRunner | None = None
    port_env = os.environ.get("PORT")
    # HuggingFace Spaces' Docker SDK doesn't set PORT — instead it expects
    # the container to listen on whatever port is declared in the README
    # frontmatter (we use 7860, the HF default). Detect that environment via
    # the always-present ``SPACE_ID`` variable so the health server still
    # comes up.
    if not port_env and os.environ.get("SPACE_ID"):
        port_env = "7860"
    if port_env:
        try:
            health_runner = await _start_health_server(int(port_env))
        except (ValueError, OSError) as exc:
            logger.warning("Failed to start health server on PORT=%r: %s", port_env, exc)

    logger.info("Starting bot polling")
    try:
        await dp.start_polling(bot)
    finally:
        if health_runner is not None:
            await health_runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
