"""Aggregate Telegram album messages into a single batch.

Telegram delivers an "album" (media group) as N separate ``Message`` updates
sharing the same ``media_group_id``. We need to OCR them in order and reply
once with the combined text, so the middleware below buffers them, waits a
short ``latency`` for the rest to arrive, and then dispatches the handler
exactly once with the full sorted list under ``data['album']``.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware
from aiogram.types import Message


class AlbumMiddleware(BaseMiddleware):
    """Collect album messages and pass them to the handler as a single list."""

    def __init__(self, latency: float = 1.2) -> None:
        self.latency = latency
        self._buffers: dict[str, list[Message]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def __call__(  # type: ignore[override]
        self,
        handler: Callable[[Message, dict[str, Any]], Awaitable[Any]],
        event: Message,
        data: dict[str, Any],
    ) -> Any:
        mgid = event.media_group_id
        if not mgid:
            data["album"] = [event]
            return await handler(event, data)

        is_leader = False
        async with self._lock:
            if mgid not in self._buffers:
                is_leader = True
            self._buffers[mgid].append(event)

        if not is_leader:
            # A leader is already waiting — it will dispatch the handler.
            return None

        await asyncio.sleep(self.latency)
        async with self._lock:
            messages = self._buffers.pop(mgid, [])
        messages.sort(key=lambda m: m.message_id)
        data["album"] = messages
        return await handler(event, data)
