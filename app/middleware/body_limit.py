"""Request body size limit, enforced while the body streams in."""

from __future__ import annotations

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class _BodyTooLarge(Exception):
    """Raised from the wrapped receive once the body passes the cap."""


class BodySizeLimitMiddleware:
    """Reject request bodies larger than ``max_bytes`` with a 413.

    A declared ``Content-Length`` over the cap is refused before a byte of
    body is read. Beyond that the count runs on the bytes actually received,
    so a client that omits or understates ``Content-Length`` (chunked, or just
    lying) is still cut off once it has sent too much, rather than being
    trusted to describe its own payload. ``max_bytes <= 0`` disables the check.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or self.max_bytes <= 0:
            await self.app(scope, receive, send)
            return

        declared = self._content_length(scope)
        if declared is not None and declared > self.max_bytes:
            await self._send_413(send)
            return

        received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLarge
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLarge:
            # If the route already began responding it never blocked on the
            # oversized body, so there is nothing to cut off and no clean way
            # to replace a response mid-flight; let that case propagate.
            if response_started:
                raise
            await self._send_413(send)

    @staticmethod
    def _content_length(scope: Scope) -> int | None:
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    return int(value)
                except ValueError:
                    return None
        return None

    @staticmethod
    async def _send_413(send: Send) -> None:
        body = b'{"detail":"Request body too large"}'
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
