"""Typed exceptions for the gesim public API."""


class GesimError(Exception):
    """Base class for all gesim errors."""


class GesimConnectionError(GesimError):
    """Raised when a gesim server cannot be reached."""


class ServerError(GesimError):
    """Raised when a gesim server returns an error response."""

    def __init__(self, status: int, message: str):
        super().__init__(f"server returned {status}: {message}")
        self.status = status
        self.message = message


def raise_for_status(resp) -> None:
    """Map an HTTP error response (status >= 400) to ServerError."""
    if resp.status_code < 400:
        return
    try:
        message = resp.json().get("error", resp.text)
    except ValueError:
        message = resp.text
    raise ServerError(resp.status_code, message)
