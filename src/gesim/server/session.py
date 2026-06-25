"""Per-episode session bookkeeping for the world-model server."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class Session:
    """One connected client. The server hosts at most one at a time."""

    user_name: str
    client_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    task: str = ""
