"""Binary wire-format helpers shared by the client transport and the server.

A multi-view frame ``(3, V, H, W)`` float32 in ``[0, 1]`` travels as a single
JPEG: the V views are tiled horizontally into one ``(H, V*W, 3)`` uint8 image.
Request/response sections are length-prefixed with a 4-byte big-endian size.
"""

import io

import numpy as np
from PIL import Image

JPEG_QUALITY = 95


def encode_frame_jpeg(frame: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    """Encode one ``(3, V, H, W)`` float32 ``[0, 1]`` frame as a tiled JPEG."""
    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim != 4 or frame.shape[0] != 3:
        raise ValueError(f"expected frame (3, V, H, W), got {frame.shape}")
    views = np.transpose(frame, (1, 2, 3, 0))  # (V, H, W, 3)
    tiles = [(np.clip(view, 0.0, 1.0) * 255.0).round().astype(np.uint8) for view in views]
    tiled = np.concatenate(tiles, axis=1)  # (H, V*W, 3)
    buf = io.BytesIO()
    Image.fromarray(tiled, mode="RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def decode_frame_jpeg(data: bytes, shape) -> np.ndarray:
    """Decode a tiled JPEG back to ``shape`` = ``(3, V, H, W)`` float32 ``[0, 1]``."""
    _, num_views, height, width = shape
    arr = np.asarray(Image.open(io.BytesIO(data)))
    expected = (height, num_views * width, 3)
    if arr.shape != expected:
        raise ValueError(
            f"decoded JPEG has shape {arr.shape}, expected {expected} "
            f"for frame shape {tuple(shape)}"
        )
    views = np.stack(np.split(arr, num_views, axis=1), axis=0)  # (V, H, W, 3)
    return np.transpose(views, (3, 0, 1, 2)).astype(np.float32) / 255.0


def pack_block(data: bytes) -> bytes:
    """Length-prefix a section with a 4-byte big-endian size."""
    return len(data).to_bytes(4, "big") + data


class BlockReader:
    """Sequential reader for length-prefixed sections of a binary body."""

    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    def read_block(self) -> bytes:
        if self._pos + 4 > len(self._data):
            raise ValueError("truncated binary body: missing length prefix")
        size = int.from_bytes(self._data[self._pos : self._pos + 4], "big")
        self._pos += 4
        if self._pos + size > len(self._data):
            raise ValueError("truncated binary body: section shorter than declared")
        block = self._data[self._pos : self._pos + size]
        self._pos += size
        return block
