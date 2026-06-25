import numpy as np
import pytest

from gesim.client.codecs import (
    BlockReader,
    decode_frame_jpeg,
    encode_frame_jpeg,
    pack_block,
)


def test_jpeg_roundtrip_shape_and_range():
    rng = np.random.default_rng(0)
    frame = rng.random((3, 3, 64, 96), dtype=np.float32)
    data = encode_frame_jpeg(frame)
    out = decode_frame_jpeg(data, frame.shape)
    assert out.shape == frame.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    # JPEG is lossy but should stay close on smooth content.
    flat = np.zeros((3, 3, 64, 96), dtype=np.float32) + 0.5
    out2 = decode_frame_jpeg(encode_frame_jpeg(flat), flat.shape)
    assert np.abs(out2 - 0.5).max() < 0.05


def test_encode_rejects_bad_shape():
    with pytest.raises(ValueError):
        encode_frame_jpeg(np.zeros((4, 3, 8, 8), dtype=np.float32))


def test_block_roundtrip():
    body = pack_block(b"hello") + pack_block(b"") + pack_block(b"world")
    reader = BlockReader(body)
    assert reader.read_block() == b"hello"
    assert reader.read_block() == b""
    assert reader.read_block() == b"world"


def test_block_reader_truncated():
    with pytest.raises(ValueError):
        BlockReader(b"\x00\x00\x00\x05ab").read_block()


def test_jpeg_roundtrip_preserves_view_identity():
    frame = np.zeros((3, 3, 32, 48), dtype=np.float32)
    frame[0, 0] = 1.0  # view 0: red
    frame[2, 2] = 1.0  # view 2: blue
    out = decode_frame_jpeg(encode_frame_jpeg(frame), frame.shape)
    assert out[0, 0].mean() > 0.9 and out[1, 0].mean() < 0.1  # view 0 red
    assert out[2, 2].mean() > 0.9 and out[0, 2].mean() < 0.1  # view 2 blue


def test_encode_rejects_wrong_ndim():
    with pytest.raises(ValueError):
        encode_frame_jpeg(np.zeros((3, 8, 8), dtype=np.float32))


def test_decode_rejects_mismatched_shape():
    frame = np.zeros((3, 3, 32, 48), dtype=np.float32)
    data = encode_frame_jpeg(frame)
    with pytest.raises(ValueError):
        decode_frame_jpeg(data, (3, 2, 32, 48))


def test_block_reader_missing_prefix():
    with pytest.raises(ValueError):
        BlockReader(b"\x00\x00").read_block()
