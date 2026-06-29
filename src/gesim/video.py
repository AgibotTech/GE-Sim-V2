"""MP4 writing for (T, 3, V, H, W) rollout videos."""

import logging
import shutil
import subprocess

import cv2
import numpy as np

logger = logging.getLogger("gesim.video")


def _resolve_ffmpeg() -> str | None:
    """Return an ffmpeg binary, preferring PATH then optional imageio-ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    return imageio_ffmpeg.get_ffmpeg_exe()


def save_video(frames: np.ndarray, path: str, fps: int = 16):
    """Save ``(T, 3, V, H, W)`` float video to mp4. Views are tiled horizontally."""
    T, C, V, H, W = frames.shape
    f = frames.astype(np.float32)
    if float(f.min()) < -0.05 and float(f.max()) <= 1.0 + 1e-3:
        f = (f + 1.0) * 0.5
    f = np.clip(f, 0.0, 1.0)
    rows = np.concatenate([f[:, :, v, :, :] for v in range(V)], axis=3)
    rows = (rows.transpose(0, 2, 3, 1) * 255.0).clip(0, 255).astype(np.uint8)
    rows = np.ascontiguousarray(rows)

    height, width = rows.shape[1], rows.shape[2]
    ffmpeg = _resolve_ffmpeg()
    if ffmpeg:
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-crf",
            "18",
            path,
        ]
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        try:
            assert proc.stdin is not None
            for frame in rows:
                proc.stdin.write(frame.tobytes())
            proc.stdin.close()
            ret = proc.wait()
        except BrokenPipeError:
            ret = proc.wait()
        finally:
            if proc.stdin is not None and not proc.stdin.closed:
                proc.stdin.close()
        if ret == 0:
            return
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        logger.warning("ffmpeg failed (exit %s): %s — falling back to OpenCV", ret, stderr.strip())
        # Fall through to OpenCV on failure.

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {path}")
    try:
        for frame in rows:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
