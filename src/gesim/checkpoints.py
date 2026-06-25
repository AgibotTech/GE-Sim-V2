"""Checkpoint reference resolution: local paths or ``hf://org/repo[/path]`` URIs."""

from __future__ import annotations

import os

from huggingface_hub import hf_hub_download, snapshot_download

HF_PREFIX = "hf://"


def resolve(ref: str) -> str:
    """Resolve a checkpoint reference to a local filesystem path.

    ``hf://org/repo`` downloads the whole repo snapshot (for directory-style
    artifacts such as diffusers model folders); ``hf://org/repo/path/to/file``
    downloads a single file. Plain paths are validated and returned unchanged.
    """
    if not ref:
        raise ValueError("empty checkpoint reference")
    if ref.startswith(HF_PREFIX):
        body = ref[len(HF_PREFIX) :]
        parts = body.split("/", 2)
        if len(parts) < 2:
            raise ValueError(f"invalid HF reference {ref!r}; expected hf://org/repo[/path]")
        repo_id = f"{parts[0]}/{parts[1]}"
        if len(parts) == 3:
            return hf_hub_download(repo_id, parts[2])
        return snapshot_download(repo_id)
    if not os.path.exists(ref):
        raise FileNotFoundError(
            f"checkpoint path does not exist: {ref}. Use a local path or an hf:// URI."
        )
    return ref
