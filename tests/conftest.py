from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_BUNDLE = REPO_ROOT / "assets" / "demo_000"


@pytest.fixture(scope="session")
def demo_bundle_dir() -> Path:
    if not DEMO_BUNDLE.is_dir():
        pytest.skip(f"demo bundle missing: {DEMO_BUNDLE}")
    return DEMO_BUNDLE
