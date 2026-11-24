"""Location of the cataloged research fixtures used by RITK tests."""
import os
from pathlib import Path


def catalog_fixture(relative_path: str) -> Path:
    cache = os.environ.get("ATLAS_FIXTURE_CACHE")
    root = Path(cache) if cache else Path(__file__).resolve().parents[3] / "test_data"
    return root / relative_path
