"""Exercise count, single-file and aggregate chooser bounds with real sparse files."""
from pathlib import Path


def invalid_studies(root: Path):
    # Independent oracles for the Métis host bounds, each exceeded by count or size.
    for name, count, size in (("Count", 513, 0), ("File", 1, (64 << 20) + 1), ("Batch", 5, 64 << 20)):
        directory = root / name
        directory.mkdir()
        files = []
        for index in range(1, count + 1):
            file = directory / f"{index:08}.dcm"
            with file.open("wb") as stream:
                stream.truncate(size)
            files.append(file)
        yield name, files
