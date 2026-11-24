"""Type stubs for the ``_ritk.io`` submodule (image and transform I/O)."""

from __future__ import annotations

from typing import Any, TypedDict

from ritk._ritk.image import Image

def read_image(path: str, series_instance_uid: str | None = None) -> Image: ...
def write_image(image: Image, path: str) -> None: ...
def read_transform(path: str) -> dict[str, Any]: ...
def write_transform(
    path: str,
    dimensionality: int,
    transforms: list[dict[str, Any]],
    description: str = "",
) -> None: ...

# -- DICOM anonymization ------------------------------------------------------

class _AnonymizeStats(TypedDict):
    file_count: int
    success_count: int
    error_count: int
    errors: list[list[str]]

def anonymize_dicom_dir(
    input_dir: str,
    output_dir: str,
    profile: str = "basic",
    patient_name: str = "ANONYMOUS",
    patient_id: str = "ANON001",
    uid_salt: str = "ritk-anon-salt",
    cleaning: str = "none",
) -> _AnonymizeStats:
    """Export immediate DICOM instances and report DICOMDIR as a per-file error."""
    ...

# -- Polygonal mesh I/O -------------------------------------------------------

def read_mesh(path: str) -> Any: ...
def write_mesh(path: str, mesh: Any) -> None: ...
