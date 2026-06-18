#!/usr/bin/env python
"""Regenerate SITK_CMAKE_COVERAGE.md.

Cross-references ritk's SimpleITK differential parity coverage against the full
upstream filter set: the 298 ``Code/BasicFilters/yaml/*.yaml`` definitions that
drive SimpleITK's generated cmake tests. Queries the authoritative list via the
GitHub API (`gh`) and scans this directory's ``test_*.py`` for ``sitk.*`` calls.

Run from anywhere:  python crates/ritk-python/tests/_gen_sitk_coverage.py
"""
from __future__ import annotations

import glob
import os
import re
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))


def _upstream_filters() -> list[str]:
    out = subprocess.run(
        ["gh", "api",
         "repos/SimpleITK/SimpleITK/contents/Code/BasicFilters/yaml?ref=main",
         "--jq", ".[].name"],
        capture_output=True, text=True, check=True).stdout
    return sorted(n[:-5] for n in out.split() if n.endswith(".yaml"))


def _tested_names() -> set[str]:
    blob = ""
    for f in glob.glob(os.path.join(_HERE, "test_*.py")):
        blob += open(f, encoding="utf-8").read()
    return set(re.findall(r"sitk\.([A-Za-z0-9]+)", blob))


def _proc(cls: str) -> str:
    for suf in ("ImageFilter", "Filter"):
        if cls.endswith(suf):
            return cls[: -len(suf)]
    return cls


def main() -> None:
    sf = _upstream_filters()
    tested = _tested_names()

    def covered(cls: str) -> bool:
        p = _proc(cls)
        # Procedural source filters expose a sitk function dropping "Image"
        # (e.g. yaml `GaussianImageSource` → `sitk.GaussianSource`).
        src = cls.replace("ImageSource", "Source")
        return (
            p in tested
            or cls in tested
            or (p + "ImageFilter") in tested
            or src in tested
        )

    cov = [f for f in sf if covered(f)]
    unc = [f for f in sf if not covered(f)]
    out = [
        "# SimpleITK cmake-test coverage survey",
        "",
        "Authoritative cross-reference of ritk's SimpleITK parity coverage against the",
        "**full upstream filter set** — the 298 `Code/BasicFilters/yaml/*.yaml` definitions",
        "that drive SimpleITK's generated cmake tests",
        "(<https://github.com/SimpleITK/SimpleITK/tree/main/Code/BasicFilters/yaml>).",
        "Regenerate with `tests/_gen_sitk_coverage.py`.",
        "",
        f"**{len(cov)} / {len(sf)} covered** by a ritk differential parity test "
        f"({len(unc)} not yet covered).",
        "",
        f"## Covered ({len(cov)})",
        "",
        ", ".join(_proc(x) for x in cov),
        "",
        f"## Not yet covered ({len(unc)})",
        "",
        "Most are filter families not implemented in ritk (label-map algebra, complex-FFT",
        "component ops, level-set/demons registration variants, image sources, object-",
        "morphology, projection variants). A minority are implemented but compared against",
        "a non-sitk oracle (FFT vs numpy; noise filters are non-deterministic; ritk",
        "`Shrink` is a box-mean = sitk `BinShrink`, not sitk `Shrink`).",
        "",
        ", ".join(_proc(x) for x in unc),
        "",
    ]
    dst = os.path.join(_HERE, "SITK_CMAKE_COVERAGE.md")
    open(dst, "w", encoding="utf-8").write("\n".join(out))
    print(f"wrote {dst}: {len(cov)} covered / {len(unc)} uncovered of {len(sf)}")


if __name__ == "__main__":
    main()
