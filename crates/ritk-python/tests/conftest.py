"""Pytest path bootstrap for ritk-python tests.

The maturin ``python-source`` directory (``crates/ritk-python/python``) holds the
pure-Python ``itk`` shim package, which is bundled into the wheel but is not on
``sys.path`` when tests run from the repository root.  It is *appended* (not
prepended) so the installed ``ritk`` wheel — which carries the compiled
``_ritk`` extension — keeps priority; prepending would let the source ``ritk``
package in the same directory shadow the wheel, breaking ``from ritk._ritk
import filter``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PYTHON_SOURCE = Path(__file__).resolve().parent.parent / "python"
if _PYTHON_SOURCE.is_dir():
    _p = str(_PYTHON_SOURCE)
    if _p not in sys.path:
        sys.path.append(_p)
