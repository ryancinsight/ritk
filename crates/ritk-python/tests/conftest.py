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

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _preserve_simpleitk_thread_budget():
    """Restore process-global SimpleITK thread state after every test."""
    sitk = sys.modules.get("SimpleITK")
    if sitk is None:
        yield
        return

    original = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    try:
        yield
    finally:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(original)


_PYTHON_SOURCE = Path(__file__).resolve().parent.parent / "python"
itk_path = _PYTHON_SOURCE / "itk"
if itk_path.is_dir():
    # Load itk/__init__.py dynamically to avoid shadowing by site-packages itk
    spec = importlib.util.spec_from_file_location("itk", str(itk_path / "__init__.py"))
    if spec and spec.loader:
        itk_mod = importlib.util.module_from_spec(spec)
        sys.modules["itk"] = itk_mod
        spec.loader.exec_module(itk_mod)
        
        # Load itk/image_ops.py
        spec_ops = importlib.util.spec_from_file_location("itk.image_ops", str(itk_path / "image_ops.py"))
        if spec_ops and spec_ops.loader:
            ops_mod = importlib.util.module_from_spec(spec_ops)
            sys.modules["itk.image_ops"] = ops_mod
            spec_ops.loader.exec_module(ops_mod)
            setattr(itk_mod, "image_ops", ops_mod)

# Keep the original sys.path append for other modules if needed, but ensure it's at the end
_p = str(_PYTHON_SOURCE)
if _p not in sys.path:
    sys.path.append(_p)
