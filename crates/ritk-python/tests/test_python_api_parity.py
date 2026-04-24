from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PYTHON_CRATE = ROOT / "crates" / "ritk-python"
SRC_DIR = PYTHON_CRATE / "src"
PYTHON_DIR = PYTHON_CRATE / "python" / "ritk"
STUB_DIR = PYTHON_DIR / "_ritk"
TOP_LEVEL_INIT = PYTHON_DIR / "__init__.py"
TOP_LEVEL_STUB = PYTHON_DIR / "__init__.pyi"
SMOKE_TEST = PYTHON_CRATE / "tests" / "test_smoke.py"

MODULES = {
    "filter": {
        "rust": SRC_DIR / "filter.rs",
        "stub": STUB_DIR / "filter.pyi",
        "smoke_test": "test_filter_public_functions_exist",
    },
    "io": {
        "rust": SRC_DIR / "io.rs",
        "stub": STUB_DIR / "io.pyi",
        "smoke_test": "test_io_public_functions_exist",
    },
    "registration": {
        "rust": SRC_DIR / "registration.rs",
        "stub": STUB_DIR / "registration.pyi",
        "smoke_test": "test_registration_public_functions_exist",
    },
    "segmentation": {
        "rust": SRC_DIR / "segmentation.rs",
        "stub": STUB_DIR / "segmentation.pyi",
        "smoke_test": "test_segmentation_public_functions_exist",
    },
    "statistics": {
        "rust": SRC_DIR / "statistics.rs",
        "stub": STUB_DIR / "statistics.pyi",
        "smoke_test": "test_statistics_public_functions_exist",
    },
}

WRAP_PATTERN = re.compile(r"wrap_pyfunction!\((?P<name>[A-Za-z0-9_]+),\s*&m\)")
DEF_PATTERN = re.compile(r"^def\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)


def parse_registered_functions(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    marker = "pub fn register"
    start = text.index(marker)
    register_text = text[start:]
    return WRAP_PATTERN.findall(register_text)


def parse_stub_functions(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return set(DEF_PATTERN.findall(text))


def parse_top_level_reexports(path: Path) -> set[str]:
    module = ast.parse(path.read_text(encoding="utf-8"))
    exported: set[str] = set()
    for node in module.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                exported.add(alias.asname or alias.name)
        elif isinstance(node, ast.Assign):
            # Handle: Name = <expr> (e.g. Image = _image_mod.Image)
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    exported.add(target.id)
    return exported


def parse_top_level_stub_reexports(path: Path) -> set[str]:
    module = ast.parse(path.read_text(encoding="utf-8"))
    exported: set[str] = set()
    for node in module.body:
        if isinstance(node, ast.ImportFrom) and node.module != "__future__":
            for alias in node.names:
                exported_name = alias.asname or alias.name
                if not exported_name.startswith("_"):
                    exported.add(exported_name)
    return exported


def parse_top_level_all(path: Path) -> list[str]:
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if not isinstance(node.value, ast.List):
                        raise AssertionError("__all__ must be a list literal")
                    return [
                        elt.value
                        for elt in node.value.elts
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                    ]
    raise AssertionError("__all__ list not found")


def parse_top_level_version(path: Path) -> str:
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__version__":
                    if not isinstance(node.value, ast.Constant) or not isinstance(
                        node.value.value, str
                    ):
                        raise AssertionError("__version__ must be a string literal")
                    return node.value.value
    raise AssertionError("__version__ assignment not found")


def parse_smoke_required_functions(path: Path, test_name: str) -> set[str]:
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == test_name:
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name) and target.id == "required":
                            if not isinstance(stmt.value, ast.List):
                                raise AssertionError(
                                    f"required must be a list literal in {test_name}"
                                )
                            return {
                                elt.value
                                for elt in stmt.value.elts
                                if isinstance(elt, ast.Constant)
                                and isinstance(elt.value, str)
                            }
    raise AssertionError(f"required list not found in {test_name}")


def test_registered_functions_have_stub_and_smoke_coverage() -> None:
    for module_name, paths in MODULES.items():
        registered = parse_registered_functions(paths["rust"])
        stubbed = parse_stub_functions(paths["stub"])
        smoke_required = parse_smoke_required_functions(SMOKE_TEST, paths["smoke_test"])

        missing_stubs = sorted(set(registered) - stubbed)
        missing_smoke = sorted(set(registered) - smoke_required)
        duplicate_registered = sorted(
            {name for name in registered if registered.count(name) > 1}
        )

        assert not duplicate_registered, (
            f"{module_name}: duplicate wrap_pyfunction registrations: {duplicate_registered}"
        )
        assert not missing_stubs, (
            f"{module_name}: registered functions missing from stub: {missing_stubs}"
        )
        assert not missing_smoke, (
            f"{module_name}: registered functions missing from smoke test required list: {missing_smoke}"
        )


def test_top_level_package_exports_and_metadata_are_consistent() -> None:
    top_level_exports = parse_top_level_reexports(TOP_LEVEL_INIT)
    top_level_stub_exports = parse_top_level_stub_reexports(TOP_LEVEL_STUB)
    top_level_all = parse_top_level_all(TOP_LEVEL_INIT)
    top_level_version = parse_top_level_version(TOP_LEVEL_INIT)

    expected_exports = {
        "Image",
        "io",
        "filter",
        "registration",
        "segmentation",
        "statistics",
    }
    expected_all = [
        "Image",
        "io",
        "filter",
        "registration",
        "segmentation",
        "statistics",
    ]

    missing_runtime_exports = sorted(expected_exports - top_level_exports)
    missing_stub_exports = sorted(expected_exports - top_level_stub_exports)
    extra_all_exports = sorted(set(top_level_all) - expected_exports)
    missing_all_exports = sorted(expected_exports - set(top_level_all))

    assert not missing_runtime_exports, (
        f"top-level package missing expected re-exports: {missing_runtime_exports}"
    )
    assert not missing_stub_exports, (
        f"top-level stub missing expected re-exports: {missing_stub_exports}"
    )
    assert top_level_all == expected_all, (
        f"top-level __all__ drifted from expected export order: {top_level_all}"
    )
    assert not extra_all_exports, (
        f"top-level __all__ has unexpected exports: {extra_all_exports}"
    )
    assert not missing_all_exports, (
        f"top-level __all__ missing expected exports: {missing_all_exports}"
    )
    assert top_level_version, "top-level __version__ must not be empty"
