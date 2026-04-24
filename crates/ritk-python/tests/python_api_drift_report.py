from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
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
ASSIGN_PATTERN = re.compile(
    r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>.+)$", re.MULTILINE
)


@dataclass(frozen=True)
class ModuleDrift:
    module_name: str
    registered: list[str]
    stubbed: set[str]
    smoke_required: set[str]
    duplicate_registered: list[str]
    missing_stubs: list[str]
    missing_smoke: list[str]
    extra_stubs: list[str]
    extra_smoke: list[str]

    @property
    def is_clean(self) -> bool:
        return not (
            self.duplicate_registered
            or self.missing_stubs
            or self.missing_smoke
            or self.extra_stubs
            or self.extra_smoke
        )


@dataclass(frozen=True)
class TopLevelDrift:
    runtime_exports: set[str]
    stub_exports: set[str]
    all_exports: list[str]
    version: str
    missing_runtime_exports: list[str]
    missing_stub_exports: list[str]
    extra_runtime_exports: list[str]
    extra_stub_exports: list[str]
    missing_all_exports: list[str]
    extra_all_exports: list[str]
    all_order_matches: bool

    @property
    def is_clean(self) -> bool:
        return not (
            self.missing_runtime_exports
            or self.missing_stub_exports
            or self.extra_runtime_exports
            or self.extra_stub_exports
            or self.missing_all_exports
            or self.extra_all_exports
            or not self.all_order_matches
            or not self.version
        )


def parse_registered_functions(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    marker = "pub fn register"
    start = text.index(marker)
    register_text = text[start:]
    return WRAP_PATTERN.findall(register_text)


def parse_stub_functions(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return set(DEF_PATTERN.findall(text))


def parse_smoke_required_functions(path: Path, test_name: str) -> set[str]:
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == test_name:
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name) and target.id == "required":
                            if not isinstance(stmt.value, ast.List):
                                raise ValueError(
                                    f"required must be a list literal in {test_name}"
                                )
                            return {
                                elt.value
                                for elt in stmt.value.elts
                                if isinstance(elt, ast.Constant)
                                and isinstance(elt.value, str)
                            }
    raise ValueError(f"required list not found in {test_name}")


def parse_top_level_reexports(path: Path) -> set[str]:
    module = ast.parse(path.read_text(encoding="utf-8"))
    exported: set[str] = set()
    for node in module.body:
        if isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            for alias in node.names:
                exported_name = alias.asname or alias.name
                if exported_name.startswith("_"):
                    continue
                exported.add(exported_name)
    return exported


def parse_top_level_stub_reexports(path: Path) -> set[str]:
    module = ast.parse(path.read_text(encoding="utf-8"))
    exported: set[str] = set()
    for node in module.body:
        if isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            for alias in node.names:
                exported_name = alias.asname or alias.name
                if exported_name.startswith("_"):
                    continue
                exported.add(exported_name)
    return exported


def parse_top_level_all(path: Path) -> list[str]:
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if not isinstance(node.value, ast.List):
                        raise ValueError("__all__ must be a list literal")
                    return [
                        elt.value
                        for elt in node.value.elts
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                    ]
    raise ValueError("__all__ list not found")


def parse_top_level_version(path: Path) -> str:
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__version__":
                    if not isinstance(node.value, ast.Constant) or not isinstance(
                        node.value.value, str
                    ):
                        raise ValueError("__version__ must be a string literal")
                    return node.value.value
    raise ValueError("__version__ assignment not found")


def compute_module_drift(module_name: str, paths: dict[str, Path]) -> ModuleDrift:
    registered = parse_registered_functions(paths["rust"])
    stubbed = parse_stub_functions(paths["stub"])
    smoke_required = parse_smoke_required_functions(SMOKE_TEST, paths["smoke_test"])

    registered_set = set(registered)
    duplicate_registered = sorted(
        {name for name in registered if registered.count(name) > 1}
    )
    missing_stubs = sorted(registered_set - stubbed)
    missing_smoke = sorted(registered_set - smoke_required)
    extra_stubs = sorted(stubbed - registered_set)
    extra_smoke = sorted(smoke_required - registered_set)

    return ModuleDrift(
        module_name=module_name,
        registered=registered,
        stubbed=stubbed,
        smoke_required=smoke_required,
        duplicate_registered=duplicate_registered,
        missing_stubs=missing_stubs,
        missing_smoke=missing_smoke,
        extra_stubs=extra_stubs,
        extra_smoke=extra_smoke,
    )


def compute_top_level_drift() -> TopLevelDrift:
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
    allowed_stub_only_exports = {"image"}

    runtime_exports = parse_top_level_reexports(TOP_LEVEL_INIT)
    stub_exports = parse_top_level_stub_reexports(TOP_LEVEL_STUB)
    all_exports = parse_top_level_all(TOP_LEVEL_INIT)
    version = parse_top_level_version(TOP_LEVEL_INIT)

    return TopLevelDrift(
        runtime_exports=runtime_exports,
        stub_exports=stub_exports,
        all_exports=all_exports,
        version=version,
        missing_runtime_exports=sorted(expected_exports - runtime_exports),
        missing_stub_exports=sorted(expected_exports - stub_exports),
        extra_runtime_exports=sorted(runtime_exports - expected_exports),
        extra_stub_exports=sorted(
            stub_exports - expected_exports - allowed_stub_only_exports
        ),
        missing_all_exports=sorted(expected_exports - set(all_exports)),
        extra_all_exports=sorted(set(all_exports) - expected_exports),
        all_order_matches=all_exports == expected_all,
    )


def format_list(items: list[str]) -> str:
    return ", ".join(items) if items else "none"


def print_module_report(drift: ModuleDrift) -> None:
    print(f"[module:{drift.module_name}]")
    print(f"  registered count : {len(drift.registered)}")
    print(f"  stubbed count    : {len(drift.stubbed)}")
    print(f"  smoke count      : {len(drift.smoke_required)}")
    print(f"  duplicate regs   : {format_list(drift.duplicate_registered)}")
    print(f"  missing stubs    : {format_list(drift.missing_stubs)}")
    print(f"  missing smoke    : {format_list(drift.missing_smoke)}")
    print(f"  extra stubs      : {format_list(drift.extra_stubs)}")
    print(f"  extra smoke      : {format_list(drift.extra_smoke)}")
    print(f"  status           : {'clean' if drift.is_clean else 'drift detected'}")
    print()


def print_top_level_report(drift: TopLevelDrift) -> None:
    expected_all = [
        "Image",
        "io",
        "filter",
        "registration",
        "segmentation",
        "statistics",
    ]
    allowed_stub_only_exports = ["image"]

    print("[top-level:ritk]")
    print(f"  runtime exports  : {format_list(sorted(drift.runtime_exports))}")
    print(f"  stub exports     : {format_list(sorted(drift.stub_exports))}")
    print(f"  __all__          : {format_list(drift.all_exports)}")
    print(f"  expected __all__ : {format_list(expected_all)}")
    print(f"  stub-only allow  : {format_list(allowed_stub_only_exports)}")
    print(f"  __version__      : {drift.version or '<empty>'}")
    print(f"  missing runtime  : {format_list(drift.missing_runtime_exports)}")
    print(f"  missing stub     : {format_list(drift.missing_stub_exports)}")
    print(f"  extra runtime    : {format_list(drift.extra_runtime_exports)}")
    print(f"  extra stub       : {format_list(drift.extra_stub_exports)}")
    print(f"  missing __all__  : {format_list(drift.missing_all_exports)}")
    print(f"  extra __all__    : {format_list(drift.extra_all_exports)}")
    print(
        f"  __all__ order    : {'matches expected order' if drift.all_order_matches else 'drift detected'}"
    )
    print(f"  status           : {'clean' if drift.is_clean else 'drift detected'}")
    print()


def main() -> int:
    print("RITK Python API Drift Report")
    print("============================")
    print(f"root: {ROOT}")
    print()

    module_drifts = [
        compute_module_drift(module_name, paths)
        for module_name, paths in MODULES.items()
    ]
    top_level_drift = compute_top_level_drift()

    for drift in module_drifts:
        print_module_report(drift)

    print_top_level_report(top_level_drift)

    has_drift = (
        any(not drift.is_clean for drift in module_drifts)
        or not top_level_drift.is_clean
    )

    print("Summary")
    print("-------")
    print(f"modules checked    : {len(module_drifts)}")
    print(
        f"modules with drift : {sum(1 for drift in module_drifts if not drift.is_clean)}"
    )
    print(f"top-level clean    : {'yes' if top_level_drift.is_clean else 'no'}")
    print(f"overall status     : {'clean' if not has_drift else 'drift detected'}")

    return 1 if has_drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
