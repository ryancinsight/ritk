"""Exercise the committed manual-capture command without fabricating a capture."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import tomllib
import unittest

from scripts import capture_manual_images, capture_source


def atlas_root() -> Path:
    value = os.environ.get("RITK_MANUAL_ATLAS_ROOT")
    if value is None:
        raise RuntimeError("RITK_MANUAL_ATLAS_ROOT must name the Atlas checkout")
    return Path(value)


class CaptureManualImagesTests(unittest.TestCase):
    def test_command_selects_exact_native_capture(self) -> None:
        config = Path("neutral") / "config.toml"
        command = capture_manual_images.capture_command(config, Path("source"))
        self.assertEqual(command[:4], ["cargo", "--config", str(config), "nextest"])
        self.assertEqual(command.count("--locked"), 1)
        self.assertEqual(command[command.index("--features") + 1], "eframe-shell")
        self.assertEqual(
            command[command.index("-E") + 1], "test(native_manual_annotation_frames)"
        )
        self.assertEqual(command[-2:], ["--profile", "default"])

    def test_neutral_config_retains_profiles_without_patches(self) -> None:
        root = atlas_root()
        builder = capture_source.atlas_build_source(root)
        config = builder.cargo_config_without_patches(root / ".cargo/config.toml")
        parsed = tomllib.loads(config)
        self.assertNotIn("patch", parsed)
        self.assertEqual(parsed["build"]["target-dir"], "target")
        self.assertEqual(parsed["profile"]["dev"]["debug"], "line-tables-only")

    def test_existing_capture_root_is_rejected_without_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            repository = root / "repository"
            capture = root / "capture"
            for path in (source, repository / ".git", capture):
                path.mkdir(parents=True)
            (source / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
            record = root / "capture.json"
            arguments = SimpleNamespace(
                atlas_root=atlas_root(), source_root=source, repository=repository,
                capture_root=capture, capture_record=record,
            )
            with self.assertRaisesRegex(
                capture_manual_images.CaptureCommandError, "capture root already exists"
            ):
                capture_manual_images.run_capture(arguments)
            self.assertFalse(record.exists())

    def test_isolated_help_describes_record_output(self) -> None:
        root = atlas_root()
        runner = capture_source.atlas_process_tree(root)
        script = Path(capture_manual_images.__file__).resolve()
        result = runner.run([sys.executable, "-I", str(script), "--help"], timeout=30)
        self.assertEqual(result.returncode, 0)
        help_text = result.stdout.decode("utf-8")
        for option in ("--atlas-root", "--source-root", "--repository",
                       "--capture-root", "--capture-record"):
            with self.subTest(option=option):
                self.assertIn(option, help_text)


if __name__ == "__main__":
    unittest.main()
