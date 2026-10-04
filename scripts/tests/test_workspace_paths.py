"""Tests for resolving first-party tools across supported repository layouts."""

import tempfile
import unittest
from pathlib import Path

from workspace_paths import metis_scripts


class WorkspacePathsTests(unittest.TestCase):
    def test_resolves_metis_inside_atlas_repositories(self):
        with tempfile.TemporaryDirectory() as temporary:
            atlas = Path(temporary) / "atlas"
            source = atlas / "repos" / "ritk" / "scripts" / "tests" / "case.py"
            metis = atlas / "repos" / "metis" / "scripts"
            source.parent.mkdir(parents=True)
            metis.mkdir(parents=True)
            source.touch()
            (metis / "browser_canvas.py").touch()

            self.assertEqual(metis_scripts(str(source)), metis.resolve())

    def test_resolves_sibling_repository_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            projects = Path(temporary) / "projects"
            source = projects / "ritk" / "scripts" / "tests" / "case.py"
            metis = projects / "metis" / "scripts"
            source.parent.mkdir(parents=True)
            metis.mkdir(parents=True)
            source.touch()
            (metis / "browser_canvas.py").touch()

            self.assertEqual(metis_scripts(str(source)), metis.resolve())

    def test_rejects_unlinked_repository_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "ritk" / "scripts" / "tests" / "case.py"
            source.parent.mkdir(parents=True)
            source.touch()

            with self.assertRaises(FileNotFoundError):
                metis_scripts(str(source))


if __name__ == "__main__":
    unittest.main()
