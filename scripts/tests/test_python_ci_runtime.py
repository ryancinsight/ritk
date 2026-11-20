"""Bind manual-capture tests to the pinned Atlas process runtime."""

from __future__ import annotations

import os
from pathlib import Path
import unittest


ROOT = Path(os.environ.get("RITK_CI_CONTRACT_ROOT", Path(__file__).resolve().parents[2]))
ATLAS_SOURCE = "42f451bf117af8d1549460c8a04cbc28bba065db"


class PythonRuntimeWorkflowTests(unittest.TestCase):
    """Keep the external process runtime explicit and immutable."""

    @classmethod
    def setUpClass(cls) -> None:
        workflow = (ROOT / ".github/workflows/python_ci.yml").read_text(encoding="utf-8")
        start = workflow.index("\n  build-and-smoke-test:\n")
        cls.job = workflow[start:]

    def test_job_checks_out_pinned_atlas_runtime(self) -> None:
        for required in (
            "repository: ryancinsight/atlas",
            f"ref: {ATLAS_SOURCE}",
            "path: atlas",
        ):
            with self.subTest(required=required):
                self.assertEqual(self.job.count(required), 1)

    def test_script_tests_receive_checked_out_runtime(self) -> None:
        checkout = self.job.index("- name: Check out Atlas process runtime")
        tests = self.job.index("- name: Run Python script tests")
        self.assertLess(checkout, tests)
        self.assertEqual(
            self.job.count("RITK_MANUAL_ATLAS_ROOT: ${{ github.workspace }}/atlas"),
            1,
        )


if __name__ == "__main__":
    unittest.main()
