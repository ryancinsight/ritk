"""Bind CI script tests to the pinned Atlas process runtime."""

from __future__ import annotations

import os
from pathlib import Path
import re
import unittest


ROOT = Path(os.environ.get("RITK_CI_CONTRACT_ROOT", Path(__file__).resolve().parents[2]))
ATLAS_SOURCE = "42f451bf117af8d1549460c8a04cbc28bba065db"
CHECKOUT_ACTION = "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803"
SCRIPT_TEST_JOBS = (
    ("python_ci.yml", "build-and-smoke-test", "Run Python script tests"),
    ("metis-browser-dicom.yml", "build-browser", "Validate RITK browser tooling"),
)


class PythonRuntimeWorkflowTests(unittest.TestCase):
    """Keep the external process runtime explicit and immutable in each job."""

    def job(self, path: str, name: str) -> str:
        workflow = (ROOT / ".github/workflows" / path).read_text(encoding="utf-8")
        match = re.search(
            rf"(?ms)^  {re.escape(name)}:\n(.*?)(?=^  [\w-]+:\n|\Z)",
            workflow,
        )
        self.assertIsNotNone(match, f"{path} is missing job {name}")
        return match.group(1)

    def step(self, job: str, name: str) -> re.Match[str]:
        matches = list(re.finditer(
            rf"(?ms)^      - name: {re.escape(name)}\n.*?(?=^      - |\Z)",
            job,
        ))
        self.assertEqual(len(matches), 1, f"expected one step named {name!r}")
        return matches[0]

    def test_jobs_check_out_pinned_atlas_runtime_before_discovery(self) -> None:
        for path, job_name, test_name in SCRIPT_TEST_JOBS:
            with self.subTest(workflow=path):
                job = self.job(path, job_name)
                checkout = self.step(job, "Check out Atlas process runtime")
                tests = self.step(job, test_name)
                self.assertLess(checkout.start(), tests.start())
                lines = checkout.group().splitlines()
                for required in (
                    f"        uses: {CHECKOUT_ACTION}",
                    "          repository: ryancinsight/atlas",
                    f"          ref: {ATLAS_SOURCE}",
                    "          path: atlas",
                    "          persist-credentials: false",
                ):
                    self.assertIn(required, lines)
                if job_name == "build-browser":
                    self.assertNotRegex(checkout.group(), r"(?m)^        if:")

    def test_script_discovery_receives_the_checked_out_runtime(self) -> None:
        for path, job_name, test_name in SCRIPT_TEST_JOBS:
            with self.subTest(workflow=path):
                tests = self.step(self.job(path, job_name), test_name).group()
                self.assertRegex(
                    tests,
                    r"(?m)^        env:\n(?:^          [^\n]*\n)*"
                    r"^          RITK_MANUAL_ATLAS_ROOT: \$\{\{ github\.workspace \}\}/atlas$",
                )
                self.assertIn("python -m unittest discover -s scripts/tests", tests)


if __name__ == "__main__":
    unittest.main()
