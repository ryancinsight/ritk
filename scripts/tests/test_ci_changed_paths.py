"""Exercise the path predicates used by both pull-request workflows."""
from pathlib import Path
import subprocess
import unittest


WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"


class ChangedPathsTests(unittest.TestCase):
    def test_large_diff_and_docs_only_diff_select_expected_jobs(self):
        for name in ("ci.yml", "python_ci.yml"):
            workflow = (WORKFLOWS / name).read_text()
            block = workflow.split('changed="$(git diff --name-only "$base" HEAD)"', 1)[1]
            predicates = block.split("          fi\n          #", 1)[0]
            lines = [line.strip() for line in predicates.splitlines()
                     if line.strip().startswith("if ! ") and '"$changed"' in line]
            self.assertEqual(len(lines), 2)
            for line in lines:
                for first_path, expected in (("Cargo.lock", "yes"), ("README.md", "no")):
                    with self.subTest(workflow=name, predicate=line, first=first_path):
                        paths = "\n".join([first_path, *(f"test_data/study/{i:05}.dcm"
                                                       for i in range(16000))])
                        command = ('set -euo pipefail\nchanged="$(cat)"\n' + line +
                                   '\n echo no\nelse\n echo yes\nfi')
                        result = subprocess.run(["bash", "-c", command], input=paths,
                                                capture_output=True, text=True, check=True)
                        self.assertEqual(result.stdout.strip(), expected)
