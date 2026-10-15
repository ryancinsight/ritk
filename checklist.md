# Active target: Python script compatibility and cache budget [patch]

- [x] [patch] Reproduce the older-Python lock-parser failure on unchanged main and add regression coverage.
- [x] [patch] Preserve TOML 1.0 validation, selected-interpreter execution and UTF-8/LF gallery evidence bytes.
- [x] [patch] Pass all 41 script tests on Python 3.9, 3.10, 3.11 and 3.13 with unchanged matrix, gates and limits.
- [x] [patch] Measure the cancelled Windows job and remove optional branch cache writes using the pinned action's main-only save policy.
- [ ] [patch] Verify final-revision cache post-step timings and the complete Python matrix within the unchanged 30-minute bound in [RITK-PYTHON-CI-CACHE-SAVE-001](backlog/RITK-PYTHON-CI-CACHE-SAVE-001.md).
- [ ] [patch] Review the separate draft prerequisite and its exact-revision full hosted-matrix evidence in [RITK-PYTHON-SCRIPT-RUNTIME-001](backlog/RITK-PYTHON-SCRIPT-RUNTIME-001.md).
