# Phase 4 completion report

Status: complete. The environment runtime and verifier are ready for Phase 5 and
Phase 6 consumers.

| Requirement | Evidence |
| --- | --- |
| Isolated executable environments and exact verifier reward | Focused Phase 4 suite: 108 passed |
| Worker crash, timeout, concurrent routing, and compile-once contracts | `tests/test_pool_timeout.py`, `tests/test_worker_crash_blast_radius.py`, and `tests/test_worker_isolation_secrets.py` |
| Pinned executable inputs | EnvScaler submodule at `87e667397abacf274858c0964796beb8f984aafe`; metadata hashes match `configs/base/grpo.yaml` |
| End-to-end runtime | `smolqwen env-selftest --profile l4`: initial 0.0, partial 0.3333, full 1.0 |
| Repository verification | 203 tests passed; Ruff, mypy, and `uv build` passed |

Known warning: one `DeprecationWarning` originates in an executed released
verifier fixture (`invalid escape sequence`); no project code warning or failure
remains.

Next: Phase 5 evaluation harness, which consumes the Phase 4 environment runtime
for held-out reward evaluation.
