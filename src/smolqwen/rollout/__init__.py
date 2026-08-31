"""Phase 6 rollout: async ready-queue scheduling behind TRL's `rollout_func`.

Public surface for the two rollout paths:

- `factory_env` — the `environment_factory` oracle, run inside its own trainer.
- `rollout_func` — the production path. `rollout_func` with `tools=None` is the
  only combination under which TRL reads our `env_mask`, so the two paths are
  two separately constructed trainers, enforced by `test_trainer_kwargs_exclusive`.
"""
