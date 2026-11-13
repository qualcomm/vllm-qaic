# Role: aot — 8 rules

Mined from this repo's human review history. Background in `repo_conventions.md`.

| # | Rule | Severity | Trigger |
|---|---|---|---|
| 1 | `aot.ccl_defaults` | BLOCKING | `comp_ctx_lengths_prefill` / `_decode` left empty instead of defaulting to `[max_model_len]` when CCL is disabled (`ccl_enabled` false/absent) |
| 2 | `aot.mdp_stages_consistency` | BLOCKING | `mdp_num_partitions` and `cfg['stages']` set independently with no assert that they agree |
| 3 | `aot.qpc_dict_keys` | BLOCKING | QPC path extracted from the compile-result dict without validating the expected keys exist |
| 4 | `aot.specialization_covers_shapes` | BLOCKING | New runtime shape/seq-len path with no matching compile specialization (`prefill_seq_len`, `vision_size`) |
| 5 | `aot.session_input_guard` | BLOCKING | Feeding a session input without checking `in self.session.input_names` |
| 6 | `disagg.device_group_validation` | BLOCKING | Prefill/decode/encode device groups unvalidated — empty group, or duplicate QIDs across groups |
| 7 | `disagg.proc_lifecycle` | non-blocking | New subprocess/resource-tracker with no account of the P/D process fan-out (6P4D → 10 vllm processes, each spawning trackers) |
| 8 | `spd.aot_only` | BLOCKING | Spec-decode path reachable without the AoT guard (the eager-mode `speculative_config` rejection in `QaicPlatform.check_and_update_config`) |
