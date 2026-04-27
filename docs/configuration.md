# Configuration

Shardon keeps desired state in YAML.

## Folder Pattern

- `config/backends-available` and `config/backends-enabled`
- `config/models-available` and `config/models-enabled`
- `config/deployments-available` and `config/deployments-enabled`
- `config/gpu-inventory-available` and `config/gpu-inventory-enabled`
- `config/gpu-groups-available` and `config/gpu-groups-enabled`
- `config/auth/admins-available` and `config/auth/admins-enabled`

The `enabled/` folders are intended to be symlink-friendly, mirroring the familiar Nginx pattern.
Shardon now treats symlinks as the default operational model: enabled entries should point at the canonical file in the matching `available/` directory.

## Important Fields

- Backends define runtime folder, launch command, capabilities, type, version, and health endpoint.
- Backend `capabilities` should explicitly include modality metadata (`modalities`) and operation flags (for example `audio_speech`, `audio_transcriptions`, `audio_translations`) so routing can filter candidates correctly.
- Models define logical identity, source, tokenizer, display metadata, and backend compatibility.
- Models can also define `model_capabilities` (for example `text`, `audio`, `image`, `video`) for modality-aware routing and visibility in `/v1/models`.
- Models can define per-model runtime launch flags:
  - `runtime_launch_args`: appended for every backend launch of that model.
  - `runtime_launch_args_by_backend_type`: appended only for matching backend types (for example `vllm`).
  Example:
  ```yaml
  runtime_launch_args_by_backend_type:
    vllm:
      - --enable-auto-tool-choice
      - --tool-call-parser
      - qwen3_coder
      - --reasoning-parser
      - qwen3
      - --enable-prefix-caching
      - --quantization
      - compressed-tensors
      - --speculative-config={"method":"mtp","num_speculative_tokens":3}
  ```
- Deployments bind model plus backend runtime plus API-visible alias and memory budget.
- Deployments can optionally define `deployment_capabilities` to narrow a model/backend pair to a subset of modalities for that deployment.
- Deployments can declare one group (`gpu_group_id`) or an ordered list (`gpu_group_ids`); Shardon chooses one concrete group at load/start time.
- One logical model can still have multiple deployments when desired, but duplicate per-group deployments are no longer required for primary/fallback group placement.
- GPU inventory captures stable identity using UUID and PCI bus information instead of only transient indices.
- GPU groups define scheduling targets, usable memory fraction, and `keep_free`.
- Backends may use `gpu_group_overrides` for per-group launch command, base URL, environment, and health/readiness settings while keeping one backend definition.

## Environment Variables

- Use a repo-local `.env` file or normal process environment variables for secrets and machine-specific paths.
- `HF_TOKEN` is the intended place for Hugging Face download access.
- `HF_HOME` can be set to control the Hugging Face cache location.
- These values are intentionally kept out of the admin config and out of persisted runtime state.
