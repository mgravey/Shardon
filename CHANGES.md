# Changes

## 2026-04-22

- Added multimodal routing scaffolding across scheduler, runtime, backend adapters, and status output.
- Added OpenAI-style audio endpoints on router API:
  - `POST /v1/audio/speech`
  - `POST /v1/audio/transcriptions`
  - `POST /v1/audio/translations`
- Added backend adapter support for multipart audio upload flows and binary speech responses.
- Added WhisperX adapter translation layer for OpenAI-compatible transcription/translation outputs.
- Added first-class modality metadata on models/deployments/backends (`text`, `audio`, `image`, `video`) and capability-aware candidate filtering.
- Added regression coverage for dependency binding and audio JSON/multipart routes, plus scheduler/config capability behavior.
