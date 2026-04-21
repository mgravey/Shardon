# Running Backends Outside Shardon

Each backend runtime folder should remain independently runnable.

## Why

- Direct debugging without the router in the middle
- Easy capability testing across versions
- Cleaner separation between Shardon orchestration and backend implementation

## Demo Examples

- `demo/runtimes/mock-vllm-v1/launcher.py`
- `demo/runtimes/mock-vllm-v2/launcher.py`
- `demo/runtimes/mock-sglang/launcher.py`

Each one can be launched directly with `python3 launcher.py --port <port> --runtime-label <label>`.
