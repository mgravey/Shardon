# Mock vLLM v2 Runtime

This folder is independently runnable for local debugging:

```bash
python3 launcher.py --port 19092 --runtime-label mock-vllm-v2
```

Sleep mode (mirrors vLLM `--enable-sleep-mode` with `VLLM_SERVER_DEV_MODE=1`):

```bash
python3 launcher.py --port 19092 --runtime-label mock-vllm-v2 --enable-sleep-mode
curl -X POST 'http://127.0.0.1:19092/sleep?level=1&mode=wait'
curl http://127.0.0.1:19092/is_sleeping
curl -X POST http://127.0.0.1:19092/wake_up
curl http://127.0.0.1:19092/mock/stats
```

Inference returns HTTP 503 while asleep. Fault injection: `MOCK_WAKE_FAIL=1`,
`MOCK_WAKE_DELAY_SECONDS`, `MOCK_SLEEP_DELAY_SECONDS`. Without `--port`, the
port is read from `IGF_SHARDON_RUNTIME_PORT`.
