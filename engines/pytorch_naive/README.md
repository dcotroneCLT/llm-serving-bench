# E3 — PyTorch + HuggingFace naive serving

The unoptimized baseline (C1) of the WoSAR 2026 study. HuggingFace
`transformers` loaded directly, served via FastAPI/uvicorn, no
PagedAttention, no continuous batching, no static padded batching: one
request occupies the GPU at a time.

## Endpoint contract

This engine is the server side of `client/protocols/pytorch_hf.py`.

### `POST /generate`

Request body:

```json
{"prompt": "...", "max_tokens": 200, "stream": false}
```

Non-streaming response:

```json
{"text": "...", "prompt_tokens": 123, "completion_tokens": 200}
```

Streaming response (`text/event-stream`):

```
data: {"text": "Hello"}

data: {"text": " world"}

data: {"prompt_tokens": 123, "completion_tokens": 200}

data: [DONE]

```

The text frames come first, then the usage frame, then `[DONE]`. The
client adapter sets `first_token_at_unix` on the first non-empty `data:`
line, so the order is load-bearing.

### `GET /healthz`, `GET /readyz`

`/healthz` returns 200 as soon as uvicorn is up. `/readyz` returns 200
only after the model has finished loading; before that it returns 503.

## Truncation policy

```
MAX_BUDGET = MAX_MODEL_LEN - max_new_tokens
```

If the tokenized prompt is longer than `MAX_BUDGET`, it is left-truncated
(the tail of the prompt is kept). The server never returns 400 for an
oversized prompt; it always serves a request, and reports the post-
truncation `prompt_tokens` count back to the client.

## Build & run

The launcher builds the image on first run; subsequent runs reuse the
cached layers.

```bash
# Build only (optional, launch.sh will do it on demand)
docker build -t pytorch_naive:wosar2026 .

# Launch and wait for /readyz
./launch.sh /home/dcotrone/wosar/runs/<run_id>
```

After launch.sh returns, the file `<run_id>/engine.pid` contains the
host PID of uvicorn, suitable for `proc_monitor.py --pidfile`.

### Default knobs (override via env vars)

| Variable | Default | Notes |
|---|---|---|
| `IMAGE` | `pytorch_naive:wosar2026` | Image tag |
| `CONTAINER_NAME` | `pytorch_naive_e3` | |
| `GPU_INDEX` | `2` | Host GPU index per ADR / protocol |
| `PORT` | `8002` | Host port mapped to container :8000 |
| `HF_CACHE` | `$HOME/wosar/hf_cache` | Bind-mounted into the container |
| `MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct` | |
| `MAX_MODEL_LEN` | `8192` | |
| `DTYPE` | `bfloat16` | |
| `READYZ_TIMEOUT_S` | `600` | Cold loads of a 7B model can be slow |

## Integration with monitoring

```bash
# 1. Start the engine
./launch.sh ~/wosar/runs/primary_e3_replicate1

# 2. Start the three monitors against the same run dir
python ../../monitoring/run_monitors.py \
  --run-id primary_e3_replicate1 \
  --runs-root ~/wosar/runs \
  --gpu-index 2 \
  --pidfile ~/wosar/runs/primary_e3_replicate1/engine.pid \
  --duration-seconds 86400 \
  --label-engine pytorch_naive

# 3. Start the client (on a separate machine on the LAN)
python ../../client/run_client.py \
  --config ../../client/config.yaml \
  --output-dir ~/wosar/runs/primary_e3_replicate1/client \
  --duration-seconds 86400 \
  --protocol pytorch_hf \
  --base-url http://<server>:8002 \
  --model Qwen/Qwen2.5-7B-Instruct
```

## Why this design

The point of E3 is to be the worst-case reference inside the comparison.
Every aging-relevant choice is explicitly the unoptimized one:

- **Single-flight** (`asyncio.Lock`). All requests serialize on the GPU.
  The client's `queue_time_s` becomes a first-class aging signal: as the
  engine slows down per-request, the queue grows and arrivals beyond
  `concurrency_cap` are dropped.
- **Stock HF KV-cache**. Per-request `DynamicCache`, allocated and freed
  through PyTorch's caching allocator. No paged attention, no shared
  cache pool. Fragmentation over time is exactly what we want to
  observe.
- **No `empty_cache()` between requests**. We do not paper over the
  caching allocator's behavior. Whatever it does is part of the signal.
- **Sampling parameters fixed server-side** (`temperature=0.7`,
  `top_p=0.95`) to match the vLLM and Triton+vLLM adapters; the
  workload distribution is identical across E1/E2/E3.

The HF generate() call cannot be interrupted; on client disconnect the
GPU still pays the full generation time before the lock is released for
the next request. This is the honest behavior of a naive server and is
documented as such for threats-to-validity.
