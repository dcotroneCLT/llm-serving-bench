# Client

Async, open-loop benchmark client with concurrency cap.

## Files

- `run_client.py` entry point
- `benchmark.py` open-loop scheduler, drop-rate accounting, CSV logging
- `prompt_sampler.py` corpus-based prompt sampler with token-length matching
- `build_corpus.py` one-shot script to fetch arXiv abstracts as the prompt corpus
- `protocols/` adapters for the three engines (`vllm_openai`, `pytorch_hf`, `triton_vllm`)
- `_types.py` `RequestResult` dataclass and CSV schema
- `config.yaml` default configuration

## Prerequisites

```
pip install httpx pyyaml tiktoken
```

`tiktoken` is optional but strongly recommended for fast token counting.
Without it, the sampler falls back to a 4-chars-per-token heuristic
which is fine for workload shaping but loose.

## Build the prompt corpus (run once)

```
cd client
python build_corpus.py --output prompts/arxiv_corpus.jsonl --target 3000
```

The script politely paces requests to arXiv (3 s between calls) so it
will take 10-15 minutes. The resulting JSONL is a few MB and gets
committed to the repo for reproducibility.

## Smoke test

The client smoke test does not need a real engine: any HTTP echo server
that returns JSON will do for plumbing validation. The cleanest test is
to point the client at a tiny mock and verify the CSV format.

```
# Terminal 1: start a local mock that pretends to be an OpenAI-compatible engine
python -m http.server 8000  # any 200-returning server is fine for connectivity check

# Terminal 2: short benchmark run
python run_client.py \
    --config config.yaml \
    --output-dir /tmp/client_smoke \
    --duration-seconds 30 \
    --target-rate-rps 1 \
    --concurrency-cap 4
```

After the run, inspect:

```
ls -la /tmp/client_smoke/
head -5 /tmp/client_smoke/requests_000000.csv
```

You should see a CSV with one row per request, mostly with status="error"
because the mock isn't a real engine. What matters at this stage is that
the schema is correct and the rotation works.

## Real run

Once an engine is up (vLLM, Triton+vLLM, or our PyTorch+HF server):

```
python run_client.py \
    --config config.yaml \
    --output-dir ../runs/<run_id>/client \
    --duration-seconds 86400 \
    --protocol vllm_openai \
    --base-url http://localhost:8000 \
    --target-rate-rps 6 \
    --concurrency-cap 64
```

The `--target-rate-rps` for each engine should come from the pilot run
(85% of measured saturation throughput).

## Why open-loop with concurrency cap

Closed-loop schedulers (fixed in-flight count) auto-adapt to engine
slowdown, masking aging. Pure open-loop schedulers can collapse if
saturation is exceeded. The cap-with-drop variant gives us a stable
workload while exposing degradation through a measurable drop rate.

## Output format

One CSV file per ~60 s of run, named `requests_NNNNNN.csv`. Schema in
`_types.py`. Important columns:

- `submitted_at_unix`: when the client decided to issue the request
- `started_at_unix`: when the HTTP call actually started (after queueing
  inside the asyncio scheduler)
- `first_token_at_unix`: streaming only, when the first token came back
- `finished_at_unix`: completion timestamp
- `status`: ok | error | timeout | dropped
- `queue_time_s`, `ttft_s`, `e2e_latency_s`, `inter_token_latency_mean_s`
- `actual_input_tokens`, `actual_output_tokens` when reported by the engine

`status="dropped"` rows have no started/finished timestamps and indicate
the concurrency cap was full at submit time. The dropped fraction over
time is itself an aging indicator.

## Restartability

The client persists `state.json` (just the next req_id) in the output
directory every 30 s. On restart, req_ids continue monotonically, and
new CSV files start at the next sequence number, so analysis can simply
read all files in order.

## Caveats and threats to validity

- Output token counting depends on the engine. vLLM and our PyTorch+HF
  server return usage in the response; Triton+vLLM does not, so we
  approximate via character count and reconcile with the tokenizer
  client-side. Differences will be discussed in the paper.
- For a single run, the client runs on the same host as the engine in
  this project. CPU and RAM impact is logged via the system monitor
  and is small relative to engine resource use, but it is a known
  threat to validity that we declare in the paper.
