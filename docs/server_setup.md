# Server setup guide

End-to-end procedure to prepare the experimental server for the
benchmark campaign. Target host: Ubuntu LTS, 4 x NVIDIA L40S, 256 GB
RAM, dedicated for two weeks. NVIDIA driver 580+, CUDA 13.0 runtime
already installed.

The procedure is split into nine numbered steps. Each step ends with a
verification command whose expected output is described. If a step
fails, stop and resolve before proceeding.

The whole setup takes about 30-45 minutes if everything goes smoothly.

---

## Step 0. Preflight checks

Confirm that the host is in the expected state.

```bash
nvidia-smi
lsb_release -a
df -h /var/lib
free -g
uname -r
```

What to look for:
- `nvidia-smi` lists 4 L40S GPUs, all idle (0 MiB used), no processes running.
- `lsb_release` reports Ubuntu 22.04 or 24.04 LTS.
- `/var/lib` partition has at least 100 GB free (Docker stores images here).
- `free -g` confirms ~256 GB total RAM.
- Kernel 5.15 or newer.

If any of these is off, stop and discuss before continuing.

---

## Step 1. Install Docker Engine

We use Docker's official APT repository for a recent, supported version.

```bash
# Remove any old/conflicting Docker packages
sudo apt-get remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true

# Prerequisites
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# Add Docker's GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add the Docker APT repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine, CLI, containerd, buildx, compose
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin

# Allow your user to run docker without sudo
sudo usermod -aG docker $USER
```

Activate the new group membership without logging out:

```bash
newgrp docker
```

Verification:

```bash
docker --version
docker compose version
docker run --rm hello-world
```

Expected: `docker --version` reports Docker 24.x or newer; `hello-world`
prints the standard "Hello from Docker!" greeting.

---

## Step 2. Install NVIDIA Container Toolkit

This is what makes `docker run --gpus all` work. Without it, containers
cannot see the GPUs.

```bash
# Add NVIDIA's GPG key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Add the NVIDIA Container Toolkit APT repository
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use the NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Reload Docker daemon to apply the runtime change
sudo systemctl restart docker
```

Verification:

```bash
docker info | grep -i runtime
```

Expected: at least one line mentioning `nvidia` among the available
runtimes.

---

## Step 3. Smoke test GPU inside Docker

This is the critical end-to-end test for the Docker + NVIDIA path.

```bash
# Test all GPUs from inside a container
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Expected: the command pulls a small CUDA base image (a few hundred MB
the first time) and then prints `nvidia-smi` output identical to what
you see on the host: 4 L40S GPUs, driver 580.x, all idle.

Then test single-GPU isolation, which is what the experiments will use:

```bash
docker run --rm --gpus '"device=0"' nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
docker run --rm --gpus '"device=1"' nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Expected: each command shows exactly one GPU, with index 0 in the first
case (note: NVIDIA renumbers devices visible to the container starting
from 0 even when isolated). What matters is that only one GPU is
visible.

If all of the above works, the Docker + NVIDIA infrastructure is ready.

---

## Step 4. Set up the workspace

We keep three things outside the Git repo: experiment runs, the
HuggingFace cache, and the conda environment.

```bash
mkdir -p ~/wosar/runs ~/wosar/hf_cache
cd ~/wosar
git clone https://github.com/dcotroneCLT/llm-serving-bench.git
cd llm-serving-bench
ls -la
git log --oneline -5
```

Expected: clone succeeds, `git log` shows the commits we have pushed so
far (initial skeleton, monitoring, client).

---

## Step 5. Conda environment for the host-side tooling

The host-side tools (monitoring agents, benchmark client, analysis
scripts) live in a conda environment. The serving engines themselves
will run inside Docker containers in the next phases, NOT inside this
conda env.

```bash
# Create environment
conda create -n wosar python=3.11 -y
conda activate wosar

# Upgrade pip
pip install --upgrade pip

# Monitoring + client dependencies (pip is fine)
pip install psutil nvidia-ml-py httpx pyyaml tiktoken aiohttp

# Analysis dependencies (conda-forge for clean binaries)
conda install -y -c conda-forge pandas numpy scipy statsmodels \
    matplotlib pymannkendall pyarrow
```

Verification:

```bash
python3 -c "import psutil, pynvml, httpx, yaml, tiktoken, pandas, scipy, statsmodels, pymannkendall; print('all ok')"
```

Expected output: `all ok`.

Important: from now on, every host-side command in this guide assumes
you have run `conda activate wosar` first. Add this to your shell
session whenever you reconnect to the server.

---

## Step 6. Smoke test the monitoring agents

Now we validate the three monitoring agents on the real hardware. This
is what will catch any L40S-specific NVML quirks before they bite us
during a 36-hour production run.

Three short tests, ~1 minute each.

### 6a. System monitor (no GPU, no privileges)

```bash
cd ~/wosar/llm-serving-bench/monitoring
mkdir -p /tmp/mon_smoke
timeout 30 python3 system_monitor.py --output-dir /tmp/mon_smoke --period-seconds 1
ls -la /tmp/mon_smoke/
head -3 /tmp/mon_smoke/system_000000.csv
```

What to check: about 30 rows of data, all 26 columns populated. Pay
attention to `mem_cached_bytes`, `mem_buffers_bytes`,
`swap_in_bytes_cumulative`, `cpu_percent_iowait`, `fd_allocated`,
`fd_max`. If any of these are empty in every row, note it and report
back.

### 6b. GPU monitor on L40S (the critical test)

```bash
mkdir -p /tmp/gpu_smoke
timeout 30 python3 gpu_monitor.py --gpu-index 0 --output-dir /tmp/gpu_smoke --period-seconds 1
head -3 /tmp/gpu_smoke/gpu0_000000.csv
```

What to check: `vram_used_bytes`, `vram_total_bytes`, `gpu_util_percent`,
`temperature_c`, `power_draw_w` MUST all be populated (these are the
fields we will base the paper on).

It is OK if these are EMPTY on L40S: `fan_percent` (data-center cards
have no NVML-reported fan), and possibly `pcie_tx_kb`, `pcie_rx_kb` on
some driver versions. If any of these are empty in every row, that's
expected.

### 6c. Process monitor

```bash
mkdir -p /tmp/proc_smoke
sleep 100 &
SLEEP_PID=$!
timeout 30 python3 proc_monitor.py --pid $SLEEP_PID \
    --output-dir /tmp/proc_smoke --label sleeptest --period-seconds 2
head -3 /tmp/proc_smoke/sleeptest_000000.csv
kill $SLEEP_PID 2>/dev/null
```

What to check: all fields populated except `cpu_percent` on the first
row (None by design). `uss_bytes` and `pss_bytes` should be populated
on Linux; if they are empty, report it and we will fall back to `rss`.

After all three tests, send me the first 3 lines of each of the three
CSV files (system, gpu, proc). From those 9 lines I can confirm
everything is healthy.

---

## Step 7. Build the prompt corpus

One-shot job. Downloads a few thousand arXiv abstracts and writes them
as JSONL. Polite delay = 3 s between API calls, so this takes 10-15
minutes.

```bash
cd ~/wosar/llm-serving-bench/client
python3 build_corpus.py --output prompts/arxiv_corpus.jsonl --target 3000
ls -la prompts/
wc -l prompts/arxiv_corpus.jsonl
head -1 prompts/arxiv_corpus.jsonl | python3 -c "import sys, json; d=json.loads(sys.stdin.read()); print('keys:', list(d.keys())); print('text length:', len(d['text']), 'chars')"
```

Expected: ~3000 lines, each a JSON object with at least `text`. Average
text length around 1000-1500 characters per record.

Then commit it to the repo:

```bash
cd ~/wosar/llm-serving-bench
git add client/prompts/arxiv_corpus.jsonl
git commit -m "Add arXiv prompt corpus (3000 items, multi-category)"
git push
```

---

## Step 8. Verify campaign tooling

From the repo root, verify that the campaign descriptor parses and the
slot schedule is sane:

```bash
cd ~/wosar/llm-serving-bench
python3 scripts/campaign.py \
  --campaign-yaml campaigns/wosar2026/campaign.yaml \
  --dry-run
```

Expected: 18 production runs plus the configured sanity run, split
across GPU slots `gpu0`, `gpu1`, and `gpu2`.

Run a short smoke gate before any production launch:

```bash
bash scripts/smoke_test.sh campaigns/wosar2026/cells/e1.yaml
```

Expected: the script exits 0 and reports GO. If it fails, inspect the
printed hard-fail line before launching the 36h campaign slot.

## Step 9. Pin images and launch

Pin or verify the engine images:

```bash
bash scripts/utils/pin_images.sh
```

Then launch the campaign through the orchestrator:

```bash
python3 scripts/campaign.py \
  --campaign-yaml campaigns/wosar2026/campaign.yaml \
  --start
```

Resume after interruption:

```bash
python3 scripts/campaign.py \
  --campaign-yaml campaigns/wosar2026/campaign.yaml \
  --resume
```

---

## What to do if a step fails

Send me the exact command and the exact output (or paste the error
message). Most failures are well-known and have one-line fixes (group
membership not active yet, repo signing key missing, kernel module not
loaded, etc).

## Reference

- Docker official install guide: https://docs.docker.com/engine/install/ubuntu/
- NVIDIA Container Toolkit install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- Conda environments: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
