# E3 — PyTorch+HF naive serving baseline.
#
# Base: NGC PyTorch 25.09 — same release line as the Triton+vLLM container
# used in E2 (nvcr.io/nvidia/tritonserver:25.09-vllm-python-py3). This
# aligns CUDA/cuDNN/NCCL versions across E2 and E3, removing one source
# of confounding from the cross-engine comparison.
FROM nvcr.io/nvidia/pytorch:25.09-py3

# uvicorn must be PID 1 inside the container so that
# `docker inspect -f '{{.State.Pid}}' <container>` returns the host PID
# of the actual serving process. proc_monitor reads that PID via the
# pidfile written by launch.sh and tracks RSS / FDs / threads / IO of
# the right process.
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY server/ /app/server/

# HF cache is bind-mounted from the host by launch.sh so model downloads
# are not paid on every container restart.
ENV HF_HOME=/cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Exec form so uvicorn is PID 1 (no shell wrapper). HOST/PORT are fixed
# inside the container; external mapping happens via `docker run -p`.
CMD ["uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--no-access-log"]
