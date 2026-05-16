# ADR 002: Server preparation and version pinning for the experimental campaign

Status: accepted
Date: 2026-05-05

## Context

Experimental campaign requires a stable, reproducible server
environment for two consecutive weeks. The server (cci-csgpu11) is a
shared lab machine with 4 x NVIDIA L40S, 256 GB RAM, Ubuntu 24.04.4 LTS,
under administrative auto-update policies that include the NVIDIA
driver. Long-running benchmark runs (36 h production aging windows) cannot tolerate
mid-run kernel-module or library swaps, and require GPU-enabled Docker
containers.

During initial setup the following issues had to be resolved:

1. The default Docker data root (/var/lib/docker) is on a partition with
   only 32 GB free, insufficient for the engine container images
   (vLLM, Triton+vLLM, custom PyTorch) plus build cache.
2. The NVIDIA driver had been auto-upgraded between boots, leaving the
   running kernel module at 580.126.20 while user-space libraries
   advanced to 580.159.03. This produced "Driver/library version
   mismatch" failures from NVML and prevented `nvidia-persistenced`
   from starting, which in turn blocked the NVIDIA Container CDI mount
   and thus prevented `docker run --gpus` from succeeding.
3. The CDI specification at /var/run/cdi/nvidia.yaml mounts
   /run/nvidia-persistenced/socket; when persistenced is not running,
   any attempt to start a GPU-enabled container fails with an OCI
   mount error.

## Decisions

### D1. Move Docker data root to /home

Configuration change in /etc/docker/daemon.json:

    {
        "data-root": "/home/dcotrone/docker-data"
    }

/home is on the system_vg-home_lv LVM volume, 6.9 TB total, 6.4 TB
free. Permissions of the new directory set to 711 to limit visibility
while preserving root access for the daemon. Docker daemon restarted
to apply.

### D2. Reboot to realign NVIDIA driver

Server rebooted to allow the kernel module to be reloaded from the
updated 580.159.03 DKMS build, matching the installed user-space
libraries. After reboot:

    NVIDIA-SMI 580.159.03    Driver Version: 580.159.03    CUDA Version: 13.0

`nvidia-persistenced.service` started successfully on its own after
the realignment, restoring the persistence socket and unblocking the
CDI mount path used by the NVIDIA Container Toolkit.

### D3. Pin NVIDIA-related packages for the duration of the campaign

The following packages are placed on `apt-mark hold` for the entire
two-week experimental window. They will be released via `apt-mark
unhold` once the final campaign run completes.

    libnvidia-compute-580
    libnvidia-container-tools
    libnvidia-container1
    libnvidia-decode-580
    libnvidia-encode-580
    libnvidia-extra-580
    libnvidia-fbc1-580
    libnvidia-gl-580
    nvidia-container-toolkit
    nvidia-container-toolkit-base
    nvidia-driver-580
    nvidia-kernel-source-580
    nvidia-utils-580

Rationale. Mid-run NVIDIA driver upgrades are a known cause of
abrupt benchmark failures and silent measurement drift. Holding the
versions guarantees that all runs in the campaign observe the same
driver, NVML library, and container toolkit, eliminating a category of
confounders in cross-run comparisons.

### D4. Pinned environment baseline

For documentation in the paper's methodology section:

    OS:                 Ubuntu 24.04.4 LTS (noble), kernel 6.17.0-19-generic
    GPUs:               4 x NVIDIA L40S, 46 GB VRAM each
    NVIDIA driver:      580.159.03
    CUDA runtime:       13.0
    Docker:             29.4.2 (build 055a478)
    NVIDIA Toolkit:     1.19.0
    Conda env Python:   3.11
    System RAM:         256 GB

## Consequences

- Engine container images and build artifacts will be stored on /home,
  with abundant headroom for the three engines we plan to compare and
  any subsequent experiments.
- The driver version is fixed for the campaign. If a security advisory
  for this driver appears mid-campaign, the paper will document the
  decision to defer the upgrade to preserve experimental integrity.
- Future replication of the campaign on a different machine will need
  to either match this exact driver/toolkit combination or, if it
  diverges, validate that the resulting aging signatures are robust to
  the change. This will be addressed in the threats-to-validity
  section.

## References

- Docker daemon configuration: https://docs.docker.com/engine/daemon/
- apt-mark hold: https://manpages.ubuntu.com/manpages/noble/man8/apt-mark.8.html
- NVIDIA Container Toolkit CDI: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html
