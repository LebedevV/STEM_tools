# Dockerfile for the abtem-run worker pipeline.
#
# Pins abtem 1.0.9 (the conditional monkey-patches in abtem_run/__init__.py
# target specific 1.0.9 source substrings) and CUDA 13.x runtime (pairs
# with the gpu-cu13 pyproject extra: cupy-cuda13x + dask-cuda).
#
# Build context = repo root, so the build can reach LICENSE at the root
# alongside abTEM_simulations/:
#
#     docker build -f abTEM_simulations/Dockerfile -t abtem-run:0.1.0 .
#
# Run as a one-shot worker on a single .todo (spot-friendly — restart on
# preemption picks up the same .todo, SIGTERM handler cleans the partial):
#
#     docker run --rm --gpus all -v /host/job_root:/job abtem-run:0.1.0 \
#         abtem-run-worker /job/<phase>_<hkl>_<tilt> \
#                          /job/<phase>_<hkl>_<tilt>/seeds/seed_NNNNNN.todo
#
# Run the planning generator (one-shot, cheap, no GPU needed):
#
#     docker run --rm -v /host/job_root:/job abtem-run:0.1.0 \
#         abtem-run-generate /job/config.toml
#
# Run the aggregator (one-shot, no GPU needed):
#
#     docker run --rm -v /host/job_root:/job abtem-run:0.1.0 \
#         abtem-run-aggregate /job/<phase>_<hkl>_<tilt>

FROM nvidia/cuda:13.0.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Python 3.11 to match requires-python in pyproject.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/abtem-run

# COPY everything pyproject needs (LICENSE lives at the repo root; the
# package source lives under abTEM_simulations/).
COPY LICENSE ./
COPY abTEM_simulations/pyproject.toml ./
COPY abTEM_simulations/README.md ./
COPY abTEM_simulations/abtem_run ./abtem_run

# Install with the gpu-cu13 extra — cupy-cuda13x + dask-cuda. abtem is pinned
# to ==1.0.9 in pyproject (the runtime monkey-patches target its source); the
# pipeline tolerates rmm being absent (gpu_related.dask_cuda gates it, soft-dep).
RUN python3.11 -m pip install --no-cache-dir ".[gpu-cu13]"

# Build-time gate: fail the build if the abtem monkey-patches did not bind. A
# drifted abtem would silently no-op (_PATCHES_APPLIED records the miss) and the
# GPU paths fail only at runtime on a spot node. Imports fine without a GPU.
RUN python3.11 -c "import abtem_run, sys; sys.exit(0 if all(abtem_run._PATCHES_APPLIED.values()) else 1)"

# /work is the mount point for the per-run job tree (config.toml, CIFs,
# the gen_<UTC>/ output tree). The console scripts take absolute paths,
# so leaving CWD here is just convenience.
WORKDIR /work

# No ENTRYPOINT — the orchestrator picks one of abtem-run-{generate,
# worker,aggregate,extend,to-ensemble} per invocation.
CMD ["abtem-run-worker", "--help"]
