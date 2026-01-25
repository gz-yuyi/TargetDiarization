# syntax=docker/dockerfile:1.4
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    git-lfs \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
ARG TORCH_VERSION=2.2.2
ARG TORCH_CUDA=cu121
RUN python -m pip install --upgrade pip && \
    PIP_USE_DEPRECATED=legacy-resolver python -m pip install -r /app/requirements.txt && \
    python -m pip install --no-deps --force-reinstall \
      --index-url https://download.pytorch.org/whl/${TORCH_CUDA} \
      torch==${TORCH_VERSION} \
      torchaudio==${TORCH_VERSION}

COPY . /app

ARG DOWNLOAD_MODELS=0
RUN if [ "${DOWNLOAD_MODELS}" = "1" ]; then \
      git lfs install --skip-repo && \
      git clone --depth 1 https://www.modelscope.cn/models/jzx-ai-lab/target_diarization_models.git /app/modelscope_models && \
      for path in /app/modelscope_models/*; do \
        [ -e "$path" ] || continue; \
        name="$(basename "$path")"; \
        dest="/app/$name"; \
        if [ -d "$path" ] && [ -d "$dest" ]; then \
          find "$path" -mindepth 1 -maxdepth 1 -exec mv -f {} "$dest/" \; && \
          rmdir "$path" 2>/dev/null || true; \
        else \
          mv -f "$path" /app/; \
        fi; \
      done && \
      rmdir /app/modelscope_models 2>/dev/null || true; \
    fi

EXPOSE 8000 8300

ENTRYPOINT ["python"]
CMD ["webui.py"]
