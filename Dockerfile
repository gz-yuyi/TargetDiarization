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
RUN pip install --upgrade pip && \
    PIP_USE_DEPRECATED=legacy-resolver pip install -r /app/requirements.txt

COPY . /app

ARG DOWNLOAD_MODELS=0
RUN if [ "${DOWNLOAD_MODELS}" = "1" ]; then \
      git lfs install --skip-repo && \
      git clone https://www.modelscope.cn/models/jzx-ai-lab/target_diarization_models.git /app/modelscope_models && \
      cp -a /app/modelscope_models/. /app/ && \
      rm -rf /app/modelscope_models; \
    fi

ARG DOWNLOAD_HF_MODELS=0
RUN --mount=type=secret,id=HF_TOKEN \
    if [ "${DOWNLOAD_HF_MODELS}" = "1" ]; then \
      if [ -f /run/secrets/HF_TOKEN ]; then \
        HF_TOKEN=$(cat /run/secrets/HF_TOKEN) && \
        git -c http.extraHeader="Authorization: Bearer ${HF_TOKEN}" \
          clone https://huggingface.co/pyannote/speaker-diarization-3.1 \
          /app/pyannote/speaker-diarization-3.1; \
      else \
        echo "HF_TOKEN secret not provided; skip HuggingFace model download."; \
      fi; \
    fi

EXPOSE 8000 8300

ENTRYPOINT ["python"]
CMD ["webui.py"]
