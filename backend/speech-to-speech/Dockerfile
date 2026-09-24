FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NUMBA_DISABLE_CUDA=1
ENV OMP_NUM_THREADS=2
ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/src/app/.venv/bin:${PATH}"

WORKDIR /usr/src/app

# Install packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        build-essential \
        git \
        libportaudio2 \
        libsndfile1 \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir --break-system-packages uv

COPY pyproject.toml README.md LICENSE MANIFEST.in ./
RUN uv sync --python /usr/bin/python3 --no-install-project --no-dev --extra magpie

COPY . .
RUN uv sync --python /usr/bin/python3 --no-dev --extra magpie
RUN python -c "import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"
RUN python -c "import speech_to_speech.s2s_pipeline; print('speech-to-speech startup imports passed')"

EXPOSE 7860

CMD ["sh", "-c", "test -n \"$HF_TOKEN\" || { echo 'HF_TOKEN secret is required' >&2; exit 1; }; exec speech-to-speech serve --host 0.0.0.0 --port 7860 --thresh 0.6 --stt nemotron-streaming --nemotron_streaming_language en-US --nemotron_streaming_num_lookahead_tokens 6 --llm_backend responses-api --tts magpie --magpie_tts_device cuda --magpie_tts_voice Aria --magpie_tts_language en --model_name \"${LLM_MODEL_NAME:-meta-models/Muse-Glimmer-30B:together}\" --chat_size 30 --stream_batch_sentences 1 --responses_api_base_url \"${RESPONSES_API_BASE_URL:-https://router.huggingface.co/v1}\" --responses_api_api_key \"$HF_TOKEN\" --responses_api_reasoning_effort low --responses_api_stream --enable_live_transcription"]
