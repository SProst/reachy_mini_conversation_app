FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS nemo-speech-builder

ARG NEMO_SPEECH_CPP_REVISION=4f9676226f667d14608487df744f375db87127f8

ENV DEBIAN_FRONTEND=noninteractive
ENV NEMO_SPEECH_MODEL_DIR=/opt/nemo-models

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --filter=blob:none https://github.com/NVIDIA/NeMo-Speech.cpp.git /opt/nemo-speech-src \
    && git -C /opt/nemo-speech-src checkout "${NEMO_SPEECH_CPP_REVISION}" \
    && git -C /opt/nemo-speech-src submodule update --init --depth 1 ggml

WORKDIR /opt/nemo-speech-src

RUN scripts/configure.sh cuda-tts \
        -DCMAKE_CUDA_ARCHITECTURES=89 \
        -DGGML_NATIVE=OFF \
        -DNEMO_SPEECH_BUILD_MIC_CAPTURE=OFF \
    && cmake --build --preset cuda-tts --parallel 2 \
    && cmake --install build/cuda-tts --prefix /opt/nemo-speech

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && LD_LIBRARY_PATH=/opt/nemo-speech/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs \
        /opt/nemo-speech/bin/nemo-speech pull magpie


FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NUMBA_DISABLE_CUDA=1
ENV OMP_NUM_THREADS=2
ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/src/app/.venv/bin:/opt/nemo-speech/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/nemo-speech/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV NEMO_SPEECH_MODEL_DIR=/opt/nemo-models
ENV NEMO_SPEECH_TTS_LIBRARY=/opt/nemo-speech/lib/libnemo_speech_tts.so
ENV NEMO_SPEECH_TTS_MODEL_PATH=/opt/nemo-models/nvidia/magpie_tts_multilingual_357m/452ef560f972c38d5fc16476259aac9456453547/magpie_tts_multilingual_357m.v2602.f16.gguf
ENV NEMO_SPEECH_TTS_CODEC_PATH=/opt/nemo-models/nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps/fc00890b604aa2de298d2641ffc6c5f6caf8c4d7/nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf
ENV NEMO_SPEECH_TTS_TOKENIZER_PATH=/opt/nemo-models/nvidia/magpie_tts_multilingual_357m/452ef560f972c38d5fc16476259aac9456453547/tokenizer

WORKDIR /usr/src/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        libportaudio2 \
        libsndfile1 \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir --break-system-packages uv

COPY --from=nemo-speech-builder /opt/nemo-speech /opt/nemo-speech
COPY --from=nemo-speech-builder /opt/nemo-models /opt/nemo-models

COPY pyproject.toml README.md LICENSE MANIFEST.in ./
RUN uv sync --python /usr/bin/python3 --no-install-project --no-dev

COPY . .
RUN uv sync --python /usr/bin/python3 --no-dev
RUN python -c "import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"
RUN python -c "import speech_to_speech.s2s_pipeline; print('speech-to-speech startup imports passed')"
RUN python -c "from speech_to_speech.TTS.nemo_speech_cpp import NeMoSpeechTTSRuntime; print('NeMo-Speech.cpp bindings passed')"

EXPOSE 7860

CMD ["sh", "-c", "test -n \"$HF_TOKEN\" || { echo 'HF_TOKEN secret is required' >&2; exit 1; }; exec speech-to-speech serve --host 0.0.0.0 --port 7860 --thresh 0.6 --stt nemotron-streaming --nemotron_streaming_language en-US --nemotron_streaming_num_lookahead_tokens 6 --llm_backend responses-api --tts magpie --magpie_tts_device cuda --magpie_tts_voice Aria --magpie_tts_language en-US --magpie_tts_chunk_frames 3 --magpie_tts_codec_queue_depth 4 --model_name \"${LLM_MODEL_NAME:-meta-models/Muse-Glimmer-30B:together}\" --chat_size 30 --stream_batch_sentences 1 --responses_api_base_url \"${RESPONSES_API_BASE_URL:-https://router.huggingface.co/v1}\" --responses_api_api_key \"$HF_TOKEN\" --responses_api_reasoning_effort low --responses_api_stream --enable_live_transcription"]
