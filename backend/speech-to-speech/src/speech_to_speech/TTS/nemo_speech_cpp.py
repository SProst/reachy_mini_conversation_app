"""ctypes bindings for the stable NeMo-Speech.cpp TTS C API."""

from __future__ import annotations

import ctypes
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from queue import Queue
from threading import Event, Thread
from typing import Any

logger = logging.getLogger(__name__)

NEMO_SPEECH_TTS_OK = 0
NEMO_SPEECH_TTS_ERROR_CANCELLED = 4


class _ModelConfig(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("magpie_model", ctypes.c_char_p),
        ("codec_model", ctypes.c_char_p),
        ("tokenizer_model_dir", ctypes.c_char_p),
        ("text_normalizer_model_dir", ctypes.c_char_p),
    ]


class _RuntimeConfig(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("speaker", ctypes.c_int32),
        ("threads", ctypes.c_int32),
        ("codec_threads", ctypes.c_int32),
        ("seed", ctypes.c_int32),
        ("steps", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("chunk_frames", ctypes.c_int32),
        ("codec_queue_depth", ctypes.c_int32),
        ("codec_history_frames", ctypes.c_int32),
        ("codec_future_frames", ctypes.c_int32),
        ("window_ms", ctypes.c_int32),
        ("temperature", ctypes.c_float),
        ("override_temperature", ctypes.c_bool),
        ("cfg_scale", ctypes.c_float),
        ("override_cfg_scale", ctypes.c_bool),
        ("use_cfg", ctypes.c_bool),
        ("use_local_transformer", ctypes.c_bool),
        ("use_kv_cache", ctypes.c_bool),
        ("use_stateful_codec", ctypes.c_bool),
        ("codec_cpu", ctypes.c_bool),
        ("flush_partial_chunk", ctypes.c_bool),
        ("verbose", ctypes.c_bool),
        ("lt_backend", ctypes.c_int),
        ("sampling_backend", ctypes.c_int),
        ("uma_mode", ctypes.c_int),
        ("longform_mode", ctypes.c_int),
        ("lt_fp32", ctypes.c_bool),
    ]


class _SynthesizerConfig(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("model", ctypes.POINTER(_ModelConfig)),
        ("runtime", ctypes.POINTER(_RuntimeConfig)),
        ("default_language_code", ctypes.c_char_p),
        ("default_voice_name", ctypes.c_char_p),
    ]


class _SynthesisOptions(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("request_id", ctypes.c_char_p),
        ("language_code", ctypes.c_char_p),
        ("speaker", ctypes.c_int32),
        ("seed", ctypes.c_int32),
        ("steps", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("temperature", ctypes.c_float),
        ("override_temperature", ctypes.c_bool),
        ("cfg_scale", ctypes.c_float),
        ("override_cfg_scale", ctypes.c_bool),
        ("voice_name", ctypes.c_char_p),
        ("output_sample_rate", ctypes.c_int32),
    ]


class SynthesisStats(ctypes.Structure):
    """Native per-request latency and throughput measurements."""

    _fields_ = [
        ("size", ctypes.c_size_t),
        ("sample_rate", ctypes.c_int32),
        ("generated_frames", ctypes.c_int32),
        ("chunks", ctypes.c_int32),
        ("e2e_chunks", ctypes.c_int32),
        ("samples_written", ctypes.c_uint64),
        ("tokenizer_ms", ctypes.c_double),
        ("encoder_ms", ctypes.c_double),
        ("audio_s", ctypes.c_double),
        ("elapsed_s", ctypes.c_double),
        ("rtf", ctypes.c_double),
        ("rtfx", ctypes.c_double),
        ("ttfa_ms", ctypes.c_double),
        ("icl_avg_ms", ctypes.c_double),
        ("icl_min_ms", ctypes.c_double),
        ("icl_max_ms", ctypes.c_double),
        ("decoder_audio_s", ctypes.c_double),
        ("decoder_elapsed_s", ctypes.c_double),
        ("decoder_rtfx", ctypes.c_double),
        ("decoder_ttft_ms", ctypes.c_double),
        ("decoder_itl_avg_ms", ctypes.c_double),
        ("decoder_itl_min_ms", ctypes.c_double),
        ("decoder_itl_max_ms", ctypes.c_double),
        ("decoder_itl_p95_ms", ctypes.c_double),
        ("decoder_itl_p99_ms", ctypes.c_double),
        ("codec_audio_s", ctypes.c_double),
        ("codec_elapsed_s", ctypes.c_double),
        ("codec_rtfx", ctypes.c_double),
        ("codec_ttfa_ms", ctypes.c_double),
        ("codec_icl_avg_ms", ctypes.c_double),
        ("codec_icl_min_ms", ctypes.c_double),
        ("codec_icl_max_ms", ctypes.c_double),
        ("codec_icl_p95_ms", ctypes.c_double),
        ("codec_icl_p99_ms", ctypes.c_double),
        ("e2e_ttfa_ms", ctypes.c_double),
        ("e2e_icl_avg_ms", ctypes.c_double),
        ("e2e_icl_min_ms", ctypes.c_double),
        ("e2e_icl_max_ms", ctypes.c_double),
        ("e2e_icl_p95_ms", ctypes.c_double),
        ("e2e_icl_p99_ms", ctypes.c_double),
        ("e2e_rtfx", ctypes.c_double),
    ]


_PcmCallback = ctypes.CFUNCTYPE(
    ctypes.c_bool,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
    ctypes.c_void_p,
)


@dataclass(frozen=True)
class _SynthesisFinished:
    status: int
    error: str
    stats: SynthesisStats


class NeMoSpeechTTSRuntime:
    """Stream Magpie PCM from a NeMo-Speech.cpp synthesizer."""

    def __init__(
        self,
        library_path: str,
        model_path: str,
        codec_path: str,
        tokenizer_path: str,
        device: str,
        voice: str,
        language: str,
        threads: int,
        codec_threads: int,
        chunk_frames: int,
        codec_queue_depth: int,
        codec_history_frames: int,
        codec_future_frames: int,
        window_ms: int,
        use_cfg: bool,
        use_kv_cache: bool,
        use_stateful_codec: bool,
        codec_cpu: bool,
        flush_partial_chunk: bool,
        verbose: bool,
    ) -> None:
        for description, path in (
            ("NeMo-Speech.cpp library", library_path),
            ("Magpie GGUF", model_path),
            ("NanoCodec GGUF", codec_path),
            ("Magpie tokenizer directory", tokenizer_path),
        ):
            if not path:
                raise ValueError(f"{description} path is required")

        self._library = ctypes.CDLL(library_path)
        self._configure_functions()
        runtime = self._library.nemo_speech_tts_runtime_config_default()
        runtime.threads = threads
        runtime.codec_threads = codec_threads
        runtime.chunk_frames = chunk_frames
        runtime.codec_queue_depth = codec_queue_depth
        runtime.codec_history_frames = codec_history_frames
        runtime.codec_future_frames = codec_future_frames
        runtime.window_ms = window_ms
        runtime.use_cfg = use_cfg
        runtime.use_kv_cache = use_kv_cache
        runtime.use_stateful_codec = use_stateful_codec
        runtime.codec_cpu = codec_cpu
        runtime.flush_partial_chunk = flush_partial_chunk
        runtime.verbose = verbose
        if device.lower() == "cuda":
            runtime.lt_backend = 2
            runtime.sampling_backend = 2
        elif device.lower() == "cpu":
            runtime.lt_backend = 1
            runtime.sampling_backend = 1

        model = _ModelConfig(
            size=ctypes.sizeof(_ModelConfig),
            magpie_model=model_path.encode(),
            codec_model=codec_path.encode(),
            tokenizer_model_dir=tokenizer_path.encode(),
            text_normalizer_model_dir=None,
        )
        config = _SynthesizerConfig(
            size=ctypes.sizeof(_SynthesizerConfig),
            model=ctypes.pointer(model),
            runtime=ctypes.pointer(runtime),
            default_language_code=language.encode(),
            default_voice_name=voice.encode(),
        )
        self._synthesizer = ctypes.c_void_p()
        status = self._library.nemo_speech_tts_create(ctypes.byref(config), ctypes.byref(self._synthesizer))
        if status != NEMO_SPEECH_TTS_OK:
            raise RuntimeError(f"Failed to create NeMo-Speech.cpp TTS runtime: {self._last_error()}")

        try:
            count = self._library.nemo_speech_tts_speaker_count(self._synthesizer)
            self.voices = tuple(
                self._library.nemo_speech_tts_speaker_name(self._synthesizer, index).decode() for index in range(count)
            )
            self.sample_rate = self._library.nemo_speech_tts_sample_rate(self._synthesizer)
            self.version = self._library.nemo_speech_tts_version().decode()
        except Exception:
            self._library.nemo_speech_tts_destroy(self._synthesizer)
            self._synthesizer = ctypes.c_void_p()
            raise
        self.last_stats: SynthesisStats | None = None

    def _configure_functions(self) -> None:
        library = self._library
        library.nemo_speech_tts_runtime_config_default.argtypes = []
        library.nemo_speech_tts_runtime_config_default.restype = _RuntimeConfig
        library.nemo_speech_tts_synthesis_options_default.argtypes = []
        library.nemo_speech_tts_synthesis_options_default.restype = _SynthesisOptions
        library.nemo_speech_tts_synthesis_stats_default.argtypes = []
        library.nemo_speech_tts_synthesis_stats_default.restype = SynthesisStats
        library.nemo_speech_tts_create.argtypes = [
            ctypes.POINTER(_SynthesizerConfig),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        library.nemo_speech_tts_create.restype = ctypes.c_int
        library.nemo_speech_tts_destroy.argtypes = [ctypes.c_void_p]
        library.nemo_speech_tts_destroy.restype = None
        library.nemo_speech_tts_sample_rate.argtypes = [ctypes.c_void_p]
        library.nemo_speech_tts_sample_rate.restype = ctypes.c_int32
        library.nemo_speech_tts_speaker_count.argtypes = [ctypes.c_void_p]
        library.nemo_speech_tts_speaker_count.restype = ctypes.c_int32
        library.nemo_speech_tts_speaker_name.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        library.nemo_speech_tts_speaker_name.restype = ctypes.c_char_p
        library.nemo_speech_tts_synthesize_text.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_SynthesisOptions),
            ctypes.c_char_p,
            _PcmCallback,
            ctypes.c_void_p,
            ctypes.POINTER(SynthesisStats),
        ]
        library.nemo_speech_tts_synthesize_text.restype = ctypes.c_int
        library.nemo_speech_tts_last_error.argtypes = []
        library.nemo_speech_tts_last_error.restype = ctypes.c_char_p
        library.nemo_speech_tts_version.argtypes = []
        library.nemo_speech_tts_version.restype = ctypes.c_char_p

    def _last_error(self) -> str:
        message = self._library.nemo_speech_tts_last_error()
        return message.decode(errors="replace") if message else "unknown native error"

    def stream(
        self,
        text: str,
        language: str,
        voice: str,
        output_sample_rate: int,
        cancelled: Callable[[], bool],
    ) -> Iterator[bytes]:
        """Yield PCM16 chunks while the native synthesizer is running."""

        output: Queue[bytes | _SynthesisFinished] = Queue()
        consumer_stopped = Event()

        def should_cancel() -> bool:
            return consumer_stopped.is_set() or cancelled()

        @_PcmCallback
        def receive_pcm(pcm: Any, n_bytes: int, _user_data: Any) -> bool:
            try:
                if should_cancel():
                    return False
                output.put(ctypes.string_at(pcm, n_bytes))
                return not should_cancel()
            except Exception:
                logger.exception("Failed to receive NeMo-Speech.cpp PCM")
                return False

        def synthesize() -> None:
            stats = SynthesisStats(size=ctypes.sizeof(SynthesisStats))
            try:
                options = self._library.nemo_speech_tts_synthesis_options_default()
                options.language_code = language.encode()
                options.voice_name = voice.encode()
                options.output_sample_rate = output_sample_rate
                stats = self._library.nemo_speech_tts_synthesis_stats_default()
                status = self._library.nemo_speech_tts_synthesize_text(
                    self._synthesizer,
                    ctypes.byref(options),
                    text.encode(),
                    receive_pcm,
                    None,
                    ctypes.byref(stats),
                )
                error = self._last_error() if status != NEMO_SPEECH_TTS_OK else ""
            except Exception as exc:
                logger.exception("NeMo-Speech.cpp synthesis call failed")
                status = 3
                error = str(exc)
            output.put(_SynthesisFinished(status, error, stats))

        worker = Thread(target=synthesize, name="nemo-speech-tts", daemon=True)
        worker.start()
        finished: _SynthesisFinished | None = None
        try:
            while finished is None:
                item = output.get()
                if isinstance(item, _SynthesisFinished):
                    finished = item
                else:
                    yield item
        finally:
            consumer_stopped.set()
            worker.join(timeout=5)
            if worker.is_alive():
                logger.warning("NeMo-Speech.cpp synthesis worker did not stop within 5 seconds")

        self.last_stats = finished.stats
        if finished.status == NEMO_SPEECH_TTS_ERROR_CANCELLED and cancelled():
            return
        if finished.status != NEMO_SPEECH_TTS_OK:
            raise RuntimeError(f"NeMo-Speech.cpp synthesis failed: {finished.error}")

    def close(self) -> None:
        """Release the native synthesizer."""

        if self._synthesizer.value:
            self._library.nemo_speech_tts_destroy(self._synthesizer)
            self._synthesizer = ctypes.c_void_p()
