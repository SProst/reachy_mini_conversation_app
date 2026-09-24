from threading import Event
from types import SimpleNamespace

import numpy as np

from speech_to_speech.arguments_classes.magpie_tts_arguments import MagpieTTSHandlerArguments
from speech_to_speech.backend_registry import TTS_BACKENDS
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput
from speech_to_speech.TTS.magpie_tts_handler import MAGPIE_VOICES, MagpieTTSHandler


class FakeRuntime:
    def __init__(self) -> None:
        self.voices = tuple(MAGPIE_VOICES)
        self.finished = False
        self.calls: list[tuple[str, str, str, int]] = []
        self.last_stats = SimpleNamespace(
            audio_s=0.08,
            elapsed_s=0.04,
            rtf=0.5,
            e2e_ttfa_ms=20.0,
            e2e_chunks=2,
        )

    def stream(self, text, language, voice, output_sample_rate, cancelled):
        self.calls.append((text, language, voice, output_sample_rate))
        if cancelled():
            return
        yield np.arange(512, dtype=np.int16).tobytes()
        if cancelled():
            return
        yield np.arange(128, dtype=np.int16).tobytes()
        self.finished = True

    def close(self) -> None:
        pass


def _handler() -> MagpieTTSHandler:
    handler = object.__new__(MagpieTTSHandler)
    handler.voice = "Aria"
    handler.language = "en-US"
    handler.blocksize = 512
    handler.speculative_turns = None
    handler.cancel_scope = None
    handler.stop_event = Event()
    handler.runtime = FakeRuntime()
    return handler


def test_magpie_backend_is_registered(monkeypatch):
    for name in (
        "NEMO_SPEECH_TTS_LIBRARY",
        "NEMO_SPEECH_TTS_MODEL_PATH",
        "NEMO_SPEECH_TTS_CODEC_PATH",
        "NEMO_SPEECH_TTS_TOKENIZER_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    spec = TTS_BACKENDS["magpie"]

    assert spec.kind == "tts"
    assert spec.required_extra is None
    assert spec.normalize(MagpieTTSHandlerArguments()) == {
        "library_path": "libnemo_speech_tts.so",
        "model_path": "",
        "codec_path": "",
        "tokenizer_path": "",
        "device": "cuda",
        "voice": "Aria",
        "language": "en-US",
        "threads": 4,
        "codec_threads": 0,
        "chunk_frames": 3,
        "codec_queue_depth": 4,
        "codec_history_frames": -1,
        "codec_future_frames": 1,
        "window_ms": 0,
        "use_cfg": True,
        "use_kv_cache": True,
        "use_stateful_codec": True,
        "codec_cpu": False,
        "flush_partial_chunk": True,
        "verbose": False,
        "blocksize": 512,
        "gen_kwargs": {},
    }


def test_runtime_voice_supports_magpie_names_case_insensitively():
    handler = _handler()
    audio_output = SimpleNamespace(voice="sofia")
    runtime_config = SimpleNamespace(session=SimpleNamespace(audio=SimpleNamespace(output=audio_output)))
    tts_input = TTSInput.model_construct(text="Hello", runtime_config=runtime_config, response=None)

    assert handler._resolve_voice(tts_input) == "Sofia"
    assert set(MAGPIE_VOICES) == {"Aria", "Jason", "John", "Leo", "Sofia"}


def test_language_normalization_supports_native_locales():
    assert MagpieTTSHandler._normalize_language("en-US", fallback="fr") == "en-US"
    assert MagpieTTSHandler._normalize_language("de_DE", fallback="en") == "de-DE"
    assert MagpieTTSHandler._normalize_language("unknown", fallback="fr-FR") == "fr-FR"


def test_text_is_trimmed_and_terminated_for_magpie():
    assert MagpieTTSHandler._prepare_text(" Hello ") == "Hello."
    assert MagpieTTSHandler._prepare_text("Already done!") == "Already done!"


def test_process_yields_pcm_before_native_synthesis_finishes():
    handler = _handler()
    runtime = handler.runtime
    assert isinstance(runtime, FakeRuntime)
    tts_input = TTSInput.model_construct(text="Hello", language_code="en-US", runtime_config=None, response=None)

    chunks = handler.process(tts_input)
    first = next(chunks)

    assert first.shape == (512,)
    assert first.dtype == np.int16
    assert not runtime.finished
    remaining = list(chunks)
    assert runtime.finished
    assert len(remaining) == 1
    assert remaining[0].shape == (512,)
    assert runtime.calls == [("Hello.", "en-US", "Aria", 16000)]


def test_process_stops_native_stream_after_cancellation():
    handler = _handler()
    handler.cancel_scope = CancelScope()
    tts_input = TTSInput.model_construct(text="Hello", language_code="en-US", runtime_config=None, response=None)

    chunks = handler.process(tts_input)
    next(chunks)
    handler.cancel_scope.cancel()

    assert list(chunks) == []


def test_end_of_response_closes_audio_response():
    assert list(_handler().process(EndOfResponse())) == [AUDIO_RESPONSE_DONE]
