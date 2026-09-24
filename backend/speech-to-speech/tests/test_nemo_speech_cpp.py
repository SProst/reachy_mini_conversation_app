import ctypes

import pytest

from speech_to_speech.TTS.nemo_speech_cpp import NeMoSpeechTTSRuntime


class FakeFunction:
    def __init__(self, implementation):
        self.implementation = implementation
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.implementation(self, *args)


class FakeLibrary:
    def __init__(self) -> None:
        self.destroyed = False
        self.nemo_speech_tts_runtime_config_default = FakeFunction(self._runtime_config_default)
        self.nemo_speech_tts_synthesis_options_default = FakeFunction(self._synthesis_options_default)
        self.nemo_speech_tts_synthesis_stats_default = FakeFunction(self._synthesis_stats_default)
        self.nemo_speech_tts_create = FakeFunction(self._create)
        self.nemo_speech_tts_destroy = FakeFunction(self._destroy)
        self.nemo_speech_tts_sample_rate = FakeFunction(lambda _fn, _handle: 22050)
        self.nemo_speech_tts_speaker_count = FakeFunction(lambda _fn, _handle: 2)
        self.nemo_speech_tts_speaker_name = FakeFunction(lambda _fn, _handle, index: (b"Aria", b"Sofia")[index])
        self.nemo_speech_tts_synthesize_text = FakeFunction(self._synthesize_text)
        self.nemo_speech_tts_last_error = FakeFunction(lambda _fn: b"")
        self.nemo_speech_tts_version = FakeFunction(lambda _fn: b"nemo-speech-tts 0.1.0")

    @staticmethod
    def _runtime_config_default(function):
        config = function.restype()
        config.size = ctypes.sizeof(config)
        return config

    @staticmethod
    def _synthesis_options_default(function):
        options = function.restype()
        options.size = ctypes.sizeof(options)
        options.speaker = -1
        return options

    @staticmethod
    def _synthesis_stats_default(function):
        stats = function.restype()
        stats.size = ctypes.sizeof(stats)
        return stats

    @staticmethod
    def _create(_function, _config, handle):
        ctypes.cast(handle, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(7)
        return 0

    def _destroy(self, _function, _handle):
        self.destroyed = True

    @staticmethod
    def _synthesize_text(function, _handle, _options, _text, callback, _user_data, stats):
        first = (ctypes.c_uint8 * 4)(1, 2, 3, 4)
        second = (ctypes.c_uint8 * 2)(5, 6)
        assert callback(first, len(first), None)
        assert callback(second, len(second), None)
        typed_stats = ctypes.cast(stats, function.argtypes[-1]).contents
        typed_stats.sample_rate = 16000
        typed_stats.audio_s = 0.25
        typed_stats.elapsed_s = 0.10
        typed_stats.e2e_ttfa_ms = 25.0
        return 0


def test_runtime_streams_native_callback_chunks(monkeypatch):
    library = FakeLibrary()
    monkeypatch.setattr("speech_to_speech.TTS.nemo_speech_cpp.ctypes.CDLL", lambda _path: library)
    runtime = NeMoSpeechTTSRuntime(
        library_path="libnemo_speech_tts.so",
        model_path="magpie.gguf",
        codec_path="codec.gguf",
        tokenizer_path="tokenizer",
        device="cuda",
        voice="Aria",
        language="en-US",
        threads=4,
        codec_threads=0,
        chunk_frames=3,
        codec_queue_depth=4,
        codec_history_frames=-1,
        codec_future_frames=1,
        window_ms=0,
        use_cfg=True,
        use_kv_cache=True,
        use_stateful_codec=True,
        codec_cpu=False,
        flush_partial_chunk=True,
        verbose=False,
    )

    chunks = list(runtime.stream("Hello.", "en-US", "Aria", 16000, lambda: False))

    assert chunks == [b"\x01\x02\x03\x04", b"\x05\x06"]
    assert runtime.version == "nemo-speech-tts 0.1.0"
    assert runtime.sample_rate == 22050
    assert runtime.voices == ("Aria", "Sofia")
    assert runtime.last_stats is not None
    assert runtime.last_stats.sample_rate == 16000
    assert runtime.last_stats.e2e_ttfa_ms == 25.0

    runtime.close()
    assert library.destroyed


def test_runtime_surfaces_native_call_exceptions(monkeypatch):
    library = FakeLibrary()

    def fail_synthesis(_function, *_args):
        raise OSError("native call failed")

    library.nemo_speech_tts_synthesize_text = FakeFunction(fail_synthesis)
    monkeypatch.setattr("speech_to_speech.TTS.nemo_speech_cpp.ctypes.CDLL", lambda _path: library)
    runtime = NeMoSpeechTTSRuntime(
        library_path="libnemo_speech_tts.so",
        model_path="magpie.gguf",
        codec_path="codec.gguf",
        tokenizer_path="tokenizer",
        device="cuda",
        voice="Aria",
        language="en-US",
        threads=4,
        codec_threads=0,
        chunk_frames=3,
        codec_queue_depth=4,
        codec_history_frames=-1,
        codec_future_frames=1,
        window_ms=0,
        use_cfg=True,
        use_kv_cache=True,
        use_stateful_codec=True,
        codec_cpu=False,
        flush_partial_chunk=True,
        verbose=False,
    )

    with pytest.raises(RuntimeError, match="native call failed"):
        list(runtime.stream("Hello.", "en-US", "Aria", 16000, lambda: False))
