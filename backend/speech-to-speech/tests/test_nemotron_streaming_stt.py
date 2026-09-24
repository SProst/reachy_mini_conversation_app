from queue import Queue
from types import SimpleNamespace

import numpy as np

from speech_to_speech.arguments_classes.nemotron_streaming_stt_arguments import (
    NemotronStreamingSTTHandlerArguments,
)
from speech_to_speech.backend_registry import STT_BACKENDS
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, VADAudio
from speech_to_speech.STT.nemotron_streaming_handler import NemotronStreamingSTTHandler


class FakeBatch(dict):
    def to(self, *_args, **_kwargs):
        return self


class FakeFeatures:
    def __getitem__(self, _key):
        return self


class FakeProcessor:
    num_samples_first_audio_chunk = 800
    num_mel_frames_first_audio_chunk = 5
    num_samples_per_audio_chunk = 800
    num_mel_frames_per_audio_chunk = 4
    feature_extractor = SimpleNamespace(hop_length=160, n_fft=400)
    tokenizer = object()

    def __init__(self):
        self.streaming_chunks = []

    def __call__(self, audio, **kwargs):
        if kwargs.get("is_streaming"):
            self.streaming_chunks.append((len(audio), kwargs["is_first_audio_chunk"]))
        return FakeBatch(input_features=FakeFeatures(), prompt_ids=object())

    def decode(self, _sequences, skip_special_tokens):
        return "Hello robot." if skip_special_tokens else "Hello robot. <en-US>"


class FakeStreamer:
    def __init__(self, *_args, **_kwargs):
        self.queue = Queue()

    def put_text(self, text):
        self.queue.put(text)

    def end(self):
        self.queue.put(None)

    def __iter__(self):
        while True:
            text = self.queue.get()
            if text is None:
                return
            yield text


class FakeModel:
    device = "cpu"
    dtype = "float32"

    def generate(self, **kwargs):
        features = kwargs["input_features"]
        streamer = kwargs["streamer"]
        chunk_count = 0
        for _ in features:
            chunk_count += 1
            streamer.put_text("Hello " if chunk_count == 1 else "robot.")
        streamer.end()
        return SimpleNamespace(sequences=object())


def _handler(monkeypatch):
    handler = object.__new__(NemotronStreamingSTTHandler)
    handler.processor = FakeProcessor()
    handler.model = FakeModel()
    handler.language = "en-US"
    handler.last_language = "en"
    handler.num_lookahead_tokens = 6
    handler.gen_kwargs = {}
    handler.enable_live_transcription = True
    handler.live_transcription_update_interval = 0.0
    handler.sample_rate = 16000
    handler._stream = None
    handler._stream_turn_key = None
    handler._next_mel_frame_idx = 0
    handler._last_partial_text = ""
    monkeypatch.setattr(
        "speech_to_speech.STT.nemotron_streaming_handler.TextIteratorStreamer",
        FakeStreamer,
    )
    return handler


def test_progressive_and_final_events_share_native_stream(monkeypatch):
    handler = _handler(monkeypatch)
    progressive = VADAudio(
        audio=np.zeros(800, dtype=np.float32),
        mode="progressive",
        turn_id="turn-1",
        turn_revision=0,
    )
    partials = list(handler.process(progressive))

    final = VADAudio(
        audio=np.zeros(1440, dtype=np.float32),
        mode="final",
        turn_id="turn-1",
        turn_revision=0,
    )
    transcriptions = list(handler.process(final))

    assert len(partials) <= 1
    if partials:
        assert isinstance(partials[0], PartialTranscription)
        assert partials[0].turn_id == "turn-1"
    assert transcriptions == [
        Transcription(
            text="Hello robot.",
            language_code="en",
            turn_id="turn-1",
            turn_revision=0,
            speech_stopped_at_s=final.created_at_s,
        )
    ]
    assert handler.processor.streaming_chunks == [(800, True), (800, False)]


def test_auto_language_uses_nemotron_language_tag(monkeypatch):
    handler = _handler(monkeypatch)
    handler.language = "auto"

    text, language = handler._decode_output(SimpleNamespace(sequences=object()))

    assert text == "Hello robot."
    assert language == "en"


def test_decode_output_accepts_batched_processor_results(monkeypatch):
    handler = _handler(monkeypatch)
    handler.language = "auto"
    handler.processor.decode = lambda _sequences, skip_special_tokens: (
        ["Hello robot."] if skip_special_tokens else ["Hello robot. <en-US>"]
    )

    text, language = handler._decode_output(SimpleNamespace(sequences=object()))

    assert text == "Hello robot."
    assert language == "en"


def test_nemotron_backend_is_registered_with_streaming_defaults():
    spec = STT_BACKENDS["nemotron-streaming"]

    assert spec.kind == "stt"
    assert spec.normalize(NemotronStreamingSTTHandlerArguments()) == {
        "model_name": "nvidia/nemotron-3.5-asr-streaming-0.6b",
        "device": "auto",
        "compute_type": "float16",
        "language": "en-US",
        "num_lookahead_tokens": 6,
        "gen_kwargs": {},
    }
