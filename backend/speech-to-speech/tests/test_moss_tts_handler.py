from types import SimpleNamespace

import numpy as np
import torch

from speech_to_speech.arguments_classes.moss_tts_arguments import MossTTSHandlerArguments
from speech_to_speech.backend_registry import TTS_BACKENDS
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput
from speech_to_speech.TTS.moss_tts_handler import (
    MOSS_VOICE_FILES,
    MossTTSHandler,
    _accept_legacy_input_embeds,
)


def _handler() -> MossTTSHandler:
    handler = object.__new__(MossTTSHandler)
    handler.voice = "MOSS Voice 1"
    handler.speculative_turns = None
    return handler


def test_moss_backend_is_registered_with_realtime_defaults():
    spec = TTS_BACKENDS["moss"]

    assert spec.kind == "tts"
    assert spec.normalize(MossTTSHandlerArguments()) == {
        "model_name": "OpenMOSS-Team/MOSS-TTS-Realtime",
        "codec_name": "OpenMOSS-Team/MOSS-Audio-Tokenizer",
        "model_revision": "75682787d8e2fcc73faca37ba2931453ca9c4022",
        "codec_revision": "3cd226ba2947efa357ef453bcad111b6eafba782",
        "device": "cuda",
        "dtype": "auto",
        "voice": "MOSS Voice 1",
        "blocksize": 512,
        "max_length": 3000,
        "gen_kwargs": {},
    }


def test_runtime_voice_selects_only_bundled_reference_profiles():
    handler = _handler()
    audio_output = SimpleNamespace(voice="MOSS Voice 2")
    runtime_config = SimpleNamespace(session=SimpleNamespace(audio=SimpleNamespace(output=audio_output)))
    tts_input = TTSInput.model_construct(text="Hello", runtime_config=runtime_config, response=None)

    assert handler._resolve_voice(tts_input) == "MOSS Voice 2"
    assert set(MOSS_VOICE_FILES) == {"MOSS Voice 1", "MOSS Voice 2"}

    audio_output.voice = "Aiden"
    assert handler._resolve_voice(tts_input) == "MOSS Voice 2"


def test_voice_prompts_use_existing_audio_loader_without_torchcodec(monkeypatch):
    loaded = []

    class FakeCodec:
        def encode(self, waveform, *, chunk_duration):
            assert waveform.shape == (1, 1, 240)
            assert chunk_duration == 0.24
            return {"audio_codes": torch.zeros((1, 1, 2, 16), dtype=torch.int64)}

    monkeypatch.setattr(
        "speech_to_speech.TTS.moss_tts_handler.librosa.load",
        lambda path, *, sr, mono: (loaded.append((path.name, sr, mono)) or np.zeros(240, dtype=np.float32), sr),
    )
    handler = _handler()
    handler.device = torch.device("cpu")
    handler.codec = FakeCodec()

    encoded = handler._encode_voice_prompts()

    assert set(encoded) == set(MOSS_VOICE_FILES)
    assert loaded == [("moss_voice_1.mp3", 24000, True), ("moss_voice_2.mp3", 24000, True)]


def test_causal_mask_adapter_translates_legacy_keywords():
    received = {}

    def current_mask_function(*, inputs_embeds, attention_mask):
        received.update(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return "mask"

    compatibility_mask = _accept_legacy_input_embeds(current_mask_function)

    assert (
        compatibility_mask(
            input_embeds="embeddings",
            attention_mask="attention",
            cache_position="cache position",
        )
        == "mask"
    )
    assert received == {"inputs_embeds": "embeddings", "attention_mask": "attention"}


def test_end_of_response_closes_audio_response():
    assert list(_handler().process(EndOfResponse())) == [AUDIO_RESPONSE_DONE]
