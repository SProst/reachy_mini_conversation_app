from types import SimpleNamespace

import numpy as np

from speech_to_speech.arguments_classes.chatterbox_flash_tts_arguments import (
    ChatterboxFlashTTSHandlerArguments,
)
from speech_to_speech.backend_registry import TTS_BACKENDS
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput
from speech_to_speech.TTS.chatterbox_flash_handler import (
    CHATTERBOX_VOICE_FILES,
    ChatterboxFlashTTSHandler,
)


def _handler() -> ChatterboxFlashTTSHandler:
    handler = object.__new__(ChatterboxFlashTTSHandler)
    handler.voice = "Chatterbox Voice 1"
    handler.speculative_turns = None
    return handler


def test_chatterbox_flash_backend_is_registered():
    spec = TTS_BACKENDS["chatterbox-flash"]

    assert spec.kind == "tts"
    assert spec.normalize(ChatterboxFlashTTSHandlerArguments()) == {
        "model_name": "ResembleAI/chatterbox-flash",
        "device": "cuda",
        "dtype": "bfloat16",
        "voice": "Chatterbox Voice 1",
        "backend": "torch",
        "blocksize": 512,
        "drf_block_size": 16,
        "num_steps": 10,
        "max_speech_tokens": 512,
        "gen_kwargs": {},
    }


def test_runtime_voice_supports_chatterbox_names_and_moss_aliases():
    handler = _handler()
    audio_output = SimpleNamespace(voice="Chatterbox Voice 2")
    runtime_config = SimpleNamespace(session=SimpleNamespace(audio=SimpleNamespace(output=audio_output)))
    tts_input = TTSInput.model_construct(text="Hello", runtime_config=runtime_config, response=None)

    assert handler._resolve_voice(tts_input) == "Chatterbox Voice 2"
    assert set(CHATTERBOX_VOICE_FILES) == {"Chatterbox Voice 1", "Chatterbox Voice 2"}

    audio_output.voice = "MOSS Voice 1"
    assert handler._resolve_voice(tts_input) == "Chatterbox Voice 1"


def test_process_resamples_and_chunks_generated_audio():
    handler = _handler()
    handler.cancel_scope = None
    handler.blocksize = 512
    handler.model = SimpleNamespace(sr=24000)
    handler._generate_audio = lambda text, voice: np.linspace(-0.5, 0.5, 1200, dtype=np.float32)
    tts_input = TTSInput.model_construct(text="Hello", runtime_config=None, response=None)

    chunks = list(handler.process(tts_input))

    assert len(chunks) == 2
    assert all(chunk.shape == (512,) for chunk in chunks)
    assert all(chunk.dtype == np.int16 for chunk in chunks)


def test_end_of_response_closes_audio_response():
    assert list(_handler().process(EndOfResponse())) == [AUDIO_RESPONSE_DONE]
