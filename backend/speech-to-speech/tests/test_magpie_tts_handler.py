from types import SimpleNamespace

import numpy as np

from speech_to_speech.arguments_classes.magpie_tts_arguments import MagpieTTSHandlerArguments
from speech_to_speech.backend_registry import TTS_BACKENDS
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput
from speech_to_speech.TTS.magpie_tts_handler import MAGPIE_VOICES, MagpieTTSHandler


def _handler() -> MagpieTTSHandler:
    handler = object.__new__(MagpieTTSHandler)
    handler.voice = "Aria"
    handler.language = "en"
    handler.speculative_turns = None
    return handler


def test_magpie_backend_is_registered():
    spec = TTS_BACKENDS["magpie"]

    assert spec.kind == "tts"
    assert spec.required_extra == "magpie"
    assert spec.normalize(MagpieTTSHandlerArguments()) == {
        "model_name": "nvidia/magpie_tts_multilingual_357m",
        "checkpoint_filename": "magpie_tts_multilingual_357m.nemo",
        "revision": "452ef560f972c38d5fc16476259aac9456453547",
        "codec_model_name": "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps",
        "device": "cuda",
        "voice": "Aria",
        "language": "en",
        "apply_text_normalization": False,
        "use_cfg": True,
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


def test_language_normalization_supports_pipeline_locales():
    assert MagpieTTSHandler._normalize_language("en-US", fallback="fr") == "en"
    assert MagpieTTSHandler._normalize_language("pt-BR", fallback="en") == "pt-BR"
    assert MagpieTTSHandler._normalize_language("unknown", fallback="fr-FR") == "fr"


def test_text_is_trimmed_and_terminated_for_magpie():
    assert MagpieTTSHandler._prepare_text(" Hello ") == "Hello."
    assert MagpieTTSHandler._prepare_text("Already done!") == "Already done!"


def test_process_resamples_and_chunks_generated_audio():
    handler = _handler()
    handler.cancel_scope = None
    handler.blocksize = 512
    handler.model = SimpleNamespace(sample_rate=22050)
    handler._generate_audio = lambda text, language, voice: np.linspace(-0.5, 0.5, 1200, dtype=np.float32)
    tts_input = TTSInput.model_construct(text="Hello", language_code="en-US", runtime_config=None, response=None)

    chunks = list(handler.process(tts_input))

    assert len(chunks) == 2
    assert all(chunk.shape == (512,) for chunk in chunks)
    assert all(chunk.dtype == np.int16 for chunk in chunks)


def test_end_of_response_closes_audio_response():
    assert list(_handler().process(EndOfResponse())) == [AUDIO_RESPONSE_DONE]
