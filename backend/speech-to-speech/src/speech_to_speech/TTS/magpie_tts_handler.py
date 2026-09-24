"""NVIDIA MagpieTTS backend."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from math import gcd
from threading import Event
from time import perf_counter
from typing import Any

import numpy as np
import torch
from scipy.signal import resample_poly

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

logger = logging.getLogger(__name__)

MAGPIE_VOICES = {
    "Aria": 0,
    "Jason": 1,
    "John": 2,
    "Leo": 3,
    "Sofia": 4,
}
MAGPIE_LANGUAGES = {
    "ar": "ar-MSA",
    "de": "de",
    "en": "en",
    "es": "es",
    "fr": "fr",
    "hi": "hi",
    "it": "it",
    "ja": "ja",
    "ko": "ko",
    "pt": "pt-BR",
    "vi": "vi",
    "zh": "zh",
}


class MagpieTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """Generate 16 kHz speech with NVIDIA MagpieTTS."""

    def setup(
        self,
        should_listen: Event,
        model_name: str = "nvidia/magpie_tts_multilingual_357m",
        checkpoint_filename: str = "magpie_tts_multilingual_357m.nemo",
        revision: str = "452ef560f972c38d5fc16476259aac9456453547",
        codec_model_name: str = "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps",
        device: str = "cuda",
        voice: str = "Aria",
        language: str = "en",
        apply_text_normalization: bool = False,
        use_cfg: bool = True,
        blocksize: int = 512,
        gen_kwargs: dict[str, Any] | None = None,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
    ) -> None:
        from huggingface_hub import hf_hub_download

        self.should_listen = should_listen
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self.device = device
        self.voice = self._normalize_voice(voice)
        self.language = self._normalize_language(language, fallback="en")
        self.apply_text_normalization = apply_text_normalization
        self.use_cfg = use_cfg
        self.blocksize = blocksize
        self.gen_kwargs = gen_kwargs or {}

        logger.info("Downloading MagpieTTS checkpoint %s at %s", model_name, revision)
        checkpoint_path = hf_hub_download(repo_id=model_name, filename=checkpoint_filename, revision=revision)
        logger.info("Importing NVIDIA NeMo Speech for MagpieTTS")
        from nemo.collections.tts.modules.magpietts_inference.utils import ModelLoadConfig, load_magpie_model

        model_config = ModelLoadConfig(
            nemo_file=checkpoint_path,
            codecmodel_path=codec_model_name,
            legacy_codebooks=False,
            legacy_text_conditioning=False,
            hparams_from_wandb=None,
        )
        logger.info("Loading MagpieTTS model %s on %s", model_name, device)
        self.model, _ = load_magpie_model(model_config)
        self.model.eval().to(device)

        warmup_started = perf_counter()
        self.model.do_tts(
            "Hello.",
            language=self.language,
            apply_TN=False,
            use_cfg=self.use_cfg,
            speaker_index=MAGPIE_VOICES[self.voice],
        )
        logger.info("MagpieTTS warmup completed in %.3fs", perf_counter() - warmup_started)

    @staticmethod
    def _normalize_voice(voice: str) -> str:
        voice_by_lowercase = {candidate.lower(): candidate for candidate in MAGPIE_VOICES}
        return voice_by_lowercase.get(voice.strip().lower(), "Aria")

    @staticmethod
    def _normalize_language(language: str, fallback: str) -> str:
        normalized = language.strip().replace("_", "-").lower()
        base_language = normalized.split("-", maxsplit=1)[0]
        resolved = MAGPIE_LANGUAGES.get(base_language)
        if resolved is not None:
            return resolved
        fallback_base = fallback.strip().replace("_", "-").lower().split("-", maxsplit=1)[0]
        return MAGPIE_LANGUAGES.get(fallback_base, "en")

    @staticmethod
    def _prepare_text(text: str) -> str:
        prepared = text.strip()
        if prepared and not prepared.endswith((".", "?", "!")):
            prepared = f"{prepared}."
        return prepared

    def _resolve_voice(self, tts_input: TTSInput) -> str:
        selected: str | None = None
        response = tts_input.response
        if response and response.audio and response.audio.output and response.audio.output.voice:
            selected = str(response.audio.output.voice)
        if not selected and tts_input.runtime_config:
            audio_config = tts_input.runtime_config.session.audio
            audio_output = audio_config.output if audio_config is not None else None
            if audio_output is not None and audio_output.voice:
                selected = str(audio_output.voice)
        if selected:
            normalized = self._normalize_voice(selected)
            if normalized == "Aria" and selected.strip().lower() != "aria":
                logger.warning("Unknown MagpieTTS voice %r; using %s", selected, self.voice)
            else:
                self.voice = normalized
        return self.voice

    def _generate_audio(self, text: str, language: str, voice: str) -> np.ndarray:
        started = perf_counter()
        with torch.inference_mode():
            waveform, waveform_length = self.model.do_tts(
                text,
                language=language,
                apply_TN=self.apply_text_normalization,
                use_cfg=self.use_cfg,
                speaker_index=MAGPIE_VOICES[voice],
                **self.gen_kwargs,
            )
        audio_length = int(waveform_length[0].item())
        audio = waveform[0, :audio_length].detach().float().cpu().numpy()
        elapsed = perf_counter() - started
        duration = len(audio) / int(self.model.sample_rate)
        logger.info(
            "MagpieTTS generated %.3fs audio in %.3fs (RTF %.3f)",
            duration,
            elapsed,
            elapsed / duration if duration else 0.0,
        )
        return audio

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        if isinstance(tts_input, EndOfResponse):
            if self.speculative_turns and not self.speculative_turns.is_latest_after_reopen_grace(
                tts_input.turn_id,
                tts_input.turn_revision,
            ):
                if tts_input.response_key is None:
                    return
                tts_input.cleanup_only = True
            yield AUDIO_RESPONSE_DONE
            return

        if self.speculative_turns and not self.speculative_turns.is_latest_after_reopen_grace(
            tts_input.turn_id,
            tts_input.turn_revision,
        ):
            logger.debug("Dropping stale TTS input for turn=%s rev=%s", tts_input.turn_id, tts_input.turn_revision)
            return
        if self.speculative_turns:
            self.speculative_turns.commit(tts_input.turn_id, tts_input.turn_revision)

        generation = self.cancel_scope.generation if self.cancel_scope else None
        language = self._normalize_language(tts_input.language_code or self.language, fallback=self.language)
        waveform = self._generate_audio(self._prepare_text(tts_input.text), language, self._resolve_voice(tts_input))
        if generation is not None and self.cancel_scope and self.cancel_scope.is_stale(generation):
            logger.info("MagpieTTS generation cancelled")
            return

        source_rate = int(self.model.sample_rate)
        common_divisor = gcd(16000, source_rate)
        waveform = resample_poly(
            waveform,
            up=16000 // common_divisor,
            down=source_rate // common_divisor,
        )
        audio = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)
        for offset in range(0, len(audio), self.blocksize):
            chunk = audio[offset : offset + self.blocksize]
            if len(chunk) < self.blocksize:
                chunk = np.pad(chunk, (0, self.blocksize - len(chunk)))
            yield chunk
