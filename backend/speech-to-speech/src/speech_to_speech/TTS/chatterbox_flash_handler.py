"""Chatterbox Flash TTS backend."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from math import gcd
from pathlib import Path
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

CHATTERBOX_VOICE_FILES = {
    "Chatterbox Voice 1": "moss_voice_1.mp3",
    "Chatterbox Voice 2": "moss_voice_2.mp3",
}
CHATTERBOX_VOICE_ALIASES = {
    "MOSS Voice 1": "Chatterbox Voice 1",
    "MOSS Voice 2": "Chatterbox Voice 2",
}


class ChatterboxFlashTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """Generate 16 kHz speech with Chatterbox Flash."""

    def setup(
        self,
        should_listen: Event,
        model_name: str = "ResembleAI/chatterbox-flash",
        device: str = "cuda",
        dtype: str = "bfloat16",
        voice: str = "Chatterbox Voice 1",
        backend: str = "torch",
        blocksize: int = 512,
        drf_block_size: int = 16,
        num_steps: int = 10,
        max_speech_tokens: int = 512,
        gen_kwargs: dict[str, Any] | None = None,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
    ) -> None:
        from chatterbox_flash import ChatterboxFlashTTS

        self.should_listen = should_listen
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self.device = device
        self.voice = self._normalize_voice(voice)
        self.backend = backend
        self.blocksize = blocksize
        self.num_steps = num_steps
        self.max_speech_tokens = max_speech_tokens
        self.gen_kwargs = gen_kwargs or {}

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]
        logger.info("Loading Chatterbox Flash model %s on %s using %s", model_name, device, backend)
        self.model = ChatterboxFlashTTS.from_pretrained(
            model_name,
            device=device,
            dtype=torch_dtype,
            drf_block_size=drf_block_size,
        )

        voice_dir = Path(__file__).with_name("moss_voices")
        self._voice_conditionals = {}
        for voice_name, filename in CHATTERBOX_VOICE_FILES.items():
            self._voice_conditionals[voice_name] = self.model.prepare_conditionals(voice_dir / filename)
        logger.info("Chatterbox Flash loaded with %d voice profiles", len(self._voice_conditionals))

        self.model.conds = self._voice_conditionals[self.voice]
        warmup_started = perf_counter()
        self.model.generate(
            "Hello.",
            backend=self.backend,
            num_steps=self.num_steps,
            max_speech_tokens=96,
            **self.gen_kwargs,
        )
        logger.info("Chatterbox Flash warmup completed in %.3fs", perf_counter() - warmup_started)

    @staticmethod
    def _normalize_voice(voice: str) -> str:
        normalized = CHATTERBOX_VOICE_ALIASES.get(voice, voice)
        return normalized if normalized in CHATTERBOX_VOICE_FILES else "Chatterbox Voice 1"

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
            if normalized != selected and selected not in CHATTERBOX_VOICE_ALIASES:
                logger.warning("Unknown Chatterbox voice %r; using %s", selected, self.voice)
            else:
                self.voice = normalized
        return self.voice

    def _generate_audio(self, text: str, voice: str) -> np.ndarray:
        self.model.conds = self._voice_conditionals[voice]
        started = perf_counter()
        waveform = self.model.generate(
            text,
            backend=self.backend,
            num_steps=self.num_steps,
            max_speech_tokens=self.max_speech_tokens,
            **self.gen_kwargs,
        )
        audio = waveform.detach().float().cpu().numpy().reshape(-1)
        elapsed = perf_counter() - started
        duration = len(audio) / int(self.model.sr)
        logger.info(
            "Chatterbox Flash generated %.3fs audio in %.3fs (RTF %.3f)",
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
        waveform = self._generate_audio(tts_input.text, self._resolve_voice(tts_input))
        if generation is not None and self.cancel_scope and self.cancel_scope.is_stale(generation):
            logger.info("Chatterbox Flash generation cancelled")
            return

        source_rate = int(self.model.sr)
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
