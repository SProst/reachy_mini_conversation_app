"""Streaming NVIDIA MagpieTTS backend powered by NeMo-Speech.cpp."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from threading import Event
from time import perf_counter
from typing import Any

import numpy as np

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.TTS.nemo_speech_cpp import NeMoSpeechTTSRuntime

logger = logging.getLogger(__name__)

MAGPIE_VOICES = {
    "Aria": 0,
    "Jason": 1,
    "John": 2,
    "Leo": 3,
    "Sofia": 4,
}
MAGPIE_LANGUAGES = {
    "de": "de-DE",
    "en": "en-US",
    "es": "es-ES",
    "fr": "fr-FR",
    "hi": "hi-IN",
    "it": "it-IT",
    "vi": "vi-VN",
}


class MagpieTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """Stream 16 kHz MagpieTTS speech through NeMo-Speech.cpp."""

    runtime: NeMoSpeechTTSRuntime | None = None

    def setup(
        self,
        should_listen: Event,
        library_path: str = "libnemo_speech_tts.so",
        model_path: str = "",
        codec_path: str = "",
        tokenizer_path: str = "",
        device: str = "cuda",
        voice: str = "Aria",
        language: str = "en-US",
        threads: int = 4,
        codec_threads: int = 0,
        chunk_frames: int = 3,
        codec_queue_depth: int = 4,
        codec_history_frames: int = -1,
        codec_future_frames: int = 1,
        window_ms: int = 0,
        use_cfg: bool = True,
        use_kv_cache: bool = True,
        use_stateful_codec: bool = True,
        codec_cpu: bool = False,
        flush_partial_chunk: bool = True,
        verbose: bool = False,
        blocksize: int = 512,
        gen_kwargs: dict[str, Any] | None = None,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
    ) -> None:
        self.should_listen = should_listen
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self.voice = self._normalize_voice(voice)
        self.language = self._normalize_language(language, fallback="en-US")
        if blocksize <= 0:
            raise ValueError("MagpieTTS blocksize must be positive")
        self.blocksize = blocksize
        if gen_kwargs:
            logger.warning("NeMo-Speech.cpp MagpieTTS ignores unsupported generation options: %s", sorted(gen_kwargs))

        logger.info("Loading MagpieTTS v2602 with NeMo-Speech.cpp on %s", device)
        self.runtime = NeMoSpeechTTSRuntime(
            library_path=library_path,
            model_path=model_path,
            codec_path=codec_path,
            tokenizer_path=tokenizer_path,
            device=device,
            voice=self.voice,
            language=self.language,
            threads=threads,
            codec_threads=codec_threads,
            chunk_frames=chunk_frames,
            codec_queue_depth=codec_queue_depth,
            codec_history_frames=codec_history_frames,
            codec_future_frames=codec_future_frames,
            window_ms=window_ms,
            use_cfg=use_cfg,
            use_kv_cache=use_kv_cache,
            use_stateful_codec=use_stateful_codec,
            codec_cpu=codec_cpu,
            flush_partial_chunk=flush_partial_chunk,
            verbose=verbose,
        )
        logger.info(
            "Loaded %s at %d Hz with voices: %s",
            self.runtime.version,
            self.runtime.sample_rate,
            ", ".join(self.runtime.voices),
        )

        warmup_started = perf_counter()
        for _ in self.runtime.stream(
            "Hello.",
            self.language,
            self.voice,
            16000,
            lambda: False,
        ):
            pass
        logger.info("NeMo-Speech.cpp MagpieTTS warmup completed in %.3fs", perf_counter() - warmup_started)

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
        return MAGPIE_LANGUAGES.get(fallback_base, "en-US")

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

        runtime = self.runtime
        if runtime is None:
            raise RuntimeError("NeMo-Speech.cpp TTS runtime is not initialized")
        generation = self.cancel_scope.generation if self.cancel_scope else None

        def cancelled() -> bool:
            return self.stop_event.is_set() or (
                generation is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(generation)
            )

        language = self._normalize_language(tts_input.language_code or self.language, fallback=self.language)
        pending = bytearray()
        block_bytes = self.blocksize * np.dtype(np.int16).itemsize
        for pcm in runtime.stream(
            self._prepare_text(tts_input.text),
            language,
            self._resolve_voice(tts_input),
            16000,
            cancelled,
        ):
            if cancelled():
                logger.info("NeMo-Speech.cpp MagpieTTS generation cancelled")
                return
            pending.extend(pcm)
            while len(pending) >= block_bytes:
                chunk = bytes(pending[:block_bytes])
                del pending[:block_bytes]
                yield np.frombuffer(chunk, dtype="<i2").astype(np.int16, copy=False)

        if cancelled():
            logger.info("NeMo-Speech.cpp MagpieTTS generation cancelled")
            return
        if pending:
            pending.extend(b"\0" * (block_bytes - len(pending)))
            yield np.frombuffer(bytes(pending), dtype="<i2").astype(np.int16, copy=False)

        stats = runtime.last_stats
        if stats is not None:
            logger.info(
                "NeMo-Speech.cpp MagpieTTS generated %.3fs audio in %.3fs (RTF %.3f, TTFA %.0fms, %d streamed chunks)",
                stats.audio_s,
                stats.elapsed_s,
                stats.rtf,
                stats.e2e_ttfa_ms,
                stats.e2e_chunks,
            )

    def cleanup(self) -> None:
        if self.runtime is not None:
            self.runtime.close()
            self.runtime = None
