from __future__ import annotations

import logging
import re
from queue import Queue
from threading import Event, Lock, Thread
from time import perf_counter
from types import GeneratorType
from typing import Any, Iterator

import numpy as np
import torch
from transformers import AutoModelForRNNT, AutoProcessor, TextIteratorStreamer

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler

logger = logging.getLogger(__name__)

_STREAM_END = object()
_LANGUAGE_TAG = re.compile(r"<([a-z]{2}(?:-[A-Z]{2})?)>")


class _NemotronStream:
    """Own one cache-aware model generation spanning progressive VAD updates."""

    def __init__(
        self,
        model: Any,
        first_inputs: dict[str, Any],
        first_features: Any,
        streamer: TextIteratorStreamer,
        num_lookahead_tokens: int,
    ) -> None:
        self.model = model
        self.streamer = streamer
        self.feature_queue: Queue[Any] = Queue()
        self.text_parts: list[str] = []
        self.text_lock = Lock()
        self.text_updated = Event()
        self.output: Any = None
        self.error: Exception | None = None
        self._finished = False

        def input_features_generator() -> Iterator[Any]:
            yield first_features
            while True:
                features = self.feature_queue.get()
                if features is _STREAM_END:
                    return
                yield features

        features = input_features_generator()
        assert isinstance(features, GeneratorType)
        generate_kwargs = {
            **first_inputs,
            "input_features": features,
            "num_lookahead_tokens": num_lookahead_tokens,
            "return_dict_in_generate": True,
            "streamer": streamer,
        }
        self.generation_thread = Thread(
            target=self._generate,
            args=(generate_kwargs,),
            name="nemotron-asr-generation",
            daemon=True,
        )
        self.collector_thread = Thread(
            target=self._collect_text,
            name="nemotron-asr-text",
            daemon=True,
        )
        self.generation_thread.start()
        self.collector_thread.start()

    def _generate(self, generate_kwargs: dict[str, Any]) -> None:
        try:
            self.output = self.model.generate(**generate_kwargs)
        except Exception as exc:
            self.error = exc
            self.streamer.end()

    def _collect_text(self) -> None:
        for text in self.streamer:
            with self.text_lock:
                self.text_parts.append(text)
            self.text_updated.set()

    def append(self, features: Any) -> None:
        if not self._finished:
            self.feature_queue.put(features)

    def finish(self) -> None:
        if not self._finished:
            self._finished = True
            self.feature_queue.put(_STREAM_END)

    def wait(self, timeout: float = 30.0) -> None:
        self.finish()
        self.generation_thread.join(timeout)
        if self.generation_thread.is_alive():
            raise TimeoutError("Nemotron streaming generation did not finish")
        self.collector_thread.join(timeout)
        if self.collector_thread.is_alive():
            raise TimeoutError("Nemotron transcription collector did not finish")
        if self.error is not None:
            raise RuntimeError("Nemotron streaming generation failed") from self.error

    def text(self, wait_s: float = 0.0) -> str:
        if wait_s > 0:
            self.text_updated.wait(wait_s)
            self.text_updated.clear()
        with self.text_lock:
            return "".join(self.text_parts).strip()


class NemotronStreamingSTTHandler(BaseSTTHandler):
    """Transcribe speech with Nemotron 3.5 cache-aware streaming ASR."""

    def setup(
        self,
        model_name: str = "nvidia/nemotron-3.5-asr-streaming-0.6b",
        device: str = "auto",
        compute_type: str = "float16",
        language: str = "en-US",
        num_lookahead_tokens: int = 6,
        gen_kwargs: dict[str, Any] | None = None,
        enable_live_transcription: bool = False,
        live_transcription_update_interval: float = 0.5,
    ) -> None:
        if num_lookahead_tokens not in {0, 1, 6, 13}:
            raise ValueError("num_lookahead_tokens must be one of 0, 1, 6, or 13")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested for Nemotron but unavailable; falling back to CPU")
            device = "cpu"

        dtype_by_name = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        try:
            dtype = dtype_by_name[compute_type]
        except KeyError as exc:
            raise ValueError("compute_type must be 'float16', 'bfloat16', or 'float32'") from exc
        if device == "cpu" and dtype == torch.float16:
            logger.warning("float16 Nemotron inference is unsupported on CPU; using float32")
            dtype = torch.float32

        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.language = language
        self.last_language = self._language_code(language) or "en"
        self.num_lookahead_tokens = num_lookahead_tokens
        self.gen_kwargs = gen_kwargs or {}
        self.enable_live_transcription = enable_live_transcription
        self.live_transcription_update_interval = live_transcription_update_interval
        self.sample_rate = 16000
        self._stream: _NemotronStream | None = None
        self._stream_turn_key: tuple[str | None, int | None] | None = None
        self._next_mel_frame_idx = 0
        self._last_partial_text = ""

        logger.info("Loading Nemotron streaming ASR model: %s on %s", model_name, device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.processor.set_num_lookahead_tokens(num_lookahead_tokens)
        self.model = AutoModelForRNNT.from_pretrained(model_name, dtype=dtype)
        self.model.to(device)
        self.model.eval()
        logger.info(
            "Nemotron streaming ASR ready with %d ms latency",
            self.processor.streaming_latency_ms,
        )
        self.warmup()

    def warmup(self) -> None:
        """Warm the model with one second of silence."""
        logger.info("Warming up %s", self.__class__.__name__)
        try:
            self._transcribe_offline(np.zeros(self.sample_rate, dtype=np.float32))
        except Exception as exc:
            logger.warning("Nemotron warmup failed: %s", exc)

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        audio = np.asarray(vad_audio.audio, dtype=np.float32)
        turn_key = (vad_audio.turn_id, vad_audio.turn_revision)
        if turn_key != self._stream_turn_key:
            self._reset_stream()
            self._stream_turn_key = turn_key

        if self.enable_live_transcription and vad_audio.mode == "progressive":
            self._feed_stream(audio, final=False)
            if self._stream is None:
                return
            text = self._stream.text(wait_s=min(self.live_transcription_update_interval, 0.05))
            if text and text != self._last_partial_text:
                self._last_partial_text = text
                yield PartialTranscription(
                    text=text,
                    turn_id=vad_audio.turn_id,
                    turn_revision=vad_audio.turn_revision,
                )
            return

        started_at_s = perf_counter()
        try:
            if self.enable_live_transcription:
                self._feed_stream(audio, final=True)
                text, language_code = self._finish_stream()
            else:
                text, language_code = self._transcribe_offline(audio)
        except Exception:
            logger.exception(
                "Nemotron final transcription failed for turn=%s rev=%s",
                vad_audio.turn_id,
                vad_audio.turn_revision,
            )
            text = ""
            language_code = self.last_language
        finally:
            self._reset_stream()

        logger.info(
            "Nemotron final STT done turn=%s rev=%s audio=%.3fs inference=%.3fs chars=%d",
            vad_audio.turn_id,
            vad_audio.turn_revision,
            len(audio) / self.sample_rate,
            perf_counter() - started_at_s,
            len(text),
        )
        yield Transcription(
            text=text,
            language_code=language_code,
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
            speech_stopped_at_s=vad_audio.created_at_s,
        )

    def _feed_stream(self, audio: np.ndarray, *, final: bool) -> None:
        if self._stream is None:
            first_samples = self.processor.num_samples_first_audio_chunk
            if len(audio) < first_samples and not final:
                return
            first_audio = audio[:first_samples]
            if len(first_audio) < first_samples:
                first_audio = np.pad(first_audio, (0, first_samples - len(first_audio)))
            first_batch = self._streaming_inputs(first_audio, first=True)
            first_features = first_batch.pop("input_features")
            first_features = first_features[:, : self.processor.num_mel_frames_first_audio_chunk, :]
            streamer = TextIteratorStreamer(
                self.processor.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            self._stream = _NemotronStream(
                self.model,
                dict(first_batch),
                first_features,
                streamer,
                self.num_lookahead_tokens,
            )
            self._next_mel_frame_idx = self.processor.num_mel_frames_first_audio_chunk

        hop_length = self.processor.feature_extractor.hop_length
        overlap = self.processor.feature_extractor.n_fft // 2
        chunk_samples = self.processor.num_samples_per_audio_chunk
        mel_frames_per_chunk = self.processor.num_mel_frames_per_audio_chunk
        while True:
            new_audio_start = self._next_mel_frame_idx * hop_length
            chunk_start = new_audio_start - overlap
            chunk_end = chunk_start + chunk_samples
            if chunk_end > len(audio):
                break
            chunk = audio[chunk_start:chunk_end]
            self._stream.append(self._streaming_inputs(chunk, first=False)["input_features"])
            self._next_mel_frame_idx += mel_frames_per_chunk

        if final:
            new_audio_start = self._next_mel_frame_idx * hop_length
            if new_audio_start < len(audio):
                chunk_start = new_audio_start - overlap
                chunk = audio[max(0, chunk_start) :]
                if len(chunk) < chunk_samples:
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
                else:
                    chunk = chunk[:chunk_samples]
                self._stream.append(self._streaming_inputs(chunk, first=False)["input_features"])
                self._next_mel_frame_idx += mel_frames_per_chunk
            self._stream.finish()

    def _streaming_inputs(self, audio: np.ndarray, *, first: bool) -> dict[str, Any]:
        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            is_streaming=True,
            is_first_audio_chunk=first,
            language=self.language,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        return dict(inputs)

    def _finish_stream(self) -> tuple[str, str]:
        if self._stream is None:
            return "", self.last_language
        stream = self._stream
        stream.wait()
        if stream.output is None:
            raise RuntimeError("Nemotron stream completed without generation output")
        return self._decode_output(stream.output)

    def _transcribe_offline(self, audio: np.ndarray) -> tuple[str, str]:
        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            language=self.language,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                return_dict_in_generate=True,
                **self.gen_kwargs,
            )
        return self._decode_output(output)

    def _decode_output(self, output: Any) -> tuple[str, str]:
        decoded = self.processor.decode(output.sequences, skip_special_tokens=True)
        text = (decoded[0] if isinstance(decoded, list) else decoded).strip()
        language_code = self._language_code(self.language)
        if language_code is None:
            tagged_text = self.processor.decode(output.sequences, skip_special_tokens=False)
            if isinstance(tagged_text, list):
                tagged_text = tagged_text[0]
            match = _LANGUAGE_TAG.search(tagged_text)
            language_code = self._language_code(match.group(1)) if match else self.last_language
        if language_code:
            self.last_language = language_code
        return text, self.last_language

    @staticmethod
    def _language_code(language: str) -> str | None:
        if language == "auto":
            return None
        return language.split("-", 1)[0].lower()

    def _reset_stream(self) -> None:
        if self._stream is not None and self._stream.generation_thread.is_alive():
            try:
                self._stream.wait(timeout=5.0)
            except (RuntimeError, TimeoutError):
                logger.exception("Failed to close abandoned Nemotron stream")
        self._stream = None
        self._stream_turn_key = None
        self._next_mel_frame_idx = 0
        self._last_partial_text = ""

    def cleanup(self) -> None:
        """Release stream and model resources."""
        self._reset_stream()
        if hasattr(self, "model"):
            del self.model

    def on_session_end(self) -> None:
        super().on_session_end()
        self._reset_stream()
