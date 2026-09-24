"""Streaming MOSS-TTS-Realtime backend."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Iterator
from importlib import import_module
from pathlib import Path
from threading import Event
from typing import Any

import librosa
import numpy as np
import torch
from scipy.signal import resample_poly
from transformers import AutoModel, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

logger = logging.getLogger(__name__)

MOSS_VOICE_FILES = {
    "MOSS Voice 1": "moss_voice_1.mp3",
    "MOSS Voice 2": "moss_voice_2.mp3",
}


def _accept_legacy_input_embeds(mask_function: Callable[..., Any]) -> Callable[..., Any]:
    def create_causal_mask(
        *args: Any,
        input_embeds: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        kwargs["inputs_embeds"] = inputs_embeds if inputs_embeds is not None else input_embeds
        kwargs.pop("cache_position", None)
        return mask_function(*args, **kwargs)

    return create_causal_mask


class MossTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """Generate 16 kHz speech with MOSS-TTS-Realtime."""

    def setup(
        self,
        should_listen: Event,
        model_name: str = "OpenMOSS-Team/MOSS-TTS-Realtime",
        codec_name: str = "OpenMOSS-Team/MOSS-Audio-Tokenizer",
        model_revision: str = "75682787d8e2fcc73faca37ba2931453ca9c4022",
        codec_revision: str = "3cd226ba2947efa357ef453bcad111b6eafba782",
        device: str = "cuda",
        dtype: str = "auto",
        voice: str = "MOSS Voice 1",
        blocksize: int = 512,
        max_length: int = 3000,
        gen_kwargs: dict[str, Any] | None = None,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
    ) -> None:
        self.should_listen = should_listen
        self.device = torch.device(device)
        self.voice = voice if voice in MOSS_VOICE_FILES else "MOSS Voice 1"
        self.blocksize = blocksize
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns

        if dtype == "auto":
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            torch_dtype = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }[dtype]

        logger.info("Loading MOSS TTS model %s on %s", model_name, self.device)
        model_class: Any = get_class_from_dynamic_module(
            "modeling_mossttsrealtime.MossTTSRealtime",
            model_name,
            revision=model_revision,
        )
        processor_class = get_class_from_dynamic_module(
            "processing_mossttsrealtime.MossTTSRealtimeProcessor",
            model_name,
            revision=model_revision,
        )
        streaming_module = "streaming_mossttsrealtime"
        inference_class = get_class_from_dynamic_module(
            f"{streaming_module}.MossTTSRealtimeInference",
            model_name,
            revision=model_revision,
        )
        self._session_class = get_class_from_dynamic_module(
            f"{streaming_module}.MossTTSRealtimeStreamingSession",
            model_name,
            revision=model_revision,
        )
        self._decoder_class = get_class_from_dynamic_module(
            f"{streaming_module}.AudioStreamDecoder",
            model_name,
            revision=model_revision,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_revision)
        self.processor = processor_class(self.tokenizer)
        self.model = model_class.from_pretrained(
            model_name,
            attn_implementation="sdpa",
            torch_dtype=torch_dtype,
            revision=model_revision,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        local_transformer_module: Any = import_module(self.model.local_transformer.model.__class__.__module__)
        mask_function = local_transformer_module.create_causal_mask
        if "input_embeds" not in inspect.signature(mask_function).parameters:
            local_transformer_module.create_causal_mask = _accept_legacy_input_embeds(mask_function)
            logger.info("Enabled MOSS compatibility with the Transformers causal-mask API")
        self.codec = (
            AutoModel.from_pretrained(
                codec_name,
                revision=codec_revision,
                trust_remote_code=True,
            )
            .eval()
            .to(self.device)
        )
        self.inferencer = inference_class(self.model, self.tokenizer, max_length=max_length)
        self._voice_tokens = self._encode_voice_prompts()
        logger.info("MOSS TTS loaded with %d voice profiles", len(self._voice_tokens))

    def _encode_voice_prompts(self) -> dict[str, np.ndarray]:
        voice_dir = Path(__file__).with_name("moss_voices")
        encoded: dict[str, np.ndarray] = {}
        with torch.inference_mode():
            for voice, filename in MOSS_VOICE_FILES.items():
                samples, _ = librosa.load(voice_dir / filename, sr=24000, mono=True)
                waveform = torch.from_numpy(samples).unsqueeze(0)
                result = self.codec.encode(
                    waveform.unsqueeze(0).to(self.device),
                    chunk_duration=0.24,
                )
                encoded[voice] = result["audio_codes"].cpu().numpy().squeeze(1)
        return encoded

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
        if selected in MOSS_VOICE_FILES:
            self.voice = selected
        elif selected:
            logger.warning("Unknown MOSS voice %r; using %s", selected, self.voice)
        return self.voice

    def _new_session(self, voice: str) -> tuple[Any, Any]:
        self.inferencer.reset_generation_state(keep_cache=False)
        session = self._session_class(
            self.inferencer,
            self.processor,
            codec=self.codec,
            codec_sample_rate=24000,
            codec_encode_kwargs={"chunk_duration": 0.24},
            prefill_text_len=self.processor.delay_tokens_len,
            temperature=0.8,
            top_p=0.6,
            top_k=30,
            do_sample=True,
            repetition_penalty=1.1,
            repetition_window=50,
        )
        prompt_tokens = self._voice_tokens[voice]
        session.set_voice_prompt_tokens(prompt_tokens)

        system_prompt = self.processor.make_ensemble(prompt_tokens)
        assistant_ids = self.tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n")
        assistant_prefix = np.full(
            (len(assistant_ids), system_prompt.shape[1]),
            fill_value=self.processor.audio_channel_pad,
            dtype=np.int64,
        )
        assistant_prefix[:, 0] = assistant_ids
        session.reset_turn(
            input_ids=np.concatenate([system_prompt, assistant_prefix], axis=0),
            include_system_prompt=False,
            reset_cache=True,
        )
        decoder = self._decoder_class(
            self.codec,
            chunk_frames=3,
            overlap_frames=0,
            decode_kwargs={"chunk_duration": -1},
            device=self.device,
        )
        return session, decoder

    def _decode_frames(self, frames: list[torch.Tensor], session: Any, decoder: Any) -> Iterator[np.ndarray]:
        codebook_size = int(getattr(self.codec, "codebook_size", 1024))
        audio_eos_token = int(getattr(session.inferencer, "audio_eos_token", 1026))
        for frame in frames:
            tokens = frame[0] if frame.dim() == 3 else frame
            if tokens.dim() != 2:
                raise ValueError(f"Expected [T, C] audio tokens, got {tuple(tokens.shape)}")
            eos_rows = (tokens[:, 0] == audio_eos_token).nonzero(as_tuple=False)
            invalid_rows = ((tokens < 0) | (tokens >= codebook_size)).any(dim=1)
            stop_indices: list[int] = []
            if eos_rows.numel() > 0:
                stop_indices.append(int(eos_rows[0].item()))
            if invalid_rows.any():
                stop_indices.append(int(invalid_rows.nonzero(as_tuple=False)[0].item()))
            if stop_indices:
                tokens = tokens[: min(stop_indices)]
            if tokens.numel() == 0:
                continue
            decoder.push_tokens(tokens.detach())
            for waveform in decoder.audio_chunks():
                if waveform.numel() > 0:
                    yield waveform.detach().cpu().numpy().reshape(-1)

    def _generate_audio(self, text: str, voice: str) -> Iterator[np.ndarray]:
        session, decoder = self._new_session(voice)
        with (
            torch.inference_mode(),
            torch.compiler.set_stance("force_eager"),
            self.codec.streaming(batch_size=1),
        ):
            yield from self._decode_frames(session.push_text(text), session, decoder)
            yield from self._decode_frames(session.end_text(), session, decoder)
            while not session.inferencer.is_finished:
                frames = session.drain(max_steps=1)
                if not frames:
                    break
                yield from self._decode_frames(frames, session, decoder)
            final_chunk = decoder.flush()
            if final_chunk is not None and final_chunk.numel() > 0:
                yield final_chunk.detach().cpu().numpy().reshape(-1)

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
        pending = np.empty(0, dtype=np.float32)
        for waveform in self._generate_audio(tts_input.text, self._resolve_voice(tts_input)):
            if generation is not None and self.cancel_scope and self.cancel_scope.is_stale(generation):
                logger.info("MOSS TTS generation cancelled")
                return
            resampled = resample_poly(waveform.astype(np.float32), up=2, down=3)
            pending = np.concatenate([pending, resampled])
            while len(pending) >= self.blocksize:
                chunk, pending = pending[: self.blocksize], pending[self.blocksize :]
                yield (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
        if len(pending):
            pending = np.pad(pending, (0, self.blocksize - len(pending)))
            yield (np.clip(pending, -1.0, 1.0) * 32767).astype(np.int16)
