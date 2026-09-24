"""Standalone NeMo-Speech.cpp diarization ABI, loaded only in the inference worker."""

import ctypes as c
import hashlib
import os
from pathlib import Path

import numpy as np

RUNTIME_REVISION = "97a15afa5caa9bce5baaa86c1184103877af4101"
MODEL_REVISION = "f667ed73aee57d40cc39428eb768b4fd87a0a29e"
MODEL_SHA256 = "08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1"


class ModelConfig(c.Structure):
    _fields_ = [
        ("size", c.c_size_t),
        ("model_path", c.c_char_p),
        ("gpu", c.c_int32),
        ("preset", c.c_char_p),
        ("chunk_frames", c.c_int32),
        ("right_context_frames", c.c_int32),
        ("left_context_frames", c.c_int32),
        ("fifo_frames", c.c_int32),
        ("spkcache_frames", c.c_int32),
        ("update_period_frames", c.c_int32),
    ]


class SegmentationConfig(c.Structure):
    _fields_ = [
        ("size", c.c_size_t),
        ("onset", c.c_float),
        ("offset", c.c_float),
        ("pad_onset_sec", c.c_double),
        ("pad_offset_sec", c.c_double),
        ("min_gap_sec", c.c_double),
        ("min_duration_sec", c.c_double),
    ]


class Segment(c.Structure):
    _fields_ = [("start_time", c.c_double), ("end_time", c.c_double), ("speaker", c.c_int32)]


class NativeModel:
    def __init__(self, library_path: str, model_path: str, gpu: int = 0) -> None:
        path = Path(model_path)
        with path.open("rb") as source:
            digest = hashlib.sha256()
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
            if digest.hexdigest() != MODEL_SHA256:
                raise ValueError("Diarization model checksum does not match the pinned artifact")
        self._dll_directory = (
            os.add_dll_directory(str(Path(library_path).resolve().parent)) if os.name == "nt" else None
        )
        self.lib = c.CDLL(library_path)
        signatures = {
            "create": ([c.POINTER(ModelConfig), c.POINTER(c.c_void_p)], c.c_int),
            "destroy": ([c.c_void_p], None),
            "num_speakers": ([c.c_void_p], c.c_int32),
            "seconds_per_frame": ([c.c_void_p], c.c_double),
            "stream_open": ([c.c_void_p, c.POINTER(c.c_void_p)], c.c_int),
            "stream_push_f32": ([c.c_void_p, c.POINTER(c.c_float), c.c_size_t, c.c_int32], c.c_int),
            "stream_finish": ([c.c_void_p], c.c_int),
            "stream_close": ([c.c_void_p], None),
            "frame_count": ([c.c_void_p], c.c_int64),
            "frame_probs_start": ([c.c_void_p], c.c_int64),
            "frame_probs": ([c.c_void_p, c.POINTER(c.c_float), c.c_size_t], c.c_int),
            "segments": (
                [c.c_void_p, c.POINTER(SegmentationConfig), c.POINTER(Segment), c.c_size_t, c.POINTER(c.c_size_t)],
                c.c_int,
            ),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.lib, "nemo_speech_diar_" + name)
            function.argtypes, function.restype = args, result
        self.lib.nemo_speech_asr_last_error.argtypes = []
        self.lib.nemo_speech_asr_last_error.restype = c.c_char_p
        self.handle = c.c_void_p()
        config = ModelConfig(c.sizeof(ModelConfig), os.fsencode(path), gpu, b"v3-streaming", 3, 1, -1, 264, 264, 222)
        self.check(self.lib.nemo_speech_diar_create(c.byref(config), c.byref(self.handle)))
        if self.lib.nemo_speech_diar_num_speakers(self.handle) != 8:
            self.close()
            raise ValueError("Diarization runtime must expose eight speaker slots")
        self.frame_samples = round(self.lib.nemo_speech_diar_seconds_per_frame(self.handle) * 16000)
        if self.frame_samples != 160:
            self.close()
            raise ValueError("Expected 10 ms diarization output cadence")

    def check(self, status: int) -> None:
        if status:
            raise RuntimeError(
                (self.lib.nemo_speech_asr_last_error() or b"Diarization failed").decode(errors="replace")
            )

    def open(self) -> "NativeStream":
        return NativeStream(self)

    def close(self) -> None:
        if self.handle:
            self.lib.nemo_speech_diar_destroy(self.handle)
            self.handle = c.c_void_p()


class NativeStream:
    def __init__(self, model: NativeModel) -> None:
        self.model = model
        self.handle = c.c_void_p()
        self.last_frame = 0
        model.check(model.lib.nemo_speech_diar_stream_open(model.handle, c.byref(self.handle)))

    def push(self, pcm: bytes) -> None:
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        self.model.check(
            self.model.lib.nemo_speech_diar_stream_push_f32(
                self.handle, samples.ctypes.data_as(c.POINTER(c.c_float)), len(samples), 16000
            )
        )

    def snapshot(self):
        lib = self.model.lib
        end = lib.nemo_speech_diar_frame_count(self.handle)
        if end < self.last_frame:
            raise ValueError("Native diarization clock moved backwards")
        if end == self.last_frame:
            return None
        start = lib.nemo_speech_diar_frame_probs_start(self.handle)
        if start > self.last_frame or start < 0:
            raise ValueError("Native diarization output gap")
        scores = np.empty((end - start, 8), dtype=np.float32)
        self.model.check(
            lib.nemo_speech_diar_frame_probs(self.handle, scores.ctypes.data_as(c.POINTER(c.c_float)), scores.size)
        )
        count = c.c_size_t()
        self.model.check(lib.nemo_speech_diar_segments(self.handle, None, None, 0, c.byref(count)))
        segments = (Segment * count.value)()
        self.model.check(lib.nemo_speech_diar_segments(self.handle, None, segments, count.value, c.byref(count)))
        new_start = max(start, self.last_frame)
        self.last_frame = end
        return (
            new_start,
            scores[new_start - start :].copy(),
            [(round(s.start_time * 16000), round(s.end_time * 16000), s.speaker) for s in segments],
        )

    def finish(self) -> None:
        self.model.check(self.model.lib.nemo_speech_diar_stream_finish(self.handle))

    def close(self) -> None:
        if self.handle:
            self.model.lib.nemo_speech_diar_stream_close(self.handle)
            self.handle = c.c_void_p()
