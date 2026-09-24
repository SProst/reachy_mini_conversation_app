"""Continuous PCM16 resampling for the opted-in shared audio timeline."""

import numpy as np


class AudioIngress:
    def __init__(self):
        self.rate = None
        self.stream = None
        self.remainder = b""
        self.samples = 0

    def push(self, pcm: bytes, rate: int) -> bytes:
        if not 8000 <= rate <= 96000:
            raise ValueError("Unsupported input audio rate")
        if self.rate != rate:
            import soxr

            self.stream = None if rate == 16000 else soxr.ResampleStream(rate, 16000, 1, dtype="float32", quality="HQ")
            self.rate = rate
            self.remainder = b""
        pcm = self.remainder + pcm
        self.remainder = pcm[len(pcm) - len(pcm) % 2 :]
        pcm = pcm[: len(pcm) - len(pcm) % 2]
        if self.stream is not None and pcm:
            samples = self.stream.resample_chunk(np.frombuffer(pcm, dtype="<i2").astype(np.float32))
            pcm = np.clip(np.rint(samples), -32768, 32767).astype("<i2").tobytes()
        self.samples += len(pcm) // 2
        return pcm
