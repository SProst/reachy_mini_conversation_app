"""Speaker events and whole-turn attribution on the input sample clock."""

import time
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field


class PerceptionEvent(BaseModel):
    type: str
    epoch: str
    revision: int
    start_sample: int = 0
    end_sample: int = 0
    sample_rate: int = 16000
    status: str | None = None
    reason: str | None = None
    item_id: str | None = None
    turn_id: str | None = None
    turn_revision: int | None = None
    attribution: str | None = None
    speaker: str | None = None
    speakers: list[dict] = Field(default_factory=list)
    overlap: bool = False
    single_speaker: str | None = None


def attribute(segments: list[tuple[int, int, int]], start: int, end: int) -> tuple[str, int | None]:
    """Attribute only clear turns; duration unions prevent padded segments double counting."""
    if end <= start:
        return "unknown", None
    points = {start, end}
    clipped = [(max(a, start), min(b, end), speaker) for a, b, speaker in segments if a < end and b > start]
    for a, b, _ in clipped:
        points.update((a, b))
    coverage: dict[int, int] = {}
    speech = overlap = 0
    boundaries = sorted(points)
    for a, b in zip(boundaries, boundaries[1:]):
        active = {speaker for x, y, speaker in clipped if x < b and y > a}
        if active:
            speech += b - a
        if len(active) > 1:
            overlap += b - a
        for speaker in active:
            coverage[speaker] = coverage.get(speaker, 0) + b - a
    if not speech:
        return "unknown", None
    speaker, duration = max(coverage.items(), key=lambda pair: pair[1])
    if duration / speech >= 0.8 and overlap / speech <= 0.1:
        return "single", speaker
    return "mixed", None


@dataclass
class Scene:
    epoch: str
    frame_samples: int = 160
    revision: int = 0
    end_sample: int = 0
    segments: list[tuple[int, int, int]] = field(default_factory=list)
    pending: dict[str, dict] = field(default_factory=dict)
    attributed: dict[str, tuple[dict, tuple[str, int | None]]] = field(default_factory=dict)
    deadlines: dict[str, float] = field(default_factory=dict)

    def event(self, kind: str, **fields) -> dict:
        self.revision += 1
        return PerceptionEvent(type=kind, epoch=self.epoch, revision=self.revision, **fields).model_dump(
            exclude_none=True
        )

    def update(self, start_frame: int, scores: np.ndarray, segments: list[tuple[int, int, int]]) -> list[dict]:
        if scores.ndim != 2 or scores.shape[1] != 8 or len(scores) == 0 or not np.isfinite(scores).all():
            raise ValueError("Invalid diarization activity tensor")
        start = start_frame * self.frame_samples
        if start != self.end_sample:
            raise ValueError("Non-contiguous diarization timeline")
        self.end_sample = start + len(scores) * self.frame_samples
        # Match the native probability compaction horizon; no raw audio is retained here.
        self.segments = [s for s in segments if s[1] >= self.end_sample - 16000 * 1200]
        speakers = [
            {
                "speaker": f"speaker_{i + 1}",
                "activity": float(scores[:, i].mean()),
                "active_fraction": float((scores[:, i] >= 0.5).mean()),
            }
            for i in range(8)
        ]
        active = scores >= 0.5
        stable = [i for i in range(8) if active[:, i].all() and (active.sum(axis=1) == 1).all()]
        output = [
            self.event(
                "reachy.speaker.activity",
                start_sample=start,
                end_sample=self.end_sample,
                speakers=speakers,
                overlap=bool(((scores >= 0.5).sum(axis=1) > 1).any()),
                single_speaker=f"speaker_{stable[0] + 1}" if stable else None,
            )
        ]
        for item_id, turn in list(self.pending.items()):
            if turn["end_sample"] <= self.end_sample:
                mode, speaker = self.attribution(turn)
                output.append(
                    self.event(
                        "reachy.transcript.speaker",
                        **turn,
                        attribution=mode,
                        speaker=f"speaker_{speaker}" if speaker else None,
                    )
                )
                del self.pending[item_id]
                self.deadlines.pop(item_id, None)
                self.attributed[item_id] = (turn, (mode, speaker))
        for item_id, (turn, previous) in list(self.attributed.items()):
            if turn["end_sample"] < self.end_sample - 16000 * 1200:
                del self.attributed[item_id]
                continue
            current = self.attribution(turn)
            if current != previous:
                mode, speaker = current
                output.append(
                    self.event(
                        "reachy.transcript.speaker",
                        **turn,
                        attribution=mode,
                        speaker=f"speaker_{speaker}" if speaker else None,
                    )
                )
                self.attributed[item_id] = (turn, current)
        while len(self.attributed) > 128:
            self.attributed.pop(next(iter(self.attributed)))
        return output

    def attribution(self, turn: dict) -> tuple[str, int | None]:
        if turn["start_sample"] < max(0, self.end_sample - 16000 * 1200):
            return "unknown", None
        return attribute(self.segments, turn["start_sample"], turn["end_sample"])

    def turn(self, turn: dict) -> list[dict]:
        previous = self.pending.get(turn["item_id"]) or self.attributed.get(turn["item_id"], ({}, None))[0]
        if (previous.get("turn_revision") or 0) > (turn.get("turn_revision") or 0):
            return []
        self.attributed.pop(turn["item_id"], None)
        if len(self.pending) >= 128:
            return [self.event("reachy.transcript.speaker", **turn, attribution="unknown")]
        self.pending[turn["item_id"]] = turn
        self.deadlines[turn["item_id"]] = time.monotonic() + 5.0
        if turn["end_sample"] <= self.end_sample:
            mode, speaker = self.attribution(turn)
            self.attributed[turn["item_id"]] = (turn, (mode, speaker))
            del self.pending[turn["item_id"]]
            self.deadlines.pop(turn["item_id"], None)
            return [
                self.event(
                    "reachy.transcript.speaker",
                    **turn,
                    attribution=mode,
                    speaker=f"speaker_{speaker}" if speaker else None,
                )
            ]
        return []

    def expire(self, now: float) -> list[dict]:
        output = []
        for item, deadline in list(self.deadlines.items()):
            if now >= deadline:
                turn = self.pending.pop(item)
                self.deadlines.pop(item)
                output.append(self.event("reachy.transcript.speaker", **turn, attribution="unknown"))
        return output
