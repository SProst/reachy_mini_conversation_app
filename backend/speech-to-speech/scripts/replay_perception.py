"""Replay 16 kHz PCM WAV or synthetic silence; record inference time and memory, never audio."""

import argparse
import json
import os
import time
import wave
from pathlib import Path

import numpy as np
import psutil

from speech_to_speech.perception.native import MODEL_REVISION, RUNTIME_REVISION, NativeModel
from speech_to_speech.perception.scene import Scene, attribute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library", default=os.getenv("NEMO_DIAR_LIBRARY"), required=not os.getenv("NEMO_DIAR_LIBRARY")
    )
    parser.add_argument("--model", default=os.getenv("NEMO_DIAR_MODEL"), required=not os.getenv("NEMO_DIAR_MODEL"))
    parser.add_argument("--gpu", type=int, default=-1)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wav", type=Path)
    source.add_argument("--silence-seconds", type=float)
    parser.add_argument(
        "--labels", type=Path, help="JSON turns: start_sample, end_sample, expected_speaker (1..8 or mixed)"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.wav:
        with wave.open(str(args.wav), "rb") as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
                parser.error("WAV must be mono PCM16 at 16 kHz; preserve the reference sample clock")
            pcm = wav.readframes(wav.getnframes())
    else:
        if not 0 < args.silence_seconds <= 3600:
            parser.error("Silence duration must be in (0, 3600]")
        pcm = bytes(round(args.silence_seconds * 16000) * 2)
    process = psutil.Process()
    started = time.perf_counter()
    model = NativeModel(args.library, args.model, args.gpu)
    load_s = time.perf_counter() - started
    stream = model.open()
    scene = Scene("replay", frame_samples=model.frame_samples)
    timings, memory = [], []
    segments = []
    try:
        for offset in range(0, len(pcm), 10240):
            started = time.perf_counter()
            stream.push(pcm[offset : offset + 10240])
            snapshot = stream.snapshot()
            if snapshot is not None:
                scene.update(*snapshot)
                segments = snapshot[2]
            timings.append(time.perf_counter() - started)
            memory.append(process.memory_info().rss)
        stream.finish()
        snapshot = stream.snapshot()
        if snapshot is not None:
            scene.update(*snapshot)
            segments = snapshot[2]
    finally:
        stream.close()
        model.close()
    report = dict(
        runtime_revision=RUNTIME_REVISION,
        model_revision=MODEL_REVISION,
        stimulus="wav" if args.wav else "synthetic_silence",
        audio_seconds=len(pcm) / 32000,
        model_load_seconds=load_s,
        push_and_snapshot_p50_ms=float(np.percentile(timings, 50) * 1000),
        push_and_snapshot_p95_ms=float(np.percentile(timings, 95) * 1000),
        processing_realtime_factor=sum(timings) / (len(pcm) / 32000),
        peak_process_rss_bytes=max(memory),
        end_process_rss_bytes=memory[-1],
        network_latency_ms=None,
        added_conversational_latency_ms=None,
        gpu_memory_bytes=None,
        gaze_p95_ms=None,
        speaker_confusion=None,
        false_speaker_switches=None,
        notes="Process RSS includes loaded model and replay input. No network, ASR, GPU allocator, or robot measurements.",
    )
    if args.labels:
        turns = json.loads(args.labels.read_text(encoding="utf-8"))
        counts = {"clear": 0, "clear_correct": 0, "overlap": 0, "overlap_correct": 0}
        for turn in turns:
            mode, speaker = attribute(segments, turn["start_sample"], turn["end_sample"])
            predicted = speaker if mode == "single" else mode
            group = "overlap" if turn["expected_speaker"] == "mixed" else "clear"
            counts[group] += 1
            counts[group + "_correct"] += predicted == turn["expected_speaker"]
        report.update(counts)
        report["clear_turn_accuracy"] = counts["clear_correct"] / counts["clear"] if counts["clear"] else None
        report["label_note"] = (
            "Expected speaker numbers must use first-arrival order; no voice identity matching is performed."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
