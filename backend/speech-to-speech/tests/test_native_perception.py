"""Optional real C ABI/worker tests; opt in with model and library environment paths."""

import os
import time

import pytest

from speech_to_speech.perception.worker import PerceptionWorker

pytestmark = pytest.mark.skipif(
    not os.getenv("NEMO_DIAR_LIBRARY") or not os.getenv("NEMO_DIAR_MODEL"),
    reason="Pinned native library/model paths not configured",
)


def wait_for(worker, predicate, timeout=15):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        for event in worker.poll():
            assert event.status != "unavailable", event.reason
            if predicate(event):
                return event
        time.sleep(0.01)
    pytest.fail("Native worker timed out")


def test_native_worker_reuses_model_but_resets_stream(monkeypatch):
    monkeypatch.setenv("REACHY_DIARIZATION", "1")
    monkeypatch.setenv("NEMO_DIAR_GPU", "-1")
    worker = PerceptionWorker()
    try:
        for epoch in ("a", "b"):
            worker.begin(epoch)
            wait_for(worker, lambda e: e.status == "ready")
            pid = worker.process.pid
            if epoch == "a":
                first_pid = pid
            else:
                assert pid == first_pid
            for _ in range(3):
                worker.audio(bytes(10240))
            event = wait_for(worker, lambda e: e.type == "reachy.speaker.activity")
            assert event.epoch == epoch and event.start_sample == 0
            assert len(event.speakers) == 8 and not event.overlap
            worker.end()
    finally:
        worker.shutdown()


def test_native_two_clients_use_independent_streams(monkeypatch):
    monkeypatch.setenv("REACHY_DIARIZATION", "1")
    monkeypatch.setenv("NEMO_DIAR_GPU", "-1")
    first, second = PerceptionWorker(), PerceptionWorker()
    try:
        first.begin("first")
        second.begin("second")
        wait_for(first, lambda e: e.status == "ready")
        wait_for(second, lambda e: e.status == "ready")
        for _ in range(3):
            first.audio(bytes(10240))
        event = wait_for(first, lambda e: e.type == "reachy.speaker.activity")
        assert event.epoch == "first"
        assert second.samples == 0 and second.poll() == []
        for _ in range(3):
            second.audio(bytes(10240))
        event = wait_for(second, lambda e: e.type == "reachy.speaker.activity")
        assert event.epoch == "second" and event.start_sample == 0
    finally:
        first.shutdown()
        second.shutdown()
