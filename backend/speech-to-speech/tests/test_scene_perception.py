from queue import Queue
from threading import Event
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from speech_to_speech.api.openai_realtime.pipeline_unit import PipelineUnit
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.api.openai_realtime.websocket_router import create_app
from speech_to_speech.perception.ingress import AudioIngress
from speech_to_speech.perception.scene import Scene, attribute
from speech_to_speech.perception.worker import PerceptionWorker
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.events import SpeechStartedEvent, SpeechStoppedEvent, TranscriptionCompletedEvent


@pytest.mark.parametrize(
    "segments,expected",
    [
        ([], ("unknown", None)),
        ([(0, 100, 1)], ("single", 1)),
        ([(0, 80, 1), (80, 100, 2)], ("single", 1)),
        ([(0, 79, 1), (79, 100, 2)], ("mixed", None)),
        ([(0, 100, 1), (90, 100, 2)], ("single", 1)),
        ([(0, 100, 1), (89, 100, 2)], ("mixed", None)),
        ([(0, 60, 1), (40, 100, 1)], ("single", 1)),
    ],
)
def test_whole_turn_attribution(segments, expected):
    assert attribute(segments, 0, 100) == expected


def test_turn_waits_for_diarization_and_preserves_epoch():
    scene = Scene("first")
    turn = dict(item_id="item", turn_id="turn", turn_revision=2, start_sample=0, end_sample=320)
    assert scene.turn(turn) == []
    scores = np.zeros((2, 8), dtype=np.float32)
    scores[:, 2] = 0.9
    events = scene.update(0, scores, [(0, 320, 3)])
    assert [e["type"] for e in events] == ["reachy.speaker.activity", "reachy.transcript.speaker"]
    assert events[1]["speaker"] == "speaker_3"
    assert events[1]["turn_revision"] == 2
    assert events[1]["epoch"] == "first"
    assert events[1]["revision"] > events[0]["revision"]
    assert not scene.pending
    assert Scene("second").turn(turn) == []


def test_invalid_activity_fails_instead_of_emitting_gaze_evidence():
    with pytest.raises(ValueError):
        Scene("scene").update(0, np.full((2, 8), np.nan), [])


def test_overlap_is_visible_without_separate_transcripts():
    scores = np.zeros((3, 8), dtype=np.float32)
    scores[:, :2] = 0.9
    event = Scene("scene").update(0, scores, [(0, 480, 1), (0, 480, 2)])[0]
    assert event["overlap"]
    assert event["end_sample"] == 480


def test_overflow_invalidates_scene_without_blocking():
    worker = PerceptionWorker()
    worker.epoch, worker.valid = "epoch", True
    worker.incoming = Mock()
    from queue import Full

    worker.incoming.put_nowait.side_effect = Full
    worker.audio(bytes(10240))
    (event,) = worker.poll()
    assert event.status == "unavailable"
    assert event.reason == "input_overflow"
    assert not worker.valid


def test_old_epoch_results_are_ignored():
    worker = PerceptionWorker()
    worker.epoch, worker.valid, worker.ready = "new", True, True
    worker.process = Mock()
    worker.process.is_alive.return_value = True
    worker.outgoing = Queue()
    worker.outgoing.put(dict(type="reachy.speaker.activity", epoch="old", revision=12))
    assert worker.poll() == []


def test_disabled_sessions_do_not_start_process(monkeypatch):
    monkeypatch.setenv("REACHY_DIARIZATION", "0")
    worker = PerceptionWorker()
    worker.begin("a")
    (event,) = worker.poll()
    assert event.reason == "disabled"
    assert worker.process is None


def test_ingress_uses_identical_chunks_for_vad_and_perception():
    service = RealtimeService()
    conn = service.register()
    service._state(conn).perception_enabled = True
    service.perception = Mock()
    first = service.append_pcm(conn, bytes(600), 16000)
    second = service.append_pcm(conn, bytes(1448), 16000)
    assert first == []
    assert len(second) == 2
    assert [call.args[0] for call in service.perception.audio.call_args_list] == second


def test_opt_out_has_no_perception_audio():
    service = RealtimeService()
    service.perception = Mock()
    conn = service.register()
    assert len(service.append_pcm(conn, bytes(1024), 16000)) == 1
    service.perception.begin.assert_not_called()
    service.perception.audio.assert_not_called()


def test_turn_audio_clock_reaches_attribution():
    service = RealtimeService()
    service.perception = Mock()
    conn = service.register(perception=True)
    start = SpeechStartedEvent(audio_start_ms=1000, turn_id="t", turn_revision=0)
    stop = SpeechStoppedEvent(audio_end_ms=2000, duration_s=1, turn_id="t", turn_revision=0)
    service.dispatch_pipeline_event(conn, start)
    service.dispatch_pipeline_event(conn, stop)
    final = TranscriptionCompletedEvent(transcript="hello", turn_id="t", turn_revision=0)
    service.dispatch_pipeline_event(conn, final)
    payload = service.perception.turn.call_args.args[0]
    assert (payload["start_sample"], payload["end_sample"]) == (16000, 32000)
    assert payload["item_id"]
    assert final.start_sample == 16000


@pytest.mark.parametrize("query,expected", [("", False), ("?perception=0", False), ("?perception=1", True)])
def test_websocket_negotiation_preserves_standard_first_event(monkeypatch, query, expected):
    monkeypatch.setenv("REACHY_DIARIZATION", "0")
    service = RealtimeService()
    unit = PipelineUnit(
        index=0,
        service=service,
        cancel_scope=CancelScope(),
        should_listen=Event(),
        response_playing=Event(),
        input_queue=Queue(),
        output_queue=Queue(),
        text_output_queue=Queue(),
        text_prompt_queue=Queue(),
        handlers=[],
    )
    with TestClient(create_app([unit], Event())) as client:
        with client.websocket_connect("/v1/realtime" + query) as ws:
            created = ws.receive_json()
            assert created["type"] == "session.created"
            assert service._state(created["session"]["id"]).perception_enabled is expected
            if expected:
                event = ws.receive_json()
                assert event["type"] == "reachy.perception.status"
                assert event["reason"] == "disabled"


@pytest.mark.parametrize("rate", [16000, 24000, 44100, 48000])
def test_resampling_is_invariant_to_packet_boundaries(rate):
    pcm = np.random.default_rng(2).integers(-12000, 12000, rate * 2, dtype=np.int16).tobytes()
    whole = AudioIngress().push(pcm, rate)
    split = AudioIngress()
    pieces = [split.push(pcm[i : i + 731], rate) for i in range(0, len(pcm), 731)]
    assert b"".join(pieces) == whole
    assert split.samples == len(whole) // 2
    # Streaming filter retains a bounded tail; it does not round up each packet.
    assert abs(split.samples - 32000) < 2000


def test_turn_attribution_is_revised_when_segmentation_changes():
    scene = Scene("s")
    scores = np.zeros((2, 8))
    scores[:, 0] = 0.9
    scene.update(0, scores, [(0, 320, 1)])
    turn = dict(item_id="i", turn_id="t", turn_revision=2, start_sample=0, end_sample=320)
    first = scene.turn(turn)[0]
    second = scene.update(2, scores, [(0, 640, 1), (0, 320, 2)])[-1]
    assert first["attribution"] == "single"
    assert second["attribution"] == "mixed"
    assert second["revision"] > first["revision"]
    assert scene.turn({**turn, "turn_revision": 1}) == []


def test_invalid_configuration_fails_open(monkeypatch):
    monkeypatch.setenv("REACHY_DIARIZATION", "1")
    monkeypatch.setenv("NEMO_DIAR_LIBRARY", "missing")
    monkeypatch.setenv("NEMO_DIAR_MODEL", "missing")
    monkeypatch.setenv("NEMO_DIAR_GPU", "invalid")
    service = RealtimeService()
    conn = service.register(perception=True)
    assert service.perception.poll()[0].reason == "startup_failed"
    assert len(service.append_pcm(conn, bytes(1024), 16000)) == 1
    assert service.dispatch_pipeline_event(conn, SpeechStartedEvent(audio_start_ms=0))


def test_sessions_and_ingress_clocks_are_independent(monkeypatch):
    monkeypatch.setenv("REACHY_DIARIZATION", "0")
    first, second = RealtimeService(), RealtimeService()
    a, b = first.register(perception=True), second.register(perception=True)
    first.append_pcm(a, bytes(2048), 16000)
    assert first._state(a).audio_ingress.samples == 1024
    assert second._state(b).audio_ingress.samples == 0
    first.unregister(a)
    c = first.register(perception=True)
    assert c not in (a, b)
    assert first._state(c).audio_ingress.samples == 0


def test_activity_gap_invalidates_evidence():
    with pytest.raises(ValueError, match="timeline"):
        Scene("s").update(5, np.zeros((2, 8)), [])


def test_alternating_speakers_within_chunk_is_not_stable_evidence():
    scores = np.zeros((4, 8))
    scores[:2, 0] = 0.9
    scores[2:, 1] = 0.9
    event = Scene("s").update(0, scores, [])[0]
    assert not event["overlap"]
    assert "single_speaker" not in event


def test_pending_turn_expires_without_inventing_attribution():
    scene = Scene("s")
    scene.turn(dict(item_id="i", start_sample=0, end_sample=99999))
    (event,) = scene.expire(float("inf"))
    assert event["attribution"] == "unknown"
    assert not scene.pending
    assert not scene.deadlines
