"""Bounded, process-isolated inference; native TTS and diarization never share a loader."""

import logging
import multiprocessing as mp
import os
import time
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from queue import Empty, Full
from threading import Thread

from speech_to_speech.perception.scene import PerceptionEvent, Scene

logger = logging.getLogger(__name__)


def run_worker(incoming, outgoing, options: dict) -> None:
    # Only this subprocess imports/loads the native library.
    from speech_to_speech.perception.native import NativeModel

    model = stream = scene = None
    try:
        while True:
            if scene is not None:
                for event in scene.expire(time.monotonic()):
                    outgoing.put_nowait(event)
            try:
                command, epoch, payload = incoming.get(timeout=0.25)
            except Empty:
                continue
            if command == "shutdown":
                break
            if command == "begin":
                if stream is not None:
                    stream.close()
                scene = Scene(epoch)
                if model is None:
                    model = NativeModel(**options)
                stream = model.open()
                scene.frame_samples = model.frame_samples
                outgoing.put_nowait(scene.event("reachy.perception.status", status="ready"))
            elif scene is not None and scene.epoch == epoch:
                assert stream is not None
                if command == "end":
                    stream.close()
                    stream = scene = None
                elif command == "audio":
                    stream.push(payload)
                    snapshot = stream.snapshot()
                    if snapshot is not None:
                        for event in scene.update(*snapshot):
                            outgoing.put_nowait(event)
                elif command == "turn":
                    for event in scene.turn(payload):
                        outgoing.put_nowait(event)
    except Exception:
        logger.exception("Diarization worker stopped")
        if scene is not None:
            try:
                outgoing.put_nowait(
                    scene.event(
                        "reachy.perception.status",
                        status="unavailable",
                        reason="worker_failed",
                        end_sample=scene.end_sample,
                    )
                )
            except Full:
                pass  # The parent detects process exit and invalidates the scene.
    finally:
        if stream is not None:
            stream.close()
        if model is not None:
            model.close()


class PerceptionWorker:
    """One worker per pipeline unit; independent native stream state for each client."""

    def __init__(self) -> None:
        self.process: BaseProcess | None = None
        self.incoming: Queue | None = None
        self.outgoing: Queue | None = None
        self.epoch: str | None = None
        self.pending: list[PerceptionEvent] = []
        self.revision = 0
        self.samples = 0
        self.valid = False
        self.ready = False
        self.last_progress = 0.0
        self.processed_samples = 0
        self.audio_buffer = bytearray()

    def begin(self, epoch: str) -> None:
        self.epoch, self.samples, self.processed_samples, self.revision = epoch, 0, 0, 0
        self.pending.clear()
        self.audio_buffer.clear()
        self.ready = False
        self.last_progress = time.monotonic()
        if os.getenv("REACHY_DIARIZATION", "0") != "1":
            self.fail("disabled")
            return
        library = os.getenv("NEMO_DIAR_LIBRARY", "")
        model = os.getenv("NEMO_DIAR_MODEL", "")
        if not library or not model:
            self.fail("not_configured")
            return
        try:
            if self.process is None or not self.process.is_alive():
                context = mp.get_context("spawn")
                self.incoming, self.outgoing = context.Queue(64), context.Queue(64)
                options = {"library_path": library, "model_path": model, "gpu": int(os.getenv("NEMO_DIAR_GPU", "0"))}
                self.process = context.Process(
                    target=run_worker, args=(self.incoming, self.outgoing, options), daemon=True
                )
                self.process.start()
        except Exception:
            logger.exception("Cannot start diarization worker")
            self.fail("startup_failed")
            return
        self.valid = True
        self._send("begin", None)
        if self.valid:
            self.pending.append(
                PerceptionEvent(type="reachy.perception.status", epoch=epoch, revision=0, status="loading")
            )

    def _send(self, command: str, payload) -> None:
        if not self.valid:
            return
        try:
            if self.incoming is None:
                self.fail("queue_failed")
                return
            self.incoming.put_nowait((command, self.epoch, payload))
        except Full:
            self.fail("input_overflow")
        except (OSError, ValueError):
            self.fail("queue_failed")

    def audio(self, pcm: bytes) -> None:
        self.samples += len(pcm) // 2
        if not self.valid:
            return
        self.audio_buffer.extend(pcm)
        # 64 bounded 320 ms packets give cold model loading at most 20.48 s of audio.
        while len(self.audio_buffer) >= 10240 and self.valid:
            packet = bytes(self.audio_buffer[:10240])
            del self.audio_buffer[:10240]
            self._send("audio", packet)

    def turn(self, payload: dict) -> None:
        self._send("turn", payload)

    def fail(self, reason: str) -> None:
        self.valid = False
        self.audio_buffer.clear()
        self.revision += 1
        self.pending = [
            PerceptionEvent(
                type="reachy.perception.status",
                epoch=self.epoch or "",
                revision=self.revision,
                end_sample=self.samples,
                status="unavailable",
                reason=reason,
            )
        ]
        self.shutdown()

    def poll(self) -> list[PerceptionEvent]:
        if self.valid:
            if self.outgoing is None or self.process is None:
                self.fail("worker_exited")
                result, self.pending = self.pending, []
                return result
            try:
                for _ in range(64):
                    event = PerceptionEvent.model_validate(self.outgoing.get_nowait())
                    if event.epoch != self.epoch:
                        continue
                    self.revision = max(self.revision, event.revision)
                    self.processed_samples = max(self.processed_samples, event.end_sample)
                    self.last_progress = time.monotonic()
                    self.ready |= event.status == "ready"
                    if event.status == "unavailable":
                        self.fail(event.reason or "worker_failed")
                        break
                    self.pending.append(event)
            except Empty:
                pass
            except Exception:
                logger.exception("Invalid diarization worker output")
                self.fail("invalid_worker_output")
            if self.valid and not self.process.is_alive():
                self.fail("worker_exited")
            timeout = 5.0 if self.ready else 60.0
            if (
                self.valid
                and (not self.ready or self.samples - self.processed_samples > 16000)
                and (time.monotonic() - self.last_progress > timeout)
            ):
                self.fail("worker_timeout")
        result, self.pending = self.pending, []
        return result

    def end(self) -> None:
        self._send("end", None)
        self.epoch = None
        self.valid = False
        self.pending.clear()

    def shutdown(self) -> None:
        if self.process is not None:
            # terminate() is non-blocking and affects only the process created here.
            if self.process.is_alive():
                self.process.terminate()
            process = self.process
            Thread(target=process.join, daemon=True).start()
            self.process = None
        for channel in (self.incoming, self.outgoing):
            if channel is not None:
                channel.cancel_join_thread()
                channel.close()
        self.incoming = self.outgoing = None
