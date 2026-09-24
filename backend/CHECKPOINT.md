# Windows speech-backend checkpoint

This branch preserves the Windows Reachy work found on 2026-09-24. It is a follow-on development checkpoint, not a release or a merge-ready addition to the conversation app.

- App branch: `fix/magpie-voices-stable-sdk` updates the Magpie voice catalog/default and Reachy SDK dependency to stable 1.10.0.
- This branch includes that app commit, then the original `.space-nemo-base` snapshot as a baseline commit, followed by the `.space-nemo-cpp` snapshot at `backend/speech-to-speech`.
- The snapshots' README identifies speech-to-speech base `9f59bc72f66ee84b006e2682b9547144e7f74827`. This is recorded provenance, not independently verified Git ancestry.
- NeMo-Speech.cpp is referenced at published commit `4f9676226f667d14608487df744f375db87127f8`. The clean dependency checkout is not vendored here.
- Generated `validation-nemo-cpp.wav`, caches, environments, credentials, local profiles, and service state are excluded. Original backend reference audio/assets are retained.

The native adaptation implements Magpie TTS through ctypes and NeMo-Speech.cpp. ASR remains Nemotron 3.5 streaming (16 kHz, default lookahead 6). Speaker settings select synthesized voices; the inspected transcription schemas do not implement speaker identity or diarized segments. No Nemotron-3-Diarization integration was performed.

For subsequent backend work, use `backend/speech-to-speech` as its own project root or transplant its commits into the appropriate speech backend repository. The enclosing app's checks are not configured to treat an additional Python project as part of the application. In particular, its file-path budget check can reject the nested backend paths.

## Validation on Windows

Used existing Python 3.11.15 and development tools; no dependencies were installed and no robot/service or model inference was started.

App branch validated in a separate checkout without backend files:
- Ruff lint passed; formatting passed (79 files).
- mypy passed (45 source files).
- `uv lock --check --offline` passed.
- 318 tests passed with two realtime tests deselected; `test_parallel_tool_calls_trigger_single_response` then passed separately: 319 app tests passed in total.
- `test_partial_transcription_uses_latest_snapshot` hangs in the mocked realtime session, so the full suite did not complete. It was stopped after a diagnostic retry; no application behavior or tests were changed to mask it.

Backend snapshot:
- Ruff lint and formatting passed (181 files).
- `tests/test_nemo_speech_cpp.py`: 2 passed, using a fake native library.
- `tests/test_magpie_tts_handler.py` could not collect in the available app environment: Pillow (`PIL`) is absent. The complete backend suite and native Docker/CUDA runtime have not been validated in this publication task.
