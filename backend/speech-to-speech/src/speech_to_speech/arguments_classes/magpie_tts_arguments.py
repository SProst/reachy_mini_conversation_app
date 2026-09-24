import os
from dataclasses import dataclass, field


@dataclass
class MagpieTTSHandlerArguments:
    magpie_tts_library_path: str = field(
        default_factory=lambda: os.getenv("NEMO_SPEECH_TTS_LIBRARY", "libnemo_speech_tts.so"),
        metadata={"help": "Path to the NeMo-Speech.cpp TTS shared library."},
    )
    magpie_tts_model_path: str = field(
        default_factory=lambda: os.getenv("NEMO_SPEECH_TTS_MODEL_PATH", ""),
        metadata={"help": "Path to the MagpieTTS v2602 GGUF."},
    )
    magpie_tts_codec_path: str = field(
        default_factory=lambda: os.getenv("NEMO_SPEECH_TTS_CODEC_PATH", ""),
        metadata={"help": "Path to the NanoCodec decoder GGUF."},
    )
    magpie_tts_tokenizer_path: str = field(
        default_factory=lambda: os.getenv("NEMO_SPEECH_TTS_TOKENIZER_PATH", ""),
        metadata={"help": "Path to the tokenizer assets extracted from the MagpieTTS checkpoint."},
    )
    magpie_tts_device: str = field(
        default="cuda",
        metadata={"help": "NeMo-Speech.cpp backend preference: cuda, cpu, or auto."},
    )
    magpie_tts_voice: str = field(
        default="Aria",
        metadata={"help": "One of MagpieTTS's baked speaker voices."},
    )
    magpie_tts_language: str = field(
        default="en-US",
        metadata={"help": "Fallback synthesis language code."},
    )
    magpie_tts_threads: int = field(
        default=4,
        metadata={"help": "CPU threads used by NeMo-Speech.cpp."},
    )
    magpie_tts_codec_threads: int = field(
        default=0,
        metadata={"help": "NanoCodec CPU threads; zero uses the main thread count."},
    )
    magpie_tts_chunk_frames: int = field(
        default=3,
        metadata={"help": "NanoCodec frames generated per native streaming chunk."},
    )
    magpie_tts_codec_queue_depth: int = field(
        default=4,
        metadata={"help": "Native codec worker queue depth."},
    )
    magpie_tts_codec_history_frames: int = field(
        default=-1,
        metadata={"help": "Rolling codec history frames; -1 uses the native default."},
    )
    magpie_tts_codec_future_frames: int = field(
        default=1,
        metadata={"help": "Rolling codec future frames."},
    )
    magpie_tts_window_ms: int = field(
        default=0,
        metadata={"help": "Overlap-add window duration in milliseconds."},
    )
    magpie_tts_use_cfg: bool = field(
        default=True,
        metadata={"help": "Use classifier-free guidance during synthesis."},
    )
    magpie_tts_use_kv_cache: bool = field(
        default=True,
        metadata={"help": "Use the native decoder KV cache."},
    )
    magpie_tts_use_stateful_codec: bool = field(
        default=True,
        metadata={"help": "Use the stateful streaming NanoCodec decoder."},
    )
    magpie_tts_codec_cpu: bool = field(
        default=False,
        metadata={"help": "Run NanoCodec on CPU instead of the selected backend."},
    )
    magpie_tts_flush_partial_chunk: bool = field(
        default=True,
        metadata={"help": "Emit the final partial native codec chunk."},
    )
    magpie_tts_verbose: bool = field(
        default=False,
        metadata={"help": "Enable verbose NeMo-Speech.cpp inference logs."},
    )
    magpie_tts_blocksize: int = field(
        default=512,
        metadata={"help": "Audio chunk size in 16 kHz samples."},
    )
