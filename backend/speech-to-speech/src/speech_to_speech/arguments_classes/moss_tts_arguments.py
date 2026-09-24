from dataclasses import dataclass, field


@dataclass
class MossTTSHandlerArguments:
    moss_tts_model_name: str = field(
        default="OpenMOSS-Team/MOSS-TTS-Realtime",
        metadata={"help": "The MOSS realtime TTS model to use."},
    )
    moss_tts_codec_name: str = field(
        default="OpenMOSS-Team/MOSS-Audio-Tokenizer",
        metadata={"help": "The MOSS audio tokenizer model to use."},
    )
    moss_tts_model_revision: str = field(
        default="75682787d8e2fcc73faca37ba2931453ca9c4022",
        metadata={"help": "Pinned MOSS realtime model revision."},
    )
    moss_tts_codec_revision: str = field(
        default="3cd226ba2947efa357ef453bcad111b6eafba782",
        metadata={"help": "Pinned MOSS audio tokenizer revision."},
    )
    moss_tts_device: str = field(
        default="cuda",
        metadata={"help": "The device to run MOSS TTS on. CUDA is recommended."},
    )
    moss_tts_dtype: str = field(
        default="auto",
        metadata={"help": "Inference dtype: auto, bfloat16, float16, or float32."},
    )
    moss_tts_voice: str = field(
        default="MOSS Voice 1",
        metadata={"help": "Bundled reference-audio voice profile."},
    )
    moss_tts_blocksize: int = field(
        default=512,
        metadata={"help": "Audio chunk size in 16 kHz samples."},
    )
    moss_tts_max_length: int = field(
        default=3000,
        metadata={"help": "Maximum multimodal sequence length used during synthesis."},
    )
