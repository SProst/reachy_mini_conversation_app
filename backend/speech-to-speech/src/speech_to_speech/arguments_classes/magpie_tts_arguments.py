from dataclasses import dataclass, field


@dataclass
class MagpieTTSHandlerArguments:
    magpie_tts_model_name: str = field(
        default="nvidia/magpie_tts_multilingual_357m",
        metadata={"help": "The MagpieTTS model repository to use."},
    )
    magpie_tts_checkpoint_filename: str = field(
        default="magpie_tts_multilingual_357m.nemo",
        metadata={"help": "The NeMo checkpoint filename in the model repository."},
    )
    magpie_tts_revision: str = field(
        default="452ef560f972c38d5fc16476259aac9456453547",
        metadata={"help": "The model revision to load."},
    )
    magpie_tts_codec_model_name: str = field(
        default="nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps",
        metadata={"help": "The NanoCodec model used to decode Magpie audio tokens."},
    )
    magpie_tts_device: str = field(
        default="cuda",
        metadata={"help": "The device on which to run MagpieTTS."},
    )
    magpie_tts_voice: str = field(
        default="Aria",
        metadata={"help": "One of MagpieTTS's baked speaker voices."},
    )
    magpie_tts_language: str = field(
        default="en",
        metadata={"help": "Fallback synthesis language code."},
    )
    magpie_tts_apply_text_normalization: bool = field(
        default=False,
        metadata={"help": "Apply NeMo text normalization before synthesis."},
    )
    magpie_tts_use_cfg: bool = field(
        default=True,
        metadata={"help": "Use classifier-free guidance during synthesis."},
    )
    magpie_tts_blocksize: int = field(
        default=512,
        metadata={"help": "Audio chunk size in 16 kHz samples."},
    )
