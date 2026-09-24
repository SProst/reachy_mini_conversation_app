from dataclasses import dataclass, field


@dataclass
class ChatterboxFlashTTSHandlerArguments:
    chatterbox_flash_model_name: str = field(
        default="ResembleAI/chatterbox-flash",
        metadata={"help": "The Chatterbox Flash model to use."},
    )
    chatterbox_flash_device: str = field(
        default="cuda",
        metadata={"help": "The device to run Chatterbox Flash on."},
    )
    chatterbox_flash_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Inference dtype: bfloat16, float16, or float32."},
    )
    chatterbox_flash_voice: str = field(
        default="Chatterbox Voice 1",
        metadata={"help": "Bundled reference-audio voice profile."},
    )
    chatterbox_flash_backend: str = field(
        default="torch",
        metadata={"help": "Inference engine: torch or flashinfer."},
    )
    chatterbox_flash_blocksize: int = field(
        default=512,
        metadata={"help": "Audio chunk size in 16 kHz samples."},
    )
    chatterbox_flash_drf_block_size: int = field(
        default=16,
        metadata={"help": "Block-diffusion speech token block size."},
    )
    chatterbox_flash_num_steps: int = field(
        default=10,
        metadata={"help": "Maximum denoising steps per speech token block."},
    )
    chatterbox_flash_max_speech_tokens: int = field(
        default=512,
        metadata={"help": "Maximum generated speech tokens per utterance."},
    )
