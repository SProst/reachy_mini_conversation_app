from dataclasses import dataclass, field


@dataclass
class NemotronStreamingSTTHandlerArguments:
    """Arguments for the Nemotron 3.5 streaming speech-to-text handler."""

    nemotron_streaming_model_name: str = field(
        default="nvidia/nemotron-3.5-asr-streaming-0.6b",
        metadata={"help": "Hugging Face model identifier for Nemotron streaming ASR."},
    )
    nemotron_streaming_device: str = field(
        default="auto",
        metadata={"help": "Device to run Nemotron on: 'auto', 'cuda', or 'cpu'."},
    )
    nemotron_streaming_compute_type: str = field(
        default="float16",
        metadata={"help": "Model precision: 'float16', 'bfloat16', or 'float32'."},
    )
    nemotron_streaming_language: str = field(
        default="en-US",
        metadata={"help": "Target locale such as 'en-US', or 'auto' for model language detection."},
    )
    nemotron_streaming_num_lookahead_tokens: int = field(
        default=6,
        metadata={"help": "Right-context frames. Supported values are 0, 1, 6, and 13; 6 gives 560 ms chunks."},
    )
