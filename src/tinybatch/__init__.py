"""tinybatch — a miniature vLLM-style LLM inference engine.

Paged KV cache, continuous batching, prefix caching, streaming — small
enough to read in an afternoon, real enough to serve Qwen2.5 on a laptop.
"""
from .block_manager import Block, BlockManager, OutOfBlocks
from .config import EngineConfig
from .engine import LLMEngine, StepOutput
from .scheduler import Request, ScheduledBatch, Scheduler, Status

__version__ = "0.1.0"
__all__ = [
    "Block", "BlockManager", "OutOfBlocks", "EngineConfig",
    "LLMEngine", "StepOutput", "Request", "ScheduledBatch", "Scheduler", "Status",
]
