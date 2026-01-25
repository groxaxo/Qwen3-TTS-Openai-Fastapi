# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Streaming utilities for Qwen3-TTS token generation.

Provides async token streaming support for real-time audio output.
"""

import asyncio
import logging
from typing import AsyncGenerator, Optional

import torch
from transformers import BaseStreamer

logger = logging.getLogger(__name__)


class AsyncCodecTokenStreamer(BaseStreamer):
    """
    Intercepts tokens during generation and emits chunks for streaming decode.

    Works with HuggingFace generate() via the streamer= parameter.
    Accumulates tokens until chunk_size is reached, then emits for decoding.

    Args:
        chunk_size: Number of tokens to accumulate before emitting a chunk.
                   Smaller values = lower latency, larger values = smoother audio.
        context_overlap: Number of tokens to keep from previous chunk for context.
                        Helps maintain audio continuity between chunks.
    """

    def __init__(self, chunk_size: int = 75, context_overlap: int = 25):
        self.chunk_size = chunk_size
        self.context_overlap = context_overlap
        self.token_buffer = []
        self.chunk_queue: Optional[asyncio.Queue] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._finished = False
        self._error: Optional[Exception] = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Set event loop for cross-thread async communication."""
        self._loop = loop
        self.chunk_queue = asyncio.Queue()

    def put(self, value):
        """
        Called by generate() for each new token batch.

        Args:
            value: Token tensor from the model's generation step.
        """
        if self._finished:
            return

        try:
            if isinstance(value, torch.Tensor):
                # Handle different tensor shapes
                # Could be (batch, seq, features) or (batch, features) or just (features,)
                tokens = value.detach().cpu()

                # Flatten batch dimension if present
                if tokens.dim() > 1:
                    # Iterate over sequence dimension
                    for i in range(tokens.shape[0] if tokens.dim() == 2 else tokens.shape[1]):
                        if tokens.dim() == 2:
                            token = tokens[i]
                        else:
                            token = tokens[0, i]
                        self.token_buffer.append(token)
                        self._maybe_emit_chunk()
                else:
                    self.token_buffer.append(tokens)
                    self._maybe_emit_chunk()

        except Exception as e:
            logger.error(f"Error in streamer put(): {e}")
            self._error = e

    def _maybe_emit_chunk(self):
        """Emit a chunk if we have enough tokens."""
        if len(self.token_buffer) >= self.chunk_size:
            chunk_tokens = self.token_buffer[:self.chunk_size]
            chunk = torch.stack(chunk_tokens)

            if self._loop and self.chunk_queue is not None:
                asyncio.run_coroutine_threadsafe(
                    self.chunk_queue.put(chunk), self._loop
                )

            # Keep overlap tokens for context continuity
            self.token_buffer = self.token_buffer[
                self.chunk_size - self.context_overlap:
            ]

    def end(self):
        """Called when generation completes."""
        if self._finished:
            return

        self._finished = True

        # Flush remaining tokens as final chunk
        if self.token_buffer:
            try:
                chunk = torch.stack(self.token_buffer)
                if self._loop and self.chunk_queue is not None:
                    asyncio.run_coroutine_threadsafe(
                        self.chunk_queue.put(chunk), self._loop
                    )
            except Exception as e:
                logger.error(f"Error flushing final tokens: {e}")

        # Signal completion
        if self._loop and self.chunk_queue is not None:
            asyncio.run_coroutine_threadsafe(
                self.chunk_queue.put(None), self._loop
            )

    async def get_chunks(self) -> AsyncGenerator[torch.Tensor, None]:
        """
        Async generator yielding token chunks as they become available.

        Yields:
            Token chunks as torch.Tensor, ready for decoding.

        Raises:
            RuntimeError: If an error occurred during generation.
        """
        if self.chunk_queue is None:
            raise RuntimeError("Event loop not set. Call set_event_loop() first.")

        while True:
            try:
                chunk = await asyncio.wait_for(
                    self.chunk_queue.get(),
                    timeout=120.0  # 2 minute timeout per chunk
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for chunk, generation may have stalled")
                break

            if chunk is None:
                # End signal received
                break

            if self._error:
                raise RuntimeError(f"Generation error: {self._error}")

            yield chunk

    @property
    def is_finished(self) -> bool:
        """Check if generation has completed."""
        return self._finished


class SentenceSplitter:
    """
    Utility class for splitting text into sentences for fallback streaming.

    Used when token-level streaming is not available or produces artifacts.
    """

    # Common sentence-ending patterns
    SENTENCE_ENDINGS = r'(?<=[.!?])\s+'

    @staticmethod
    def split(text: str) -> list[str]:
        """
        Split text into sentences.

        Args:
            text: Input text to split.

        Returns:
            List of sentences.
        """
        import re
        sentences = re.split(SentenceSplitter.SENTENCE_ENDINGS, text.strip())
        return [s.strip() for s in sentences if s.strip()]
