"""Task with direct aiohttp POST, pre-serialization, and concurrency throttling.  """

from __future__ import annotations

import array
import asyncio
import contextlib
import json
import time
from typing import Any, Optional

import aiohttp
import mlperf_loadgen as lg
from loguru import logger
from mlperf_inf_mm_q3vl.schema import Dataset, Endpoint, LoadedSample, TestScenario, TestSettings
from mlperf_inf_mm_q3vl.task import ShopifyGlobalCatalogue
from pympler import asizeof


class PreSerializedSample(LoadedSample):
    """LoadedSample with a pre-serialized request body attached."""

    request_body: bytes


class AioHttpTask(ShopifyGlobalCatalogue):
    """ShopifyGlobalCatalogue with direct aiohttp POST and pre-serialization.

    Overrides the base class to:
    - Pre-serialize request bodies at sample-load time.
    - POST raw bytes via aiohttp instead of using the OpenAI SDK.
    - Optionally throttle concurrency with an asyncio.Semaphore.
    """

    def __init__(
        self,
        dataset: Dataset,
        endpoint: Endpoint,
        settings: TestSettings,
        random_seed: int = 12345,
        max_concurrency: Optional[int] = None,
    ) -> None:
        super().__init__(dataset=dataset, endpoint=endpoint, settings=settings, random_seed=random_seed)

        self._max_concurrency = max_concurrency
        self._semaphore: Optional[asyncio.Semaphore] = None

        base_url = endpoint.url.rstrip("/")
        self._api_url = f"{base_url}/chat/completions"
        self._api_headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if endpoint.api_key:
            self._api_headers["Authorization"] = f"Bearer {endpoint.api_key}"

        request_timeout_seconds = endpoint.request_timeout.total_seconds()

        async def _create_session() -> aiohttp.ClientSession:
            return aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=0),
                timeout=aiohttp.ClientTimeout(
                    total=request_timeout_seconds,
                    sock_connect=5.0,
                ),
                headers=self._api_headers,
            )

        self._aio_session: aiohttp.ClientSession = asyncio.run_coroutine_threadsafe(
            _create_session(), self.event_loop,
        ).result()

        self._use_streaming = (
            settings.scenario is TestScenario.SERVER
            and settings.use_token_latencies
        )

        logger.info(
            "AioHttpTask initialized with max_concurrency={}, api_url={}",
            self._max_concurrency,
            self._api_url,
        )


    def __del__(self) -> None:
        """Clean up aiohttp session before base class cleanup."""
        async def _close_aio_session() -> None:
            if not self._aio_session.closed:
                await self._aio_session.close()

        try:
            asyncio.run_coroutine_threadsafe(
                _close_aio_session(),
                self.event_loop,
            ).result(timeout=5.0)
        except Exception as e:  # noqa: BLE001
            logger.trace("Error closing aiohttp session during cleanup: {}", e)

        super().__del__()


    def _ensure_semaphore(self) -> None:
        """Lazily create the semaphore inside the event-loop thread."""
        if self._max_concurrency is not None and self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrency)


    def _build_request_body(self, sample: Any) -> bytes:  # noqa: ANN401
        """Build and JSON-serialize the full chat-completion request body.

        Called **once per sample at load time** so that the benchmark loop
        only needs to POST pre-serialized bytes.
        """
        body: dict[str, Any] = {
            "model": self.endpoint.model.repo_id,
            "messages": sample.messages,
        }

        # Sampling parameters — only include non-None values.
        sp = self.endpoint.sampling_params
        if sp.frequency_penalty is not None:
            body["frequency_penalty"] = sp.frequency_penalty
        if sp.presence_penalty is not None:
            body["presence_penalty"] = sp.presence_penalty
        if sp.temperature is not None:
            body["temperature"] = sp.temperature
        if sp.top_p is not None:
            body["top_p"] = sp.top_p
        for k in ("top_k", "min_p", "repetition_penalty"):
            v = getattr(sp, k, None)
            if v is not None:
                body[k] = v

        # Guided decoding / response format
        if sample.response_format is not None:
            body["response_format"] = sample.response_format.model_dump(
                mode="json",
                by_alias=True,
            )

        # Streaming flags
        if self._use_streaming:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}

        return json.dumps(body, ensure_ascii=False).encode("utf-8")


    def construct_qsl(self) -> int:
        """Construct the LoadGen QSL with pre-serialized request bodies."""

        def _load_samples_to_ram(query_sample_indices: list[int]) -> None:
            """Load samples and pre-serialize HTTP request bodies."""
            logger.info(
                "Starting to load {} samples to RAM...",
                len(query_sample_indices),
            )
            tic = time.perf_counter()
            for index in query_sample_indices:
                sample = self.formulate_loaded_sample(
                    self.dataset[index],
                    use_guided_decoding=self.endpoint.use_guided_decoding,
                )
                # Wrap with pre-serialized request body.
                self.loaded_samples[index] = PreSerializedSample(
                    messages=sample.messages,
                    response_format=sample.response_format,
                    request_body=self._build_request_body(sample),
                )
            logger.info(
                "Loaded {} samples to RAM, which took {} seconds and {} GB in total.",
                len(query_sample_indices),
                time.perf_counter() - tic,
                asizeof.asizeof(self.loaded_samples) / 1024 / 1024 / 1024,
            )

        def _unload_samples_from_ram(query_sample_indices: list[int]) -> None:
            """Unload samples from host memory after testing."""
            logger.info(
                "Starting to unload {} samples from RAM...",
                len(query_sample_indices),
            )
            tic = time.perf_counter()
            for index in query_sample_indices:
                sample_to_unload = self.loaded_samples.pop(index, None)
                del sample_to_unload
            logger.info(
                "Unloaded {} samples from RAM, which took {} seconds.",
                len(query_sample_indices),
                time.perf_counter() - tic,
            )

        return lg.ConstructQSL(
            self.total_num_samples,
            self.estimated_num_performance_samples,
            _load_samples_to_ram,
            _unload_samples_from_ram,
        )


    async def _query_endpoint_async_batch(self, query_sample: lg.QuerySample) -> None:
        """Query the endpoint via direct HTTP POST with pre-serialized body."""
        self._ensure_semaphore()
        throttle = self._semaphore if self._semaphore is not None else contextlib.nullcontext()
        async with throttle:
            try:
                sample = self.loaded_samples[query_sample.index]
                logger.debug(
                    "Issuing query sample index: {} with response ID: {}",
                    query_sample.index,
                    query_sample.id,
                )
                tic = time.perf_counter()
                async with self._aio_session.post(
                    self._api_url,
                    data=sample.request_body,
                ) as resp:
                    resp.raise_for_status()
                    raw = await resp.read()
                    resp_data = json.loads(raw)
                logger.debug(
                    "Received response (ID: {}) from endpoint after {} seconds.",
                    query_sample.id,
                    time.perf_counter() - tic,
                )
                content = (resp_data.get("choices") or [{}])[0].get("message", {}).get("content")
                if content is None:
                    content = ""
                usage = resp_data.get("usage") or {}
                token_count = int(usage.get("completion_tokens", 0))
                logger.debug(
                    "Response token count (ID: {}): {}",
                    query_sample.id,
                    token_count,
                )
                bytes_array = array.array("B", content.encode("utf-8"))
                address, length = bytes_array.buffer_info()
                size_in_bytes = length * bytes_array.itemsize
                lg.QuerySamplesComplete(
                    [
                        lg.QuerySampleResponse(
                            query_sample.id,
                            address,
                            size_in_bytes,
                            token_count,
                        ),
                    ],
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Error processing query sample index {} with response ID {}.",
                    query_sample.index,
                    query_sample.id,
                )
                # Send empty response to LoadGen to avoid hanging.
                empty_content = ""
                bytes_array = array.array("B", empty_content.encode("utf-8"))
                address, length = bytes_array.buffer_info()
                size_in_bytes = length * bytes_array.itemsize
                lg.QuerySamplesComplete(
                    [
                        lg.QuerySampleResponse(
                            query_sample.id,
                            address,
                            size_in_bytes,
                            0,
                        ),
                    ],
                )

    async def _query_endpoint_async_stream(self, query_sample: lg.QuerySample) -> None:
        """Query the endpoint via direct HTTP POST with SSE streaming."""
        self._ensure_semaphore()
        throttle = self._semaphore if self._semaphore is not None else contextlib.nullcontext()
        async with throttle:
            ttft_set = False
            try:
                sample = self.loaded_samples[query_sample.index]
                logger.debug(
                    "Issuing query sample index: {} with response ID: {}",
                    query_sample.index,
                    query_sample.id,
                )
                word_array: list[str] = []
                total_tokens = 0

                async with self._aio_session.post(
                    self._api_url,
                    data=sample.request_body,
                ) as resp:
                    resp.raise_for_status()
                    # Parse Server-Sent Events (SSE) line by line.
                    while True:
                        line = await resp.content.readline()
                        if not line:
                            break
                        line_str = line.decode("utf-8").strip()
                        if not line_str or not line_str.startswith("data: "):
                            continue
                        data = line_str[6:]
                        if data == "[DONE]":
                            break

                        chunk = json.loads(data)

                        # Check for usage in the final chunk.
                        chunk_usage = chunk.get("usage")
                        if chunk_usage is not None:
                            total_tokens = int(
                                chunk_usage.get("completion_tokens", 0),
                            )

                        # Process content delta.
                        choices = chunk.get("choices")
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        text = delta.get("content")
                        if not text:
                            continue

                        # First non-empty token -> TTFT
                        if ttft_set is False:
                            bytes_array = array.array("B", text.encode("utf-8"))
                            address, length = bytes_array.buffer_info()
                            size_in_bytes = length * bytes_array.itemsize
                            lg.FirstTokenComplete(
                                [
                                    lg.QuerySampleResponse(
                                        query_sample.id,
                                        address,
                                        size_in_bytes,
                                        1,
                                    ),
                                ],
                            )
                            ttft_set = True
                        word_array.append(text)

                # When the stream ends — report total latency.
                content = "".join(word_array)
                logger.debug(
                    "Response token count (ID: {}): {}",
                    query_sample.id,
                    total_tokens,
                )
                bytes_array = array.array("B", content.encode("utf-8"))
                address, length = bytes_array.buffer_info()
                size_in_bytes = length * bytes_array.itemsize
                lg.QuerySamplesComplete(
                    [
                        lg.QuerySampleResponse(
                            query_sample.id,
                            address,
                            size_in_bytes,
                            total_tokens,
                        ),
                    ],
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Error processing query sample index {} with response ID {}.",
                    query_sample.index,
                    query_sample.id,
                )
                # Send empty response to LoadGen to avoid hanging.
                empty_content = ""
                bytes_array = array.array("B", empty_content.encode("utf-8"))
                address, length = bytes_array.buffer_info()
                size_in_bytes = length * bytes_array.itemsize
                # If TTFT was not set, we still need to complete that.
                if not ttft_set:
                    lg.FirstTokenComplete(
                        [
                            lg.QuerySampleResponse(
                                query_sample.id,
                                address,
                                size_in_bytes,
                                0,
                            ),
                        ],
                    )
                lg.QuerySamplesComplete(
                    [
                        lg.QuerySampleResponse(
                            query_sample.id,
                            address,
                            size_in_bytes,
                            0,
                        ),
                    ],
                )
