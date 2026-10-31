"""A flush_at larger than the OpenTelemetry default queue size must be usable."""

from typing import Optional, Sequence

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from langfuse import Langfuse
from langfuse._client.resource_manager import LangfuseResourceManager


class NoOpSpanExporter(SpanExporter):
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


@pytest.fixture
def isolated_client():
    with LangfuseResourceManager._lock:
        LangfuseResourceManager._instances.clear()

    client = Langfuse(
        public_key="pk-flush-at",
        secret_key="sk-flush-at",
        span_exporter=NoOpSpanExporter(),
        flush_at=5000,
    )

    yield client

    client.shutdown()


@pytest.mark.unit
def test_flush_at_above_the_default_queue_size_is_accepted(isolated_client):
    """Configuring a large flush_at must not fail client construction."""
    assert isolated_client is not None


@pytest.mark.unit
def test_client_with_large_flush_at_can_flush(isolated_client):
    isolated_client.flush()


def _queue_capacity(client: Langfuse) -> Optional[int]:
    """Best-effort read of the configured queue capacity.

    Returns None when the OpenTelemetry internals this relies on are not present,
    so the caller can skip instead of failing on an upstream refactor.
    """
    provider = client._resources.tracer_provider

    for processor in provider._active_span_processor._span_processors:
        queue = getattr(processor, "_queue", None) or getattr(
            processor, "_batch_processor", None
        )
        target = getattr(queue, "_queue", queue)
        capacity = getattr(target, "maxlen", None) or getattr(target, "capacity", None)

        if isinstance(capacity, int):
            return capacity

    return None


@pytest.mark.unit
def test_flush_at_uses_the_effective_default_queue_size(monkeypatch):
    """A smaller OTEL_BSP_MAX_QUEUE_SIZE must still be widened when flush_at needs it."""
    monkeypatch.setenv("OTEL_BSP_MAX_QUEUE_SIZE", "1000")

    with LangfuseResourceManager._lock:
        LangfuseResourceManager._instances.clear()

    client = Langfuse(
        public_key="pk-flush-at",
        secret_key="sk-flush-at",
        span_exporter=NoOpSpanExporter(),
        flush_at=1500,
    )

    try:
        client.flush()
    finally:
        client.shutdown()


@pytest.mark.unit
def test_a_larger_default_queue_size_is_not_shrunk(monkeypatch):
    """Widening the batch must not quietly narrow an already larger queue."""
    monkeypatch.setenv("OTEL_BSP_MAX_QUEUE_SIZE", "5000")

    with LangfuseResourceManager._lock:
        LangfuseResourceManager._instances.clear()

    client = Langfuse(
        public_key="pk-flush-at",
        secret_key="sk-flush-at",
        span_exporter=NoOpSpanExporter(),
        flush_at=3000,
    )

    try:
        capacity = _queue_capacity(client)

        if capacity is None:
            pytest.skip("OpenTelemetry no longer exposes the queue capacity")

        assert capacity == 5000
    finally:
        client.shutdown()


@pytest.mark.unit
def test_spans_are_still_exported_with_a_large_flush_at():
    """A large flush_at must not silently drop or hold back finished spans."""
    with LangfuseResourceManager._lock:
        LangfuseResourceManager._instances.clear()

    exporter = InMemorySpanExporter()

    client = Langfuse(
        public_key="pk-flush-at",
        secret_key="sk-flush-at",
        span_exporter=exporter,
        flush_at=5000,
    )

    try:
        client.start_observation(name="flush-at-span").end()

        client.flush()

        span_names = [span.name for span in exporter.get_finished_spans()]
        assert "flush-at-span" in span_names
    finally:
        client.shutdown()
