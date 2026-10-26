"""OpenTelemetry Tracing & Observability integration for Google Cloud Trace.

Conforms to Google Cloud Trace semantic conventions for microsecond-level
distributed profiling of Argon Engine inference and plasticity ticks.
"""

import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False
    trace = None
    Status = None
    StatusCode = None


class ArgonTelemetryTracer:
    """Instruments Argon Engine operations with OpenTelemetry spans for Google Cloud Trace."""

    def __init__(self, service_name: str = "argon-engine", tracer_name: str = "argon.engine"):
        self.service_name = service_name
        self.tracer_name = tracer_name
        if HAS_OTEL:
            self._tracer = trace.get_tracer(self.tracer_name)
        else:
            self._tracer = None

    @contextmanager
    def trace_tick(
        self,
        sequence_id: int,
        polarity: float = 1.0,
        span_name: str = "argon.engine.tick",
    ) -> Iterator[Dict[str, Any]]:
        """Context manager creating an OpenTelemetry span around an engine tick.

        Yields a metadata dictionary that can be populated with `mean_goodness`
        and `output_l2_norm` before exiting the context.
        """
        span_metrics: Dict[str, Any] = {
            "sequence_id": sequence_id,
            "polarity": polarity,
            "mean_goodness": 0.0,
            "output_l2_norm": 0.0,
            "total_ticks": 0,
        }

        start_time_ns = time.perf_counter_ns()

        if self._tracer is not None:
            with self._tracer.start_as_current_span(span_name) as span:
                span.set_attribute("argon.sequence_id", int(sequence_id))
                span.set_attribute("argon.polarity", float(polarity))
                span.set_attribute("argon.service", self.service_name)
                try:
                    yield span_metrics
                    elapsed_ns = time.perf_counter_ns() - start_time_ns
                    span.set_attribute("argon.latency_ns", elapsed_ns)
                    span.set_attribute("argon.mean_goodness", float(span_metrics.get("mean_goodness", 0.0)))
                    span.set_attribute("argon.output_l2_norm", float(span_metrics.get("output_l2_norm", 0.0)))
                    span.set_attribute("argon.total_ticks", int(span_metrics.get("total_ticks", 0)))
                    span.set_status(Status(StatusCode.OK))
                except Exception as ex:
                    span.set_status(Status(StatusCode.ERROR, str(ex)))
                    span.record_exception(ex)
                    raise
        else:
            # Fallback when OpenTelemetry SDK is not initialized
            yield span_metrics

    def instrument_engine_step(
        self,
        engine,
        sensory_input,
        polarity: float = 1.0,
        sequence_id: Optional[int] = None,
    ):
        """Execute a single engine step wrapped inside an OpenTelemetry trace span."""
        seq_id = sequence_id if sequence_id is not None else 0
        with self.trace_tick(sequence_id=seq_id, polarity=polarity) as metrics:
            output, telemetry = engine.step(sensory_input=sensory_input, polarity=polarity)
            metrics["sequence_id"] = telemetry.sequence_id
            metrics["mean_goodness"] = telemetry.mean_goodness
            metrics["output_l2_norm"] = telemetry.output_l2_norm
            metrics["total_ticks"] = telemetry.total_ticks
            return output, telemetry


def setup_google_cloud_trace_exporter(project_id: Optional[str] = None) -> bool:
    """Attempts to configure the Google Cloud Trace OpenTelemetry exporter.

    Returns True if successfully initialized with Google Cloud Trace credentials,
    False if opentelemetry-exporter-gcp-trace is not installed or unavailable.
    """
    if not HAS_OTEL:
        return False

    try:
        from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter  # type: ignore
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore

        provider = TracerProvider()
        trace.set_tracer_provider(provider)
        exporter = CloudTraceSpanExporter(project_id=project_id)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        return True
    except ImportError:
        return False
