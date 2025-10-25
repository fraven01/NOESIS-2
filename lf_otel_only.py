from opentelemetry.trace import get_tracer

tr = get_tracer("noesis2.otel-smoke")
with tr.start_as_current_span("smoke.otel.trace"):
    print("otel ok")
