from opentelemetry.trace import get_tracer
from langfuse import observe, get_client

tr = get_tracer('noesis2.smoke')

@observe(name='smoke.child')
def child():
    return 'ok'

with tr.start_as_current_span('smoke.trace'):
    print(child())

# Flush synchronously so the trace appears immediately in the UI
get_client().flush()
print('Flush complete.')
