
import os
import django
from drf_spectacular.generators import SchemaGenerator

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "noesis2.settings")
django.setup()

generator = SchemaGenerator()
schema = generator.get_schema(request=None, public=True)

path = "/v1/ai/frameworks/analyze/"
method = "post"

if path in schema["paths"]:
    print(f"Path {path} found.")
    operation = schema["paths"][path].get(method)
    if operation:
        print(f"Method {method} found.")
        responses = operation.get("responses", {})
        responses = operation.get("responses", {})
        print(f"Responses keys: {list(responses.keys())}")
        if "415" in responses:
            print("415 response found.")
            print(responses["415"])
        if "200" in responses:
            print("200 response found.")
            print(responses["200"])
        else:
            print("415 response NOT found.")
    else:
        print(f"Method {method} NOT found.")
else:
    print(f"Path {path} NOT found.")
