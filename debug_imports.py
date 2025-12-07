import sys
import os

print(f"CWD: {os.getcwd()}")
print(f"Path: {sys.path}")
try:
    import documents

    print(f"Documents: {documents.__file__}")
    import documents.graph

    print(f"Graph Package: {documents.graph.__file__}")
    from documents.graph import nodes

    print(f"Nodes Module: {nodes.__file__}")
    print("Import Success")
except ImportError as e:
    print(f"Import Failed: {e}")
except Exception as e:
    print(f"Error: {e}")
