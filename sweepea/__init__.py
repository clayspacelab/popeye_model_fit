import os

# Set the XLA_PYTHON_CLIENT_MEM_FRACTION environment variable to a more reasonable default
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")