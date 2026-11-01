"""CUDA dummy example: synthetic training → checkpoint → fresh-worker CUDA inference (SPEC-16).

DEMO only (`example_only=true`; it runs with the official checklist of config/cuda-example.json,
decision D74). Not a TRL model or baseline. Importing this package
does not import torch; `model`, `train` and `adapter` import torch where they need it.
"""
