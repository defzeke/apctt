"""Shared APCTT contracts: strict DTOs, canonical hashes and worker IPC (SPEC-05..SPEC-17).

Changing these contracts changes the API or worker protocol version.
Importing the package root stays dependency-free so stdlib-only helpers remain usable.
"""

CONTRACT_VERSION = "0.3.0"
KIT_VERSION = "0.5.0"
API_VERSION = "apctt-api-v0.2-draft"
WORKER_PROTOCOL_VERSION = "apctt-worker-v2"
CANONICALIZATION = "python-json-sortkeys-v1"
DEV_SCHEMA_VERSION = "apctt-pdf-example-v1"
DEMO_CHECKLIST_VERSION = "DEMO-CHECKLIST-NOT-OFFICIAL"
SUPPORTED_INPUT_PROFILES = ("pdf_report",)
LOADING_POLICY = "cold_per_case"
UNKNOWN_FIELD = "<unknown-field>"
