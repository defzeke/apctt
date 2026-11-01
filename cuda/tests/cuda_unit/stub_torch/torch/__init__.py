"""STRUCTURAL TEST STUB — not PyTorch. Lets CPU-only hosts exercise CUDA control flow.

Behaviour is chosen by environment variables set by the test wrapper interpreter. Nothing produced
with this stub is CUDA evidence; tests that use it are CPU structure tests (CUDA-01).
"""

import os

__version__ = "0.0-structural-stub"


class version:  # noqa: N801 - mirrors torch.version
    cuda = os.environ.get("STUB_CUDA_BUILD") or None


class cuda:  # noqa: N801 - mirrors torch.cuda
    @staticmethod
    def is_available() -> bool:
        return os.environ.get("STUB_CUDA_AVAILABLE") == "1"

    @staticmethod
    def is_initialized() -> bool:
        return False

    @staticmethod
    def device_count() -> int:
        return 1 if os.environ.get("STUB_CUDA_AVAILABLE") == "1" else 0

    @staticmethod
    def get_device_properties(index):
        raise RuntimeError("structural stub: no CUDA kernels")


def ones(*args, **kwargs):
    raise RuntimeError("structural stub: no tensors")
