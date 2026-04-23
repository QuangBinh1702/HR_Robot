"""
Utilities to make ONNX Runtime CUDA provider work reliably on Windows.
"""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path

_CUDA_DLL_PATHS_CONFIGURED = False
_DLL_DIR_HANDLES = []


def configure_onnxruntime_cuda_dll_paths() -> None:
    """
    Add pip-installed NVIDIA runtime folders to Windows DLL search path.

    ONNX Runtime CUDA EP loads CUDA/cuDNN DLLs by name. With pip-installed
    NVIDIA packages, those DLLs live under site-packages/nvidia/*/bin and are
    not always in the default DLL search path.
    """
    global _CUDA_DLL_PATHS_CONFIGURED

    if _CUDA_DLL_PATHS_CONFIGURED:
        return

    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        _CUDA_DLL_PATHS_CONFIGURED = True
        return

    search_roots = [Path(p) for p in site.getsitepackages()]
    subdirs = [
        ("nvidia", "cudnn", "bin"),
        ("nvidia", "cublas", "bin"),
        ("nvidia", "cuda_runtime", "bin"),
        ("nvidia", "cuda_nvrtc", "bin"),
        ("nvidia", "cufft", "bin"),
        ("nvidia", "curand", "bin"),
        ("nvidia", "nvjitlink", "bin"),
    ]

    for root in search_roots:
        for parts in subdirs:
            dll_dir = root.joinpath(*parts)
            if dll_dir.exists():
                # Keep handle alive for process lifetime, otherwise the
                # directory can be removed from DLL search path.
                _DLL_DIR_HANDLES.append(os.add_dll_directory(str(dll_dir)))

    # ONNX Runtime >= 1.21 provides an explicit preload API that helps
    # resolve CUDA/cuDNN DLL dependencies before creating a session.
    try:
        import onnxruntime as ort
        preload = getattr(ort, "preload_dlls", None)
        if callable(preload):
            preload()
    except Exception:
        # Keep startup resilient; session creation will fall back to CPU.
        pass

    _CUDA_DLL_PATHS_CONFIGURED = True


def get_onnxruntime_providers() -> tuple[list[str], str]:
    """
    Choose the best available ONNX Runtime execution providers for this machine.

    Preference:
      1. CUDA
      2. DirectML (Windows GPU fallback)
      3. CPU
    """
    configure_onnxruntime_cuda_dll_paths()

    import onnxruntime as ort

    available = ort.get_available_providers()

    if "CUDAExecutionProvider" in available:
        return (["CUDAExecutionProvider", "CPUExecutionProvider"], "GPU (CUDA)")

    if "DmlExecutionProvider" in available:
        return (["DmlExecutionProvider", "CPUExecutionProvider"], "GPU (DirectML)")

    return (["CPUExecutionProvider"], "CPU (ONNX Runtime)")
