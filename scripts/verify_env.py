"""
QuantGuard environment sanity check.

Invoked by the CCS '26 Artifact Appendix "Basic test" subsection.
Reviewers should run:

    python scripts/verify_env.py

after completing the Setup instructions in README.md. If every check
succeeds the script prints:

    [Success] All components work fine. Go ahead to run experiments!

Any failure prints an actionable [FAIL] line and exits with a non-zero
status so the reviewer immediately knows which dependency is missing
or misconfigured.

This script intentionally avoids downloading any model weights and
performs the bitsandbytes smoke test on a tiny dummy tensor, so it
finishes in well under a minute on a single GPU.
"""
from __future__ import annotations

import importlib
import sys
import traceback


# ----------------------------------------------------------------------
# Pretty-printing helpers
# ----------------------------------------------------------------------
def _ok(msg: str) -> None:
    print(f"[OK]   {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


# ----------------------------------------------------------------------
# Individual checks
# ----------------------------------------------------------------------
def check_python() -> None:
    _section("Python interpreter")
    major, minor = sys.version_info[:2]
    _ok(f"Python {sys.version.split()[0]} ({sys.executable})")
    if (major, minor) != (3, 11):
        print(
            f"[WARN] Python 3.11.7 is the tested version; "
            f"detected {major}.{minor}. Experiments may still work but "
            "we cannot guarantee dependency compatibility."
        )


def check_core_packages() -> None:
    _section("Core Python packages")
    required = {
        "torch": "PyTorch",
        "transformers": "HuggingFace Transformers",
        "bitsandbytes": "bitsandbytes (LLM.int8 / FP4 / NF4 kernels)",
        "accelerate": "HuggingFace Accelerate",
        "peft": "HuggingFace PEFT",
        "datasets": "HuggingFace Datasets",
        "huggingface_hub": "HuggingFace Hub client",
    }
    missing = []
    for mod, label in required.items():
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "unknown")
            _ok(f"{label:42s} {mod}=={ver}")
        except ImportError as exc:  # pragma: no cover - hard fail path
            _fail(f"missing dependency '{mod}' ({label}): {exc}")
            missing.append(mod)
    if missing:
        raise RuntimeError(
            "Some required packages are not installed. "
            "Run `pip install -r requirements.txt` from the project root."
        )


def check_cuda() -> None:
    _section("CUDA / GPU")
    import torch

    if not torch.cuda.is_available():
        _fail(
            "CUDA is not available. QuantGuard requires at least one "
            "NVIDIA GPU with a working CUDA 12.8 stack."
        )
        raise RuntimeError("torch.cuda.is_available() == False")

    n = torch.cuda.device_count()
    _ok(f"torch.version.cuda    = {torch.version.cuda}")
    _ok(f"torch.cuda.device_cnt = {n}")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024 ** 3)
        _ok(f"  GPU {i}: {props.name} ({vram_gb:.1f} GB VRAM)")

    if n < 2:
        print(
            "[WARN] The Quick Start scenarios shard each model across 2 GPUs "
            "via device_map='auto'. With only 1 GPU you may hit out-of-memory "
            "errors on phi-2."
        )


def check_quantguard_package() -> None:
    """Confirm the editable install (`pip install -e .`) succeeded."""
    _section("QuantGuard project package")
    try:
        from q_attack.backdoor_removal.bnb import (  # noqa: F401
            compute_box_4bit,
            compute_box_int8,
        )
    except ImportError as exc:
        _fail(
            "Cannot import `q_attack.backdoor_removal.bnb`. "
            "Did you forget `pip install -e .` from the repo root?"
        )
        raise exc
    _ok("q_attack.backdoor_removal.bnb is importable")


def smoke_test_bitsandbytes() -> None:
    """End-to-end micro-test: build a tiny tensor and compute INT8 / NF4 boxes."""
    _section("Smoke test: compute_box_int8 / compute_box_4bit on a dummy tensor")
    import torch
    from q_attack.backdoor_removal.bnb import compute_box_4bit, compute_box_int8

    torch.manual_seed(0)
    w = torch.randn(32, 32, device="cuda")

    box_min_i8, box_max_i8 = compute_box_int8(original_w=w)
    assert box_min_i8.shape == w.shape and box_max_i8.shape == w.shape, (
        "INT8 quantization box has unexpected shape"
    )
    assert (box_min_i8 <= box_max_i8).all(), "INT8 box_min should be <= box_max"
    _ok(f"compute_box_int8 -> shapes {tuple(box_min_i8.shape)} (min <= max ✓)")

    box_min_nf4, box_max_nf4 = compute_box_4bit(original_w=w, method="nf4")
    assert box_min_nf4.shape == w.shape and box_max_nf4.shape == w.shape, (
        "NF4 quantization box has unexpected shape"
    )
    assert (box_min_nf4 <= box_max_nf4).all(), "NF4 box_min should be <= box_max"
    _ok(f"compute_box_4bit (nf4) -> shapes {tuple(box_min_nf4.shape)} (min <= max ✓)")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> int:
    print("QuantGuard environment verification")
    print("-----------------------------------")
    try:
        check_python()
        check_core_packages()
        check_cuda()
        check_quantguard_package()
        smoke_test_bitsandbytes()
    except Exception:  # pragma: no cover
        print()
        traceback.print_exc()
        _fail(
            "Environment verification did NOT complete successfully. "
            "Please address the error above and re-run "
            "`python scripts/verify_env.py`."
        )
        return 1

    print()
    print("[Success] All components work fine. Go ahead to run experiments!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
