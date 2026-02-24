"""
GPU availability check for GO2Vec pipeline.
RAPIDS (cudf/cugraph/cuml) and PyTorch must see a CUDA device; no CPU fallback.
"""
import sys


def require_torch_cuda():
    """Require PyTorch with CUDA. Exit with error if not available."""
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is not installed. Install with CUDA support.", file=sys.stderr)
        sys.exit(1)
    if not torch.cuda.is_available():
        print("ERROR: PyTorch CUDA is not available. Pipeline requires GPU.", file=sys.stderr)
        sys.exit(1)
    return torch.device("cuda")


def require_rapids_gpu():
    """Require RAPIDS (cudf/cugraph) to see GPU. Exit with error if not."""
    try:
        import cudf
    except ImportError:
        print("ERROR: cudf (RAPIDS) is not installed. Install RAPIDS for your CUDA version.", file=sys.stderr)
        sys.exit(1)
    # Create a tiny GPU allocation to confirm CUDA works
    try:
        df = cudf.DataFrame({"a": [1, 2, 3]})
        _ = df["a"].sum()  # trigger GPU op
    except Exception as e:
        print(f"ERROR: RAPIDS GPU operation failed: {e}", file=sys.stderr)
        sys.exit(1)
    return True


def require_gpu():
    """Require both PyTorch CUDA and RAPIDS GPU. Call at start of GPU steps."""
    require_torch_cuda()
    require_rapids_gpu()
