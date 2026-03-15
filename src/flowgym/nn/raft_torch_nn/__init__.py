try:
    import torch  # noqa: F401
except ImportError:
    raise ImportError(
        "raft_torch_nn requires PyTorch. Please install PyTorch to use this module."
    )
