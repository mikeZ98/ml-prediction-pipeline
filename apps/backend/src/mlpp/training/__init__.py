"""Training stack: the PyTorch network, its fit loop, ONNX export and experiment tracking.

Requires the `train` extra. Deliberately re-exports nothing: importing a submodule
should cost exactly what that submodule needs, so `tracking` stays usable without
importing torch.
"""
