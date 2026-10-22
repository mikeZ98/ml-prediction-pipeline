"""Serving: ONNX Runtime scoring and the HTTP service around it.

Nothing in this package imports the training framework. `onnx_model` needs only
`onnxruntime`; `service` and `api` add the `serve` extra (FastAPI, uvicorn).
"""
