"""mlpp — time-series / tabular regression pipeline.

Import layers deliberately: `config`, `data`, `preprocess`, `metrics`, `session`,
`errors`, `importance` and `schemas` are framework-free and cheap to import.
`predict` and `inference` add ONNX Runtime only. `training`, `plots` and `pipeline`
pull in PyTorch (the `train` extra); `monitoring.drift` pulls in Evidently (the
`monitor` extra). Import those only when needed.
"""

from mlpp.errors import (
    ArtifactError,
    DatasetError,
    InferenceError,
    MlppError,
    NotFittedError,
    SchemaError,
    SchemaVersionError,
    TrainingError,
)

__version__ = "0.1.0"

__all__ = [
    "ArtifactError",
    "DatasetError",
    "InferenceError",
    "MlppError",
    "NotFittedError",
    "SchemaError",
    "SchemaVersionError",
    "TrainingError",
    "__version__",
]
