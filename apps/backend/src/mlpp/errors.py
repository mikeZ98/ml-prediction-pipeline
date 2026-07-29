"""Exception hierarchy. Every failure the pipeline raises is one of these."""


class MlppError(Exception):
    """Base class for all pipeline errors."""


class SchemaError(MlppError):
    """A dataframe is missing columns the configured schema requires."""


class DatasetError(MlppError):
    """A dataset directory is empty, unreadable, or yields no usable rows."""


class NotFittedError(MlppError):
    """A Preprocessor was used to transform before `fit` was called."""


class ArtifactError(MlppError):
    """A persisted artifact is missing or structurally invalid."""


class SchemaVersionError(ArtifactError):
    """A session was written by a different version of the artifact contract.

    Distinct from a corrupt artifact: the file is well-formed, this build just
    cannot honestly interpret it. Artifacts are cheap to regenerate, so the
    contract rejects rather than coerces.
    """
