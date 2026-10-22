# syntax=docker/dockerfile:1.7
#
# mlpp inference API — FastAPI + ONNX Runtime. Multi-stage:
#
#   builder  resolves the locked `serve` extra into /opt/venv with uv, compiles
#            bytecode, and installs mlpp itself as a regular (non-editable) wheel;
#   runtime  copies only that venv onto a clean slim image and runs as a non-root
#            user. No uv, no compiler, no source tree, no torch.
#
# Build context is apps/backend (see docker-compose.yml); the matching ignore file
# is api.Dockerfile.dockerignore next to this file.
#
#   docker build -f docker/api.Dockerfile -t mlpp-api apps/backend
#   docker run --rm -p 127.0.0.1:8100:8100 \
#     -v "$PWD/OUTPUTS/example:/models/session:ro" mlpp-api

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.11.7

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ---------------------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

COPY --from=uv /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /src

# Dependencies first, from the lockfile alone, so this layer survives source edits.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --extra serve --no-install-project

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra serve --no-editable

# Fail the build, not the deploy, if the serving path would pull in training code.
RUN /opt/venv/bin/python -c "\
import importlib.util, sys; \
import mlpp.inference.api; \
leaked = [m for m in ('torch', 'mlflow', 'tensorflow') if importlib.util.find_spec(m)]; \
sys.exit(f'training stack leaked into the serving image: {leaked}' if leaked else 0)"

# ---------------------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ARG APP_UID=10001
RUN groupadd --system --gid ${APP_UID} mlpp \
 && useradd --system --uid ${APP_UID} --gid mlpp --no-create-home \
      --home-dir /nonexistent --shell /usr/sbin/nologin mlpp

COPY --from=builder /opt/venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Service settings (mlpp.inference.settings); override at run time.
    MLPP_SESSION_DIR=/models/session \
    # uvicorn reads UVICORN_* for every CLI option.
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8100 \
    UVICORN_WORKERS=1 \
    UVICORN_PROXY_HEADERS=1 \
    UVICORN_SERVER_HEADER=0 \
    UVICORN_LOG_LEVEL=info

LABEL org.opencontainers.image.title="mlpp-api" \
      org.opencontainers.image.description="CNN + Bi-GRU regressor served via FastAPI and ONNX Runtime" \
      org.opencontainers.image.source="https://github.com/mikeZ98/ml-prediction-pipeline" \
      org.opencontainers.image.licenses="MIT"

USER mlpp
WORKDIR /tmp
EXPOSE 8100

# Readiness, not liveness: the container is healthy once the model is loaded.
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.environ.get('UVICORN_PORT', '8100')}/health/ready\", timeout=2)"]

CMD ["uvicorn", "mlpp.inference.api:create_app", "--factory"]
