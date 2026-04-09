# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim as base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Create a proper non-root user WITH home
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --uid "${UID}" \
    appuser

# Set correct HOME and cache
ENV HOME=/home/appuser
ENV UV_CACHE_DIR=/home/appuser/.cache/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN --mount=type=cache,target=/home/appuser/.cache/uv \
    uv sync --frozen --no-install-project

# Copy source
COPY . .

# Install project
RUN --mount=type=cache,target=/home/appuser/.cache/uv \
    uv sync --frozen

# Switch user
USER appuser

EXPOSE 7860

CMD ["uv", "run", "python", "main.py"]