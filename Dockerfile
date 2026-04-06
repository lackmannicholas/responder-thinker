FROM --platform=linux/amd64 python:3.14-slim

# Patch OS-level vulnerabilities in the base image; install libc++ for ten_vad native lib
RUN apt-get update && apt-get upgrade -y --no-install-recommends \
  && apt-get install -y --no-install-recommends libc++1 \
  && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy app code AFTER uv sync so the local .venv (excluded via .dockerignore)
# never overwrites the Linux venv built above
COPY . .

# Use the venv directly — avoids `uv run` re-validating the interpreter at startup
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
