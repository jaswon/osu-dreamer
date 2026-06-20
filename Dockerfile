FROM runpod/base:0.6.2-cuda12.4.1
ENV PYTHONUNBUFFERED=1

# screen
RUN apt-get update \
    && apt-get install -y screen \
    && rm -rf /var/lib/apt/lists/*

# uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# project deps
WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv sync --locked --no-install-project

# project data
COPY data data

# project source
RUN git init . \
    && git remote add origin https://github.com/jaswon/osu-dreamer \
    && git pull origin latent \
    && git checkout latent