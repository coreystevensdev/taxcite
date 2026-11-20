FROM python:3.12-slim

# uv, pinned rather than :latest so the image is reproducible. It installs from
# uv.lock, which means the container holds exactly the tree the lock records and
# the dependency graph scans. pip resolved pyproject's constraints fresh on every
# build instead, so the scanned tree and the shipped tree were only ever assumed
# to match.
#
# --locked and not --frozen. Both install from the lock, but only --locked fails
# when the lock no longer matches pyproject; --frozen exits 0 and quietly uses the
# stale one. Measured, not read off the flag names.
COPY --from=ghcr.io/astral-sh/uv:0.8.9 /uv /usr/local/bin/uv

WORKDIR /app

# Copy mode because there is no cache mount to hardlink against. Bytecode is
# deliberately NOT precompiled: it costs about 90MB per image, and the box these
# demos share has 16GiB for six of them and already filled up once. Python writes
# the .pyc on first import instead, once per container start.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=0 \
    PATH="/app/.venv/bin:$PATH"

# src lands before the install, not after. Hatchling is told the package is at
# src/taxcite, so installing against an absent source directory produced an
# image that built cleanly and then could not import its own package.
COPY pyproject.toml uv.lock ./
COPY src/ src/

# No extras. The eval extra used to be installed here, which put ragas, datasets
# and the two judge packages in a serving image that never calls them:
# __main__.cmd_eval imports eval_harness lazily, and weekly-eval.yml runs the
# eval on the GitHub runner rather than in this container.
RUN uv sync --locked --no-editable

COPY eval/ eval/

EXPOSE 8000

CMD ["python", "-m", "taxcite", "serve", "--host", "0.0.0.0", "--port", "8000"]
