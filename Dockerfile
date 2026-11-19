FROM python:3.12-slim

WORKDIR /app

# src lands before the install, not after. Hatchling is told the package is at
# src/taxcite, so installing against an absent source directory produced an
# image that built cleanly and then could not import its own package.
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir ".[eval]"

COPY eval/ eval/

EXPOSE 8000

CMD ["python", "-m", "taxcite", "serve", "--host", "0.0.0.0", "--port", "8000"]
