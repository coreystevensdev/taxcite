FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir ".[eval]"

COPY src/ src/
COPY eval/ eval/

EXPOSE 8000

CMD ["python", "-m", "taxcite", "serve", "--host", "0.0.0.0", "--port", "8000"]
