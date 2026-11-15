FROM python:3.11-slim-bookworm

# Lambda Web Adapter lets the existing Flask/Gunicorn application run as an
# HTTP service inside a Lambda container image.
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.1.0 /lambda-adapter /opt/extensions/lambda-adapter

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-runtime.txt .
RUN python -m pip install --no-cache-dir -r requirements-runtime.txt

COPY . .

ENV AWS_LWA_PORT=8080 \
    AWS_LWA_READINESS_CHECK_PATH=/

CMD ["gunicorn", "--bind=0.0.0.0:8080", "--workers=1", "--threads=2", "--timeout=900", "app:app"]
