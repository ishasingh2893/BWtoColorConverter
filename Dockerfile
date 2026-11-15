FROM public.ecr.aws/lambda/python:3.11

# Lambda Web Adapter lets the existing Flask/Gunicorn application run as an
# HTTP service inside a Lambda container image.
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 /lambda-adapter /opt/extensions/lambda-adapter

COPY requirements-runtime.txt .
RUN python -m pip install --no-cache-dir -r requirements-runtime.txt

COPY . .

ENV AWS_LWA_PORT=8080 \
    AWS_LWA_READINESS_CHECK_PATH=/

CMD ["gunicorn", "--bind=0.0.0.0:8080", "--workers=1", "--threads=2", "--timeout=900", "app:app"]
