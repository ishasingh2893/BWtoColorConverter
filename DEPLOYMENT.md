# AWS Lambda deployment

This repository deploys the existing Flask app as an AWS Lambda container image. The same Lambda Function URL serves both the Flask UI and its API, so this deployment has no dependency on Sites. `requirements-runtime.txt` contains only the packages needed by the deployed app; the full `requirements.txt` remains for local training and development.

## One-time AWS setup

Create a Lambda execution role with CloudWatch Logs permissions. The GitHub Actions workflow creates the ECR repository, Lambda function, and public Function URL on its first run. Give the Lambda enough memory for OpenCV/NumPy (the workflow uses 3008 MB) and a timeout appropriate for image processing.

- `https://ishasingh.cv`
- `https://www.ishasingh.cv`

The `.npz` model files must be committed because they are required at runtime. They are intentionally checked by the workflow before building.

## GitHub configuration

Add this repository variable configuration:

- `AWS_REGION`
- `ECR_REPOSITORY`
- `LAMBDA_FUNCTION_NAME`
- `LAMBDA_EXECUTION_ROLE_ARN`

Add the GitHub Actions secret `AWS_DEPLOY_ROLE_ARN`. The role should trust GitHub Actions OIDC for this repository and allow ECR push plus `lambda:UpdateFunctionCode`.

Every push to `main` builds a new container image, creates or updates the Lambda function, and prints its Function URL.

## Important request-size note

The current Flask app accepts files up to 12 MB, but synchronous Lambda invocation responses are limited to 6 MB. For production use, the next hardening step should upload source and result images through S3 and pass object keys to Lambda instead of returning large base64 payloads directly.
