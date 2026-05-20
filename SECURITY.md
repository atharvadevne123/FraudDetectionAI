# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x     | :white_check_mark: |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Email devneatharva@gmail.com with:

1. A description of the vulnerability and its potential impact
2. Steps to reproduce the issue
3. Any relevant logs, screenshots, or proof-of-concept code

You can expect an acknowledgment within 48 hours and a resolution timeline within 7 days for critical issues.

## Security Considerations

### API Authentication

All endpoints (except `/health`, `/readiness`, `/version`, and `/metrics`) require a valid API key passed via the `X-API-Key` header. Keys are validated using `hmac.compare_digest` to prevent timing attacks.

### Input Validation

Transaction payloads are validated via marshmallow schemas before processing. Invalid or malformed inputs are rejected with HTTP 422.

### Rate Limiting

The API enforces per-IP rate limits (default: 200/day, 50/hour) using Flask-Limiter with Redis backend. Adjust limits in `config/settings.py`.

### Model Artifacts

Model files (`.joblib`) should be stored in a secure location with restricted read permissions. Never expose raw model artifacts via the API.

### Dependencies

Run `pip-audit` regularly to check for known vulnerabilities in dependencies:

```bash
pip install pip-audit
pip-audit
```

### Secrets Management

Never commit credentials, API keys, or secrets to version control. Use environment variables or a secrets manager (e.g., AWS Secrets Manager, HashiCorp Vault).
