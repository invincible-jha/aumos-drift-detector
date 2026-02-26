# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

**Do not report security vulnerabilities through public GitHub issues.**

Please report security vulnerabilities by emailing: security@aumos.io

Include in your report:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if known)

You will receive an acknowledgment within 48 hours, and a detailed response within
5 business days.

## Security Considerations

`aumos-drift-detector` processes ML model telemetry and triggers retraining workflows.
Security issues here may affect model integrity and the automated retraining pipeline.

Key security-sensitive areas:

- **Reference data ingestion**: Validate and sanitise all incoming feature data to
  prevent injection attacks through data payloads
- **Retraining triggers**: The `drift.retraining_required` Kafka event must be
  authenticated — ensure consumers validate tenant_id before acting
- **Multi-tenant isolation**: All drift monitors and detections are tenant-scoped via
  Row-Level Security — cross-tenant data access is a critical vulnerability
- **Statistical result tampering**: Drift scores influence automated retraining decisions;
  ensure the detection pipeline cannot be manipulated by crafted input data

## Security Best Practices for Operators

1. Set `AUMOS_DRIFT_DEBUG=false` in production
2. Rotate `AUMOS_DRIFT_KEYCLOAK__CLIENT_SECRET` regularly
3. Restrict Kafka topic ACLs — only this service should publish to `drift.*` topics
4. Use TLS (`AUMOS_DRIFT_KAFKA__SECURITY_PROTOCOL=SSL`) in production Kafka
5. Validate reference data URIs before fetching — prevent SSRF via `reference_data_uri`
6. Apply network policy so only `aumos-mlops-lifecycle` can consume from `drift.*` topics

## Disclosure Policy

- We will acknowledge receipt of your report within 48 hours
- We will confirm the vulnerability and determine its severity within 5 business days
- We will release a patch as soon as possible, prioritised by severity
- We will publicly disclose the vulnerability after the patch is released
- We will credit reporters who wish to be acknowledged
