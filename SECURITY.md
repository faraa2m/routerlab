# Security Policy

## Supported versions

Security fixes are applied to the latest published `@routerlab/*` packages.

## Reporting a vulnerability

Use GitHub private vulnerability reporting for this repository, or contact the
maintainer listed in `package.json` if private reporting is unavailable.

Do not open a public issue for vulnerabilities involving API keys, prompt data,
gateway request handling, or CI/package-publishing credentials.

## Security model

routerlab can sit in front of LLM API calls and may process prompts supplied by
applications. Keep credential handling explicit, avoid logging secrets, and
prefer opt-in persistence for prompts or provider responses.
