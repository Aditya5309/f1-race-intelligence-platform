# Security Policy

## Scope and trust posture

This project is a **public, read-only demonstration deployment**, not a
multi-user production service handling sensitive data. Every API route is
`GET` except `POST /predict`, which only ever accepts an identity payload
(year/round, optional entry-list override) — never feature values, and
never a write to anything the deployment doesn't already control — there
is no user-account system, and nothing served is private (the underlying
data is public Formula 1 historical record). Authentication is
intentionally not implemented for this reason; see
[README.md's API Usage section](README.md#10-api-usage) for the full
reasoning.

That said, real security hygiene still applies: dependency vulnerabilities,
accidentally committed secrets, and injection-style bugs in request
handling are all things this project actively guards against.

## Supported versions

Only the latest commit on the `main` branch is supported. There is no
long-term-support branch.

## Automated protections already in place

- **Secret scanning** (gitleaks) runs on every push and pull request and
  blocks the build if it finds anything — verified clean against the
  repository's full history before being enabled.
- **Dependency vulnerability scanning** (pip-audit) runs on every push and
  pull request. It is currently report-only, not a hard gate: a scan of
  this project's actual dependency set found known vulnerabilities only in
  *transitive* packages, none of them this project's own direct pins —
  most come from the local development/notebook tooling or the
  training-side MLflow stack, neither of which is ever part of the
  deployed API or dashboard. The few findings that are on the actually
  served path (image/plotting and multipart-parsing libraries pulled in by
  the dashboard and API frameworks) are a tracked, open item — see the
  Roadmap in README.md.
- **Automated dependency updates** (Dependabot) opens weekly pull requests
  for Python, GitHub Actions, and Docker base-image updates. None are
  auto-merged; every one runs the full CI suite and is reviewed like any
  other change before merging.
- **CORS** defaults to denying all cross-origin browser access
  (`F1_CORS_ALLOW_ORIGINS` is empty by default) and is explicitly
  configurable, never left to a permissive default.
- **Error handling** never leaks internal detail: an unexpected server
  error always returns a generic `{"detail": "Internal server error."}`
  response; the real cause is logged server-side only.

## Reporting a vulnerability

If you find a genuine security issue (a real secret committed to history,
an injection vulnerability, an auth bypass on a route that's meant to be
restricted, or similar), please report it privately rather than opening a
public issue:

1. Open a [GitHub Security Advisory](../../security/advisories/new) on this
   repository (preferred — keeps the report private until resolved), or
2. Contact the maintainer directly via the email address on their GitHub
   profile.

Please include enough detail to reproduce the issue. This is a portfolio
project maintained by one person, not a funded security team — expect a
best-effort response, not a formal SLA.

## What is explicitly out of scope

- Reports about the *absence* of authentication on the demonstration API —
  this is a documented, deliberate design decision (see above), not a
  vulnerability.
- Reports about known, already-tracked dependency vulnerabilities in
  development-only or training-only tooling that never reaches the
  deployed serving path (see "Automated protections" above) — these are
  already tracked; a new report without a concrete exploitation path
  against the actual deployment won't add new information.
