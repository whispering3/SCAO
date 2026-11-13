# Security policy

## Supported version

Security fixes are applied to the latest released SCAO version. Development
snapshots on `main` may contain unreleased behavior and should not be treated
as a production distribution.

## Reporting a vulnerability

Do not open a public issue for a suspected security vulnerability. Report it
privately to the repository owner through GitHub's security advisory reporting
flow, including a minimal reproduction, affected version, impact, and any
workaround you have identified.

We will acknowledge the report, assess impact, and coordinate disclosure and a
fix before publishing technical details.

## Supply-chain practices

Release workflows use PyPI trusted publishing and pin GitHub Actions to commit
SHAs. Dependency updates are tracked by Dependabot. Consumers should install
published artifacts from PyPI and verify their own environment constraints.
