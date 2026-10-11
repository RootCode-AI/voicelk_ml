# Security policy

## Supported scope

Security fixes target the current `main` branch. There is no maintained backport schedule for older snapshots or research checkpoints. Review covers the API and repository tooling; third-party dependencies also need upstream updates.

This is a research and development project, with no guaranteed security response time.

## Report a vulnerability privately

Use [GitHub private vulnerability reporting](https://github.com/RootCode-AI/voicelk_ml/security/advisories/new) if the repository offers it.

If that option is unavailable, open a public issue with only the title **Request for private security contact** and a request for a confidential reporting channel. Do not include the vulnerability, reproduction steps, affected private systems, or sensitive evidence. Wait for a maintainer to establish a private channel before sharing details. There is currently no dedicated security email published in this policy.

In the private report, include:

- Affected revision and component.
- Impact and relevant environment.
- Minimal reproduction steps using synthetic data.
- Suggested mitigation, if known.
- Whether disclosure has already occurred elsewhere.

Do not use public issues or pull requests to disclose an unpatched vulnerability. Do not include live credentials, private recordings, or learner information.

## Response process

Maintainers should acknowledge private reports, assess impact, agree on next steps with the reporter, and coordinate a fix and disclosure. Timelines depend on severity and available maintainers. Ask through the same private channel for updates.

Test only systems and data you are authorized to access. Avoid service disruption and collection of other people's information.

## Deployment considerations

The current entry point starts Flask with debugging enabled and binds to all network interfaces. It also enables general CORS access, has no authentication or request size/rate limits configured, and may return exception text.

Before production use, disable debugging, use a production WSGI server, restrict network and CORS access, validate input types, enforce request limits, and handle errors without exposing internals. Provide authentication and authorization through the backend or gateway.

Load only trusted checkpoints: deserialization of untrusted model files can execute code. Keep dependencies updated and review dataset and artifact provenance. Avoid logging raw user text and sensitive audio.
