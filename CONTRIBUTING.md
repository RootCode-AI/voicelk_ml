# Contributing to VoiceLK ML

We welcome contributions to Sinhala-English pronunciation, accessible educational speech, documentation, and reproducible TTS research. Read the [Code of Conduct](CODE_OF_CONDUCT.md), [Code of Ethics](CODE_OF_ETHICS.md), and [Accessibility guidelines](ACCESSIBILITY.md).

## Before starting

Search existing issues and pull requests. Use the issue templates for bugs, features, and accessibility barriers. Discuss large architecture, dataset, or dependency changes before investing in them. Small fixes can go directly to a pull request.

Report vulnerabilities using [SECURITY.md](SECURITY.md), not a public issue. Do not post private recordings, credentials, student information, or identifiable learner data.

## Development setup

1. Fork the repository and create a focused branch.
2. Follow the [README setup](README.md#run-the-text-processing-api).
3. Keep API and training dependencies in separate environments.
4. Inspect script paths and configuration before running research workflows.

The root requirements file supports the NLP API. Training dependencies are in `model_training/requirements.txt`; diagnostic dependencies are described in `model_finetune/README.md`.

## Change and validation expectations

- Match existing Python style, use clear names, and keep changes focused.
- For normalization, G2P, and lexicon changes, include input text, expected pronunciation, and the reason for the change. Cover Sinhala, English, mixed text, relevant punctuation, and affected number or acronym cases.
- Ask a fluent Sinhala reviewer to verify pronunciation changes when possible. Record unresolved dialect or terminology questions.
- For API changes, check health, valid text, missing/blank text, and any changed error behavior.
- For model comparisons, record checkpoint hashes, configuration, dataset split, seeds, hardware, and metric settings. Keep evaluation data separate from training data.
- For documentation changes, verify links, commands, and readable Unicode.
- Add focused automated regression checks for changed behavior where practical. Do not present a manual printout as an automated test pass.

The existing `tests/test_api.py` script sends a request to a running local server. Install `requests` separately, then run `python tests/test_api.py`. It prints the response and has no assertions; inspect the result and describe what you checked.

Do not run training solely to validate a documentation change.

## Data and artifacts

Submit only material you have permission to contribute. Document data provenance, terms, speaker consent, and attribution without publishing private consent records.

Keep virtual environments, credentials, large audio collections, model checkpoints, and generated outputs out of pull requests. Check `git status` and the staged diff; existing ignore rules may not cover every artifact.

Preserve third-party notices. See [license scope](THIRD_PARTY_NOTICES.md). Contributions to original VoiceLK code are submitted under the [MIT License](LICENSE); changes to vendored software remain subject to the applicable upstream terms.

## Pull requests

Use the pull request template. Explain the problem, resulting behavior, validation performed, and limitations. Link related issues. Include accessible descriptions or transcripts for relevant visuals and audio.

Maintainers may request revisions or decline changes that lack evidence, permissions, or fit with the project's goals. Respond constructively and keep review discussions focused on the work.
