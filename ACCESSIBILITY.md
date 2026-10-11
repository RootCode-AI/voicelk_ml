# Accessibility

VoiceLK aims to improve access to Sinhala educational content through speech. Accessibility applies to documentation, contribution workflows, pronunciation, and interfaces built around the service.

## Scope and current limits

This repository primarily provides a Python API and research tools. It does not implement the platform's learner interface. There has been no comprehensive accessibility audit, and this statement is not a claim of conformance or validated assistive-technology support.

IPA output and research notebooks may be difficult to use with screen readers. Synthetic speech can mispronounce mixed-language text, numbers, abbreviations, and dialect-specific words. Audio alone does not make an experience accessible.

## Contribution guidelines

- Write clear instructions with logical headings and descriptive link text.
- Preserve readable Sinhala Unicode and provide explanations for unfamiliar IPA notation.
- Give diagrams a text explanation and images meaningful alternative text.
- Provide transcripts for speech samples and captions/transcripts for video demonstrations.
- Do not communicate results through color, images, or audio alone.
- Include text summaries of notebook charts and evaluation results.
- Where a change includes a web interface, target WCAG 2.2 Level AA and check keyboard operation, visible focus, labels, contrast, zoom/reflow, and screen-reader behavior.
- Preserve source text alongside generated speech where the consuming interface supports it, and make playback user-controlled.

WCAG guidance informs web content; it does not certify API pronunciation quality. Review educational pronunciation with Sinhala-speaking users and include people who use assistive technology in feedback where possible.

## Report an accessibility barrier

Use the [accessibility issue form](https://github.com/RootCode-AI/voicelk_ml/issues/new?template=accessibility.yml).

Describe the task, affected file or interface, what prevented progress, and a preferred outcome or workaround. Environment and assistive-technology versions are helpful but optional. You do not need to disclose a disability or diagnosis.

Public reports must not include private learner data or recordings. Use the [security reporting process](SECURITY.md) if the issue exposes private information.

## Ownership and follow-up

Repository maintainers own accessibility triage. They should prioritize barriers that prevent core tasks, discuss workarounds, record remaining limitations, and ask whether the fix addresses the reporter's experience. There is no fixed resolution deadline or automated accessibility gate currently promised by this project.

This statement draws on the [Open Source Accessibility Toolkit](https://github.com/open-source-accessibility/accessibility-toolkit/blob/main/ACCESSIBILITY.md) and [WCAG 2.2](https://www.w3.org/TR/WCAG22/), with scope adapted to VoiceLK's API and speech research.
