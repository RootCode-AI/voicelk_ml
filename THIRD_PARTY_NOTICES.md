# Third-party materials and license scope

The root [MIT License](LICENSE) covers original VoiceLK code and documentation to the extent contributors hold the rights to license them. It does not replace third-party licenses or grant rights to recordings, datasets, model checkpoints, speaker identities, or trademarks.

## Vendored software

| Location | Identified source | License and provenance |
| --- | --- | --- |
| `model_training/TTS/` | [Coqui TTS](https://github.com/coqui-ai/TTS), local VERSION: 0.22.0 | Upstream uses [MPL-2.0](https://github.com/coqui-ai/TTS/blob/dev/LICENSE.txt). Applicable upstream and file-specific notices remain in force. |
| `model_training/trainer/` | [Coqui Trainer](https://github.com/coqui-ai/Trainer), local VERSION: v0.0.36 | Preserve the license and notices from the exact upstream distribution; confirm them before redistribution. |

The local version files identify declared versions, not verified source commits. The repository did not contain license files for these vendored trees when this notice was added. This notice records that gap; it is not a replacement for the upstream license texts.

Before distributing these components, maintainers must recover the license and copyright notices from the exact source distributions, record their provenance and any local changes, and retain file-specific notices. Do not describe the entire repository as exclusively MIT-licensed.

## Dependencies

Packages installed through the requirements files keep their respective licenses. A dependency's inclusion does not transfer ownership to VoiceLK.

## Data, voices, and model artifacts

Record each source, version, license, attribution requirement, and permission for training and redistribution. Public availability, including on video platforms, is not evidence of permission to train or reproduce a person's voice.

Do not assume the root MIT license covers metadata, recordings, sample audio, third-party lexicons, or pretrained weights. If permission or provenance is unclear, do not redistribute the affected material until it is resolved.
