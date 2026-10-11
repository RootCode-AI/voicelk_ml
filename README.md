# VoiceLK ML

Python service and research tools for adaptive Sinhala text-to-speech, focused on Sri Lanka's G.C.E. O/L ICT curriculum.

VoiceLK aims to make Sinhala educational material easier to access through speech, including text that mixes Sinhala and English ICT terminology. This repository contains the text-processing API, pronunciation lexicon, dataset preparation, VITS training and synthesis scripts, and evaluation tools.

## Current capabilities

- Normalize Sinhala-English text, numbers, symbols, and common ICT terms.
- Convert normalized text to an International Phonetic Alphabet (IPA) sequence.
- Serve the text-processing pipeline through a Flask HTTP API.
- Prepare speech datasets and train VITS models using research scripts.
- Synthesize audio offline from a checkpoint and its matching configuration.
- Audit pronunciation, alignment, and synthesis quality with evaluation tools.

The HTTP API currently returns text and IPA, not audio. Training, synthesis, and evaluation are separate workflows; their presence does not imply validated production speech quality.

## Architecture

The wider VoiceLK platform uses a React frontend and a Java/Spring backend. The backend calls this Python service to process text. Authentication, authorization, sessions, and database storage belong to the surrounding application.

The processing sequence is: input text → normalization → code-switched grapheme-to-phoneme conversion → IPA. Offline VITS synthesis then uses a compatible trained model to produce audio.

## Repository map

| Path | Purpose |
| --- | --- |
| [api/main.py](api/main.py) | Flask application and HTTP endpoints |
| [model_engine/](model_engine/) | Normalizer, G2P engine, pipeline, and domain lexicon |
| [data_preparation/](data_preparation/) | Dataset preparation and metadata validation |
| [model_training/](model_training/) | VITS training, vocabulary tools, and offline synthesis |
| [model_evaluate/](model_evaluate/) | Evaluation and pipeline audits |
| [model_finetune/](model_finetune/) | Training-free diagnostics and evaluation reproducibility work |
| [tests/test_api.py](tests/test_api.py) | Manual request script for a running local service |
| [docs/](docs/) | Project documentation and repository administration guidance |

Training dependencies and vendored Coqui components are separate from the lightweight API environment. Data and model artifacts may need to be obtained separately; inspect each script's paths before running it.

## Run the text-processing API

Use a dedicated Python virtual environment. Python 3.11 is a practical starting point for the wider research toolchain; compatibility with every Python version is not guaranteed.

```bash
git clone https://github.com/RootCode-AI/voicelk_ml.git
cd voicelk_ml
python -m venv venv
```

Activate it in PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source venv/bin/activate
```

Install the API dependencies and run the service:

```bash
python -m pip install -r requirements.txt
python api/main.py
```

Access it at `http://127.0.0.1:5000`. No `.env` file is required by the current entry point.

**Development only:** this command enables Flask debug mode and binds to `0.0.0.0`. Run it on a trusted development machine. Production deployment needs a production WSGI server, disabled debugging, authentication at the gateway/backend, request limits, and a restricted CORS policy. See [Security](SECURITY.md).

## API

### GET /health

Returns HTTP 200:

```json
{"status": "healthy", "service": "VoiceLK ML Service"}
```

### POST /api/generate-ipa

Send a JSON object with a non-empty string in `text`:

```json
{"text": "RAM සහ CPU"}
```

A successful response contains `status`, `raw_input`, `normalized_text`, and `ipa_sequence`. Pronunciation depends on the normalizer, rules, and lexicon; do not treat an example output as a quality guarantee.

Missing or blank text returns HTTP 400 with:

```json
{"status": "error", "message": "'text' field is required and must not be empty."}
```

Processing failures return HTTP 500. Clients should send the documented JSON shape; other input types are not comprehensively validated.

With the service running, try this in another PowerShell window:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/health
$body = @{ text = "RAM and CPU" } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:5000/api/generate-ipa -Method Post -ContentType "application/json" -Body $body
```

The existing manual request script needs the extra `requests` package:

```bash
python -m pip install requests
python tests/test_api.py
```

It prints results and does not contain automated assertions.

## Training, synthesis, and evaluation

Install [training requirements](model_training/requirements.txt) in a separate environment. Review dataset paths, metadata, sample rate, compute requirements, and source permissions before running any training script.

For an existing compatible VITS model:

```bash
python model_training/synthesize.py --model_dir /path/to/model --text "RAM and CPU" --out output.wav
```

The model directory must contain a checkpoint and its matching `config.json`. Availability and permitted use of weights and recordings depend on their sources.

See [diagnostic workflow instructions](model_finetune/README.md) for the evaluation work. Record checkpoint identity, configuration, seeds, dataset split, and metric settings when comparing models.

## Contribute and participate

Bug reports, pronunciation corrections, documentation, accessibility feedback, and reproducible research contributions are welcome.

- [Contributing guidelines](CONTRIBUTING.md)
- [Code of conduct](CODE_OF_CONDUCT.md)
- [Code of ethics](CODE_OF_ETHICS.md)
- [Accessibility commitment and reporting](ACCESSIBILITY.md)
- [Security policy and private reporting](SECURITY.md)
- [Issue templates](https://github.com/RootCode-AI/voicelk_ml/issues/new/choose)
- [Repository administrator checklist](docs/REPOSITORY_ADMIN.md)

## License and third-party materials

Original VoiceLK code and documentation are available under the [MIT License](LICENSE). Vendored code, datasets, recordings, and model weights are not relicensed by that grant. See [third-party notices](THIRD_PARTY_NOTICES.md) for scope and provenance limitations.

Speaker consent, dataset permissions, and research integrity are covered by the [Code of Ethics](CODE_OF_ETHICS.md). These community expectations do not add restrictions to the MIT license.
