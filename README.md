# 🇲🇬 Malagasy Kids Stories — Audio Generator

Generate short audio stories for kids in Malagasy language.

## Architecture (MVP — Option A)

**Two-stage pipeline:**

1. **Text Generation** — `Lo-Renz-O/Mistral-7B-instruct-Malagasy-bnb-4bit`
   - Mistral-7B fine-tuned for Malagasy instruction following
   - Generates short children's stories from a prompt/theme

2. **Text-to-Speech** — `hasiniaina/mms_malagasy_finetuning`
   - VITS-based model (36M params) from Meta's MMS project
   - Converts Malagasy text to audio

## Requirements

- Python 3.10+
- GPU recommended (4-6GB VRAM for text model, quantized)
- TTS model can run on CPU

## Setup

```bash
cd malagasy-stories
pip install -r requirements.txt
```

## Usage

```bash
python -m src.generate --theme "ny ampondra" --output outputs/
```

## Project Structure

```
malagasy-stories/
├── src/
│   ├── __init__.py
│   ├── generator.py    # Text story generation
│   ├── tts.py          # Text-to-speech conversion
│   └── generate.py     # CLI entry point
├── outputs/            # Generated audio files
├── voices/             # Voice references (if needed)
├── requirements.txt
└── README.md
```

## License

Models used are under CC BY-NC 4.0 — this project is for non-commercial use.
