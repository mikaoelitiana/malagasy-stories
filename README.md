# 🇲🇬 Malagasy Kids Stories — Audio Generator

Generate short audio stories for kids in Malagasy language.

## Architecture (MVP)

Two-stage pipeline:
1. **Text Generation** — `Lo-Renz-O/Mistral-7B-instruct-Malagasy-bnb-4bit` (Malagasy story text)
2. **Text-to-Speech** — `hasiniaina/mms_malagasy_finetuning` (VITS, 36M params)

## 🍎 Setup on Mac (Apple Silicon)

### Prerequisites
- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- 16GB RAM recommended (8GB works but will be slow)

### Installation

```bash
# Clone the repo
git clone https://github.com/mikaoelitiana/malagasy-stories.git
cd malagasy-stories

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### First Run

Models will be downloaded automatically on first use (~14GB for the text model).

```bash
# Generate text only (fastest way to test)
python -m src.generate --theme "ny ampondra"

# Generate full audio story
python -m src.generate --theme "ny ampondra"

# Launch web UI
python -m src.generate --gradio
```

### ⚠️ Mac Performance Notes

| Mac Config | Text Gen Speed | Notes |
|------------|---------------|-------|
| M1/M2 8GB | ~30-60 sec | Close other apps, uses swap |
| M1/M2 16GB | ~15-30 sec | Comfortable |
| M3/M4 16GB+ | ~10-15 sec | Best experience |

- TTS is instant (tiny 36M model)
- No GPU needed — uses Apple's MPS (Metal) backend
- `bitsandbytes` is not required on Mac (not installed)

### Troubleshooting

**"MPS not available"** — Update PyTorch:
```bash
pip install --upgrade torch
```

**Out of memory** — Close other apps or add swap:
```bash
# Check memory pressure
memory_pressure
```

**Model download is slow** — The text model is ~7GB. First download takes a while.

## Usage

```bash
# Generate a story about "the horse"
python -m src.generate --theme "ny ampondra"

# Custom output directory
python -m src.generate --theme "ny akoho" --output my_stories/

# Text only (skip TTS, useful for testing)
python -m src.generate --theme "ny zaza" --text-only

# Web UI (Gradio)
python -m src.generate --gradio
```

## Project Structure

```
malagasy-stories/
├── src/
│   ├── __init__.py
│   ├── generator.py    # Mistral-7B Malagasy text generation
│   ├── tts.py          # MMS VITS text-to-speech
│   └── generate.py     # CLI + Gradio web UI
├── outputs/            # Generated stories (audio + text)
├── requirements.txt
└── README.md
```

## License

Models used are under CC BY-NC 4.0 — this project is for non-commercial use.
