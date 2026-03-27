#!/usr/bin/env python3
"""
CLI entry point — generate a Malagasy audio story.

Usage:
    python -m src.generate --theme "ny ampondra"
    python -m src.generate --theme "ny akoho" --output my_story.wav
    python -m src.generate --theme "ny zaza" --gradio  # Launch web UI
"""

import argparse
import sys
from pathlib import Path

from .generator import StoryGenerator
from .tts import MalagasyTTS


def generate_story(theme: str, output_dir: str = "outputs", text_only: bool = False) -> dict:
    """Generate a full audio story. Returns dict with text and audio path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Clean filename from theme
    safe_name = theme.replace(" ", "_").replace("/", "_")[:50]

    # Step 1: Generate story text
    print(f"\n📖 Generating Malagasy story about: {theme}\n")
    text_gen = StoryGenerator()
    story = text_gen.generate_story(theme)
    print(f"\n--- Generated Story ---\n{story}\n-----------------------\n")

    # Save text
    text_path = output_path / f"{safe_name}.txt"
    text_path.write_text(story, encoding="utf-8")
    print(f"📝 Text saved to {text_path}")

    result = {"theme": theme, "text": story, "text_path": str(text_path)}

    if text_only:
        return result

    # Step 2: Convert to audio
    print("\n🔊 Converting to audio...\n")
    tts = MalagasyTTS()
    audio_path = output_path / f"{safe_name}.wav"
    tts.synthesize_long(story, audio_path)

    result["audio_path"] = str(audio_path)
    return result


def launch_gradio():
    """Launch a simple web UI for story generation."""
    import gradio as gr

    text_gen = StoryGenerator()
    tts = MalagasyTTS()

    def generate(theme):
        story = text_gen.generate_story(theme)
        audio_path = Path("outputs") / f"{theme.replace(' ', '_')[:30]}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        tts.synthesize_long(story, audio_path)
        return story, str(audio_path)

    demo = gr.Interface(
        fn=generate,
        inputs=gr.Textbox(label="Lohateny / Theme", placeholder="Ohatra: ny ampondra"),
        outputs=[
            gr.Textbox(label="Tantely / Story"),
            gr.Audio(label="Feo / Audio"),
        ],
        title="🇲🇬 Tantely Malagasy ho an'ny ankizy",
        description="Manoratra sy mameno tantely fohy ho an'ny ankizy amin'ny teny Malagasy",
    )
    demo.launch(server_name="0.0.0.0", server_port=7860)


def main():
    parser = argparse.ArgumentParser(description="Generate Malagasy audio stories for kids")
    parser.add_argument("--theme", type=str, help="Story theme (e.g., 'ny ampondra')")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    parser.add_argument("--text-only", action="store_true", help="Only generate text, skip TTS")
    parser.add_argument("--gradio", action="store_true", help="Launch Gradio web UI")

    args = parser.parse_args()

    if args.gradio:
        launch_gradio()
        return

    if not args.theme:
        parser.error("--theme is required (or use --gradio for web UI)")

    result = generate_story(args.theme, args.output, args.text_only)

    print("\n✅ Done!")
    print(f"   Theme: {result['theme']}")
    print(f"   Text:  {result['text_path']}")
    if "audio_path" in result:
        print(f"   Audio: {result['audio_path']}")


if __name__ == "__main__":
    main()
