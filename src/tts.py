"""
Text-to-Speech using Meta MMS Malagasy VITS model.
Uses hasiniaina/mms_malagasy_finetuning.
"""

import torch
import soundfile as sf
from pathlib import Path
from transformers import AutoTokenizer, VitsModel

MODEL_ID = "hasiniaina/mms_malagasy_finetuning"


class MalagasyTTS:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading TTS model on {self.device}...")

        self.model = VitsModel.from_pretrained(MODEL_ID).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        print("TTS model loaded.")

    def synthesize(self, text: str, output_path: str | Path) -> Path:
        """Convert Malagasy text to audio and save as WAV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model(**inputs)

        waveform = output.waveform[0].cpu().numpy()
        sample_rate = self.model.config.sampling_rate

        sf.write(str(output_path), waveform, sample_rate)
        print(f"Audio saved to {output_path} ({len(waveform)/sample_rate:.1f}s, {sample_rate}Hz)")

        return output_path

    def synthesize_long(self, text: str, output_path: str | Path, max_chunk: int = 200) -> Path:
        """
        For longer text: split into sentences, synthesize each, concatenate.
        VITS models work best on shorter utterances.
        """
        # Split on sentence-ending punctuation (Malagasy uses periods, question marks)
        sentences = []
        for part in text.replace("!", ".").replace("?", ".").split("."):
            part = part.strip()
            if part:
                sentences.append(part)

        if not sentences:
            return self.synthesize(text, output_path)

        # If individual sentences are still too long, split further by commas
        chunks = []
        for sentence in sentences:
            if len(sentence) <= max_chunk:
                chunks.append(sentence)
            else:
                sub_parts = sentence.split(",")
                current = ""
                for sp in sub_parts:
                    sp = sp.strip()
                    if len(current) + len(sp) + 2 > max_chunk and current:
                        chunks.append(current)
                        current = sp
                    else:
                        current = f"{current}, {sp}" if current else sp
                if current:
                    chunks.append(current)

        if not chunks:
            return self.synthesize(text, output_path)

        # Synthesize each chunk
        all_waveforms = []
        sample_rate = None

        for i, chunk in enumerate(chunks):
            tmp_path = output_path.parent / f"_chunk_{i}.wav"
            self.synthesize(chunk, tmp_path)

            import numpy as np
            wf, sr = sf.read(str(tmp_path))
            all_waveforms.append(wf)
            if sample_rate is None:
                sample_rate = sr
            tmp_path.unlink()

        # Concatenate with a short silence gap between chunks
        import numpy as np
        silence = np.zeros(int(sample_rate * 0.3))  # 300ms gap
        combined = []
        for i, wf in enumerate(all_waveforms):
            combined.append(wf)
            if i < len(all_waveforms) - 1:
                combined.append(silence)

        final = np.concatenate(combined)
        output_path = Path(output_path)
        sf.write(str(output_path), final, sample_rate)
        print(f"Full audio saved to {output_path} ({len(final)/sample_rate:.1f}s)")

        return output_path
