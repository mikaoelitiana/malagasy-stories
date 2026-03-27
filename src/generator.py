"""
Story generator using a Malagasy-instruct LLM.
Uses Lo-Renz-O/Mistral-7B-instruct-Malagasy-bnb-4bit.

Supports CUDA (GPU), MPS (Apple Silicon), and CPU.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Lo-Renz-O/Mistral-7B-instruct-Malagasy-bnb-4bit"

STORY_PROMPT_TEMPLATE = """Mampiasà teny Malagasy fotsiny.
Manoratra tantely fohy (100-150 teny) ho an'ny ankizy momba ny: {theme}
Tsy maintsy mahafinaritra sy mampianatra.
Tsy misy fandotoana na zavatra ratsy.

Tantely:"""


def get_device():
    """Detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model():
    """Load model with appropriate settings for the current device."""
    device = get_device()
    print(f"Loading text model on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if device == "cuda":
        # Use 4-bit quantization on NVIDIA GPU
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        # Apple Silicon — load in float16, no bitsandbytes
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model = model.to("mps")
    else:
        # CPU fallback — float32, will be slow
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Text model loaded on {device}.")
    return tokenizer, model, device


class StoryGenerator:
    def __init__(self):
        self.tokenizer, self.model, self.device = load_model()

    def generate_story(self, theme: str, max_new_tokens: int = 300, temperature: float = 0.8) -> str:
        """Generate a short Malagasy children's story about the given theme."""
        prompt = STORY_PROMPT_TEMPLATE.format(theme=theme)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the generated part (skip the prompt)
        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        return generated.strip()
