import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer

class LLaMAModel:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B",
    ):
        """
        Initialize T5 model for translation

        Args:
            model_name: name of the model on huggingface
        """
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        print(f"Loading model on device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model
        print(f"Loading LLaMA model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """
        Translate text from source language to target language

        Args:
            text: Text to translate
            source_lang: Source language identifier
            target_lang: Target language identifier
            max_length: Maximum length of generated translation
            temperature: Sampling temperature
            num_beams: Number of beams for beam search
        """
        # Create T5-style prompt for translation
        prompt = f"translate {source_lang} to {target_lang}: {text}"

        # Tokenize input
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).input_ids.to(self.device)

        outputs = self.model.generate(input_ids)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class T5Translator:
    def __init__(
        self,
        model_name: str = "google-t5/t5-small",
    ):
        """
        Initialize T5 model for translation

        Args:
            model_name: Name of the T5 model on HuggingFace
            device: Device to load model on ('cuda' or 'cpu')
            load_in_8bit: Whether to load in 8-bit precision to save memory
        """
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        print(f"Loading model on device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model
        print(f"Loading T5 model: {model_name}")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """
        Translate text from source language to target language

        Args:
            text: Text to translate
            source_lang: Source language identifier
            target_lang: Target language identifier
            max_length: Maximum length of generated translation
            temperature: Sampling temperature
            num_beams: Number of beams for beam search
        """
        # Create T5-style prompt for translation
        prompt = f"translate {source_lang} to {target_lang}: {text}"

        # Tokenize input
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).input_ids.to(self.device)

        outputs = self.model.generate(input_ids)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    # Example usage
    translator = T5Translator()
    text = "He ain't got no money"
    translation = translator.translate(text=text, source_lang="AAVE", target_lang="SAE")
    print(f"Original: {text}")
    print(f"Translation: {translation}")
