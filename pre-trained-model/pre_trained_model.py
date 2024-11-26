from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer

import torch
from typing import Optional

class T5Translator:
    def __init__(
        self,
        model_name: str = "google-t5/t5-small",
        device: Optional[str] = None,
        load_in_8bit: bool = False
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
            # else "mps" if torch.backends.mps.is_available() 
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
        max_length: int = 128,
        temperature: float = 0.7,
        num_beams: int = 4,
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

        print(input_ids)

        outputs = self.model.generate(input_ids)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def batch_translate(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
        batch_size: int = 8,
        **kwargs
    ) -> list[str]:
        """
        Translate a batch of texts
        
        Args:
            texts: List of texts to translate
            source_lang: Source language identifier
            target_lang: Target language identifier
            batch_size: Size of batches for processing
            **kwargs: Additional arguments passed to translate()
        """
        translations = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_translations = [
                self.translate(text, source_lang, target_lang, **kwargs)
                for text in batch
            ]
            translations.extend(batch_translations)
            
        return translations
    
if __name__ == "__main__":
    # Example usage
    translator = T5Translator()
    text = "He ain't got no money"
    translation = translator.translate(
        text=text,
        source_lang="AAVE",
        target_lang="SAE"
    )
    print(f"Original: {text}")
    print(f"Translation: {translation}")
