from transformers import MarianMTModel, MarianTokenizer
import torch

class Translator:
    # Models
    def __init__(self):
        _model_name_list = [''
        'Helsinki-NLP/opus-mt-en-es',
        'Helsinki-NLP/opus-mt-en-fr',
        'Helsinki-NLP/opus-mt-es-fr',
        'Helsinki-NLP/opus-mt-es-en',
        'Helsinki-NLP/opus-mt-fr-en'
        ]

        self.translators = {}

        for n in _model_name_list:
            model_name = n[-5:]
            self.translators[model_name] = {
                'tokenizer': MarianTokenizer.from_pretrained(n),
                'model': MarianMTModel.from_pretrained(n).eval()
            }

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        print('Device: ', self._device)
        for key in self.translators:
            self.translators[key]['model'].to(self._device)

    def translate(self, answer, lang_from, lang_to):
        print('Generating Translation...')
        translator = self.translators[lang_from+'-'+lang_to]

        def text_translate(text, tokenizer, model):
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self._device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=512)
            return tokenizer.decode(out[0], skip_special_tokens=True)
        
        return text_translate(answer, translator['tokenizer'], translator['model'])