from transformers import MarianMTModel, MarianTokenizer
import torch

class Backtranslation:
    # Models
    def __init__(self):
        _en_to_es_model_name = "Helsinki-NLP/opus-mt-en-es"
        _es_to_en_model_name = "Helsinki-NLP/opus-mt-es-en"

        self._en_to_es_tokenizer = MarianTokenizer.from_pretrained(_en_to_es_model_name)
        self._en_to_es_model = MarianMTModel.from_pretrained(_en_to_es_model_name).eval()
        self._es_to_en_tokenizer = MarianTokenizer.from_pretrained(_es_to_en_model_name)
        self._es_to_en_model = MarianMTModel.from_pretrained(_es_to_en_model_name).eval()

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        print('Device: ', self._device)
        self._en_to_es_model.to(self._device)
        self._es_to_en_model.to(self._device)
    

    def translate(self, answer):
        print('Generating Translation...')
        def text_translate(text, tokenizer, model):
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self._device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=512)
            return tokenizer.decode(out[0], skip_special_tokens=True)

        es = text_translate(answer, self._en_to_es_tokenizer, self._en_to_es_model)
        back = text_translate(es, self._es_to_en_tokenizer, self._es_to_en_model)
        return back



