from deeppavlov import build_model, configs

class NERModel:
    def __init__(self):
        """Инициализация модели NER для русского языка."""
        self.model = build_model(configs.ner.ner_rus, download=True)

    def extract_addresses(self, text):
        """Извлекает потенциальные адреса из текста с помощью модели."""
        return self.model([text])
