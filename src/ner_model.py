import logging
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
import torch
import numpy as np

class NERModel:
    def __init__(self):
        try:
            logging.info("Начало инициализации модели NER")
            self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
            self.model = AutoModelForTokenClassification.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
            logging.info("Модель NER успешно инициализирована")
        except Exception as e:
            logging.error(f"Ошибка при инициализации модели NER: {e}", exc_info=True)
            raise

    def extract_addresses(self, text):
        try:
            logging.info("Начало извлечения адресов")
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            predictions = torch.argmax(outputs.logits, dim=2)
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            addresses = []
            current_address = ""
            for token, prediction in zip(tokens, predictions[0]):
                if prediction == 1:  # Предполагаем, что метка 1 соответствует адресу
                    if token.startswith("##"):
                        current_address += token[2:]
                    else:
                        current_address += " " + token
                elif current_address:
                    addresses.append(current_address.strip())
                    current_address = ""
            
            if current_address:
                addresses.append(current_address.strip())
            
            logging.info(f"Извлечено адресов: {len(addresses)}")
            return addresses
        except Exception as e:
            logging.error(f"Ошибка при извлечении адресов: {e}", exc_info=True)
            raise

    def fine_tune(self, train_dataset, eval_dataset, output_dir):
        try:
            logging.info("Начало дообучения модели NER")
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=3,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=64,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir='./logs',
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset
            )

            trainer.train()
            trainer.save_model()
            logging.info("Модель NER успешно дообучена и сохранена")
        except Exception as e:
            logging.error(f"Ошибка при дообучении модели NER: {e}", exc_info=True)
            raise

    def save_model(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logging.info(f"Модель NER сохранена в {path}")

    def load_model(self, path):
        self.model = AutoModelForTokenClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        logging.info(f"Модель NER загружена из {path}")
