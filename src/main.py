import sys
import os
import logging
import pandas as pd
from ner_model import NERModel
from matching import match_addresses
import argparse

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(address_file, comment_file):
    try:
        logging.info("Загрузка адресов...")
        addresses = pd.read_csv(address_file, sep=';')
        logging.info(f"Загружено {len(addresses)} адресов")
        logging.info(f"Колонки в файле адресов: {', '.join(addresses.columns)}")
        logging.info("Загрузка комментариев...")
        comments = pd.read_csv(comment_file, sep=';')
        logging.info(f"Загружено {len(comments)} комментариев")
        logging.info(f"Колонки в файле комментариев: {', '.join(comments.columns)}")
        return addresses, comments
    except Exception as e:
        logging.error(f"Ошибка при загрузке данных: {e}", exc_info=True)
        raise

def main(num_addresses):
    # Загрузка данных
    address_file = 'data/volgait2024-semifinal-addresses.csv'
    comment_file = 'data/volgait2024-semifinal-task.csv'
    
    try:
        addresses, comments = load_data(address_file, comment_file)
    except Exception as e:
        logging.error("Не удалось загрузить данные. Завершение программы.")
        return

    # Инициализация модели NER
    try:
        logging.info("Инициализация модели NER...")
        ner_model = NERModel()
    except Exception as e:
        logging.error(f"Ошибка при инициализации модели NER: {e}", exc_info=True)
        return

    # Результаты
    results = []

    # Обработка выбранного количества комментариев
    total_comments = min(num_addresses, len(comments)) if num_addresses else len(comments)
    for index, row in comments.iloc[:total_comments].iterrows():
        shutdown_id = row['shutdown_id']
        comment = row['comment']

        logging.info(f"Обработка комментария {index+1}/{total_comments}")

        try:
            # Извлечение адресов с помощью модели NER
            extracted_addresses = ner_model.extract_addresses(comment)
            logging.info(f"Извлечено адресов: {len(extracted_addresses)}")

            # Сопоставление извлечённых адресов с базой
            house_uuids = match_addresses(extracted_addresses, addresses)
            logging.info(f"Найдено соответствий: {len(house_uuids)}")

            # Добавление результата
            results.append({
                'shutdown_id': shutdown_id,
                'house_uuids': ','.join(house_uuids)
            })
        except Exception as e:
            logging.error(f"Ошибка при обработке комментария {shutdown_id}: {e}", exc_info=True)

    # Сохранение результата в CSV
    try:
        logging.info("Сохранение результатов...")
        result_df = pd.DataFrame(results)
        os.makedirs('results', exist_ok=True)
        result_df.to_csv('results/volgait2024-semifinal-result.csv', sep=';', index=False)
        logging.info(f"Результаты успешно сохранены. Обработано {len(results)} комментариев")
    except Exception as e:
        logging.error(f"Ошибка при сохранении результатов: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NER для извлечения адресов")
    parser.add_argument("--num_addresses", type=int, help="Количество адресов для обработки")
    args = parser.parse_args()
    
    main(args.num_addresses)
