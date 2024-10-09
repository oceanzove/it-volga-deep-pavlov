import sys
import os

# Добавляем путь к src в список путей поиска модулей
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from data_preprocessing import load_addresses, load_comments
from ner_model import NERModel
from matching import match_addresses

# Загрузка данных
address_file = 'data/volgait2024-semifinal-addresses.csv'
comment_file = 'data/volgait2024-semifinal-task.csv'
addresses = load_addresses(address_file)
comments = load_comments(comment_file)

# Инициализация модели NER
ner_model = NERModel()

# Результаты
results = []

# Обработка каждого комментария
for _, row in comments.iterrows():
    shutdown_id = row['shutdown_id']
    comment = row['comment']

    # Извлечение адресов с помощью модели NER
    extracted_addresses = ner_model.extract_addresses(comment)

    # Сопоставление извлечённых адресов с базой
    house_uuids = match_addresses(extracted_addresses[0], addresses)

    # Добавление результата
    results.append({
        'shutdown_id': shutdown_id,
        'house_uuids': ','.join(house_uuids)
    })

# Сохранение результата в CSV
result_df = pd.DataFrame(results)
result_df.to_csv('results/volgait2024-semifinal-result.csv', sep=';', index=False)
