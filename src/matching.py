import re
from fuzzywuzzy import fuzz

def preprocess_address(address):
    # Удаление лишних пробелов и приведение к нижнему регистру
    address = re.sub(r'\s+', ' ', address.lower().strip())
    # Удаление знаков препинания
    address = re.sub(r'[^\w\s]', '', address)
    return address

def match_addresses(extracted_addresses, address_database):
    matched_uuids = set()
    
    # Определяем доступные колонки в address_database
    available_columns = address_database.columns

    for extracted_address in extracted_addresses:
        extracted_address = preprocess_address(extracted_address)
        best_match = None
        best_score = 0
        
        for _, row in address_database.iterrows():
            # Формируем адрес из доступных колонок
            db_address_parts = []
            if 'city' in available_columns:
                db_address_parts.append(str(row['city']))
            if 'street' in available_columns:
                db_address_parts.append(str(row['street']))
            if 'house' in available_columns:
                db_address_parts.append(str(row['house']))
            
            db_address = " ".join(db_address_parts)
            db_address = preprocess_address(db_address)
            
            score = fuzz.ratio(extracted_address, db_address)
            
            if score > best_score:
                best_score = score
                best_match = row['house_uuid'] if 'house_uuid' in available_columns else None
        
        if best_score > 80 and best_match:  # Порог схожести и проверка наличия UUID
            matched_uuids.add(best_match)
    
    return list(matched_uuids)
