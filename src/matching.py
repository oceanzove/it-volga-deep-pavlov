def match_addresses(extracted_addresses, address_df):
    """
    Сопоставляет извлечённые адреса с доступной базой.

    :param extracted_addresses: Адреса, извлечённые из комментария
    :param address_df: DataFrame с полным списком адресов
    :return: Список UUID домов, которые соответствуют извлечённым адресам
    """
    matched_uuids = []
    for address in extracted_addresses:
        matches = address_df[address_df['house_full_address'].str.contains(address)]
        matched_uuids.extend(matches['house_uuid'].tolist())

    return matched_uuids
