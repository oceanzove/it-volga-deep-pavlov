import pandas as pd

def load_addresses(address_file):
    """Загружает файл с адресами."""
    return pd.read_csv(address_file, sep=';', encoding='utf-8')

def load_comments(comment_file):
    """Загружает файл с комментариями диспетчеров."""
    return pd.read_csv(comment_file, sep=';', encoding='utf-8')
