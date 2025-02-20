import os
import json
import csv

# Получаем абсолютный путь к корневой папке проекта
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Поднимаемся на уровень

def save_to_json(data, file_path):
    """Сохраняет данные в JSON-файл внутри проекта."""
    full_path = os.path.join(BASE_DIR, file_path)  # Преобразуем в абсолютный путь
    os.makedirs(os.path.dirname(full_path), exist_ok=True)  # Создаем папку, если её нет

    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def save_to_csv(data, file_path):
    """Сохраняет данные в CSV-файл внутри проекта."""
    if not data:
        return

    full_path = os.path.join(BASE_DIR, file_path)  # Преобразуем в абсолютный путь
    os.makedirs(os.path.dirname(full_path), exist_ok=True)  # Создаем папку, если её нет

    # Определяем названия колонок
    fieldnames = data[0].keys()

    with open(full_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
