import pandas as pd
import spacy
from collections import Counter
import logging

# Настройка логирования
logger = logging.getLogger(__name__)

# Загружаем модель spaCy для русского языка
nlp = spacy.load("ru_core_news_sm")

class ReviewAnalyzer:
    def __init__(self):
        self.positive_keywords = set()  # Множество для уникальных положительных ключевых слов (все сайты)
        self.negative_keywords = set()  # Множество для уникальных отрицательных ключевых слов (все сайты)

    def analyze_reviews(self, csv_path):
        """Анализ отзывов из CSV-файла и выделение плюсов, минусов и ключевых слов для одного сайта."""
        try:
            # Читаем CSV без принудительного указания имен столбцов
            df = pd.read_csv(csv_path)

            # Проверяем структуру CSV (для Отзовика — Достоинства/Недостатки/Оценка, для других — Текст отзыва/Оценка)
            if all(col in df.columns for col in ['Достоинства', 'Недостатки', 'Оценка']):
                # Структура для Отзовика
                positives = []
                negatives = []
                site_keywords = []

                for _, row in df.iterrows():
                    rating = int(row['Оценка']) if pd.notna(row['Оценка']) else 3
                    pros_text = str(row['Достоинства']) if pd.notna(row['Достоинства']) else ""
                    cons_text = str(row['Недостатки']) if pd.notna(row['Недостатки']) else ""

                    # Выделяем ключевые слова для плюсов
                    pros_keywords = self.extract_keywords(pros_text)
                    site_keywords.extend(pros_keywords)
                    if pros_text.strip() and pros_text.strip().lower() != "нет":
                        positives.append(pros_text)
                        self.positive_keywords.update(pros_keywords)

                    # Выделяем ключевые слова для минусов
                    cons_keywords = self.extract_keywords(cons_text)
                    site_keywords.extend(cons_keywords)
                    if cons_text.strip() and cons_text.strip().lower() != "нет":
                        negatives.append(cons_text)
                        self.negative_keywords.update(cons_keywords)

                # Определяем основные плюсы и минусы для этого сайта
                main_positives = self.summarize_reviews(positives) if positives else "Нет положительных отзывов"
                main_negatives = self.summarize_reviews(negatives) if negatives else "Нет отрицательных отзывов"
                top_keywords = self.get_top_keywords(site_keywords, 5)  # Топ-5 ключевых слов для этого сайта
            elif all(col in df.columns for col in ['Текст отзыва', 'Оценка']):
                # Структура для iRecommend и Ozon
                positives = []
                negatives = []
                site_keywords = []

                # Обрабатываем каждый отзыв
                for _, row in df.iterrows():
                    review_text = str(row['Текст отзыва'])
                    rating = int(row['Оценка']) if pd.notna(row['Оценка']) else 3  # По умолчанию нейтральная оценка 3

                    # Выделяем ключевые слова с помощью spaCy
                    keywords = self.extract_keywords(review_text)
                    site_keywords.extend(keywords)

                    # Классификация на основе оценки и текста
                    if rating >= 4:  # Положительные отзывы
                        positives.append(review_text)
                        self.positive_keywords.update(keywords)
                    elif rating <= 2:  # Отрицательные отзывы
                        negatives.append(review_text)
                        self.negative_keywords.update(keywords)
                    else:  # Нейтральные отзывы (оценка 3)
                        pass  # Можно игнорировать или добавить в отдельную категорию

                # Определяем основные плюсы и минусы для этого сайта
                main_positives = self.summarize_reviews(positives) if positives else "Нет положительных отзывов"
                main_negatives = self.summarize_reviews(negatives) if negatives else "Нет отрицательных отзывов"
                top_keywords = self.get_top_keywords(site_keywords, 5)  # Топ-5 ключевых слов для этого сайта
            else:
                raise ValueError("CSV-файл должен содержать либо столбцы 'Достоинства', 'Недостатки' и 'Оценка', либо 'Текст отзыва' и 'Оценка'")

            return {
                "Плюсы": main_positives,
                "Минусы": main_negatives,
                "Ключевые слова (положительные)": ", ".join(sorted(self.positive_keywords)[:5]),
                "Ключевые слова (отрицательные)": ", ".join(sorted(self.negative_keywords)[:5]),
                "Общие ключевые слова": top_keywords
            }
        except Exception as e:
            logger.error(f"Ошибка при анализе файла {csv_path}: {str(e)}")
            return {
                "Плюсы": "Ошибка",
                "Минусы": "Ошибка",
                "Ключевые слова (положительные)": "Ошибка",
                "Ключевые слова (отрицательные)": "Ошибка",
                "Общие ключевые слова": "Ошибка"
            }

    def aggregate_reviews(self, csv_paths):
        """Агрегированный анализ отзывов из всех CSV-файлов."""
        try:
            all_positives = []
            all_negatives = []
            all_keywords = []

            for csv_path in csv_paths:
                try:
                    df = pd.read_csv(csv_path)
                    if all(col in df.columns for col in ['Достоинства', 'Недостатки', 'Оценка']):
                        # Структура для Отзовика
                        for _, row in df.iterrows():
                            rating = int(row['Оценка']) if pd.notna(row['Оценка']) else 3
                            pros_text = str(row['Достоинства']) if pd.notna(row['Достоинства']) else ""
                            cons_text = str(row['Недостатки']) if pd.notna(row['Недостатки']) else ""

                            pros_keywords = self.extract_keywords(pros_text)
                            all_keywords.extend(pros_keywords)
                            if pros_text.strip() and pros_text.strip().lower() != "нет":
                                all_positives.append(pros_text)
                                self.positive_keywords.update(pros_keywords)

                            cons_keywords = self.extract_keywords(cons_text)
                            all_keywords.extend(cons_keywords)
                            if cons_text.strip() and cons_text.strip().lower() != "нет":
                                all_negatives.append(cons_text)
                                self.negative_keywords.update(cons_keywords)
                    elif all(col in df.columns for col in ['Текст отзыва', 'Оценка']):
                        # Структура для iRecommend и Ozon
                        for _, row in df.iterrows():
                            review_text = str(row['Текст отзыва'])
                            rating = int(row['Оценка']) if pd.notna(row['Оценка']) else 3

                            keywords = self.extract_keywords(review_text)
                            all_keywords.extend(keywords)

                            if rating >= 4:
                                all_positives.append(review_text)
                                self.positive_keywords.update(keywords)
                            elif rating <= 2:
                                all_negatives.append(review_text)
                                self.negative_keywords.update(keywords)
                    else:
                        logger.warning(f"CSV-файл {csv_path} не содержит нужных столбцов")
                        continue

                except Exception as e:
                    logger.warning(f"Ошибка при обработке {csv_path}: {str(e)}")
                    continue

            # Определяем общие плюсы, минусы и ключевые слова
            main_positives = self.summarize_reviews(all_positives) if all_positives else "Нет положительных отзывов"
            main_negatives = self.summarize_reviews(all_negatives) if all_negatives else "Нет отрицательных отзывов"
            top_keywords = self.get_top_keywords(all_keywords, 5)

            return {
                "Плюсы (все сайты)": main_positives,
                "Минусы (все сайты)": main_negatives,
                "Ключевые слова (положительные, все сайты)": ", ".join(sorted(self.positive_keywords)[:5]),
                "Ключевые слова (отрицательные, все сайты)": ", ".join(sorted(self.negative_keywords)[:5]),
                "Общие ключевые слова (все сайты)": top_keywords
            }
        except Exception as e:
            logger.error(f"Ошибка при агрегированном анализе: {str(e)}")
            return {
                "Плюсы (все сайты)": "Ошибка",
                "Минусы (все сайты)": "Ошибка",
                "Ключевые слова (положительные, все сайты)": "Ошибка",
                "Ключевые слова (отрицательные, все сайты)": "Ошибка",
                "Общие ключевые слова (все сайты)": "Ошибка"
            }

    def extract_keywords(self, text):
        """Извлекает ключевые слова из текста с помощью spaCy."""
        try:
            doc = nlp(text.lower())
            keywords = []
            for token in doc:
                if not token.is_stop and not token.is_punct and token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]:
                    keywords.append(token.lemma_)
            return keywords
        except Exception as e:
            logger.warning(f"Ошибка при извлечении ключевых слов из текста: {str(e)}")
            return []

    def summarize_reviews(self, reviews):
        """Составляет краткое описание плюсов или минусов на основе отзывов."""
        try:
            if not reviews:
                return "Нет данных"
            all_keywords = []
            for review in reviews:
                all_keywords.extend(self.extract_keywords(review))
            keyword_counts = Counter(all_keywords)
            top_keywords = [word for word, count in keyword_counts.most_common(5)]
            return f"Основные аспекты: {', '.join(top_keywords) if top_keywords else 'Нет ключевых слов'}"
        except Exception as e:
            logger.warning(f"Ошибка при суммировании отзывов: {str(e)}")
            return "Ошибка"

    def get_top_keywords(self, keywords, n=5):
        """Возвращает топ-n ключевых слов по частоте."""
        try:
            if not keywords:
                return "Нет ключевых слов"
            keyword_counts = Counter(keywords)
            top_keywords = [word for word, count in keyword_counts.most_common(n)]
            return ", ".join(top_keywords)
        except Exception as e:
            logger.warning(f"Ошибка при получении топ-ключевых слов: {str(e)}")
            return "Ошибка"
