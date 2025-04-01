import pandas as pd
import spacy
from collections import Counter
import logging
import sys
from typing import List, Dict, Set, Union, Tuple
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pymorphy3
from multiprocessing import Pool
import numpy as np
from datetime import datetime
import json
from collections import OrderedDict

# Попытка импорта YandexSpeller
try:
    from pyaspeller import YandexSpeller
    speller = YandexSpeller()
except ImportError:
    speller = None
    logging.warning("Модуль pyaspeller недоступен. Предобработка текста будет отключена.")

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,  # Изменим уровень на DEBUG, чтобы видеть больше информации
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Вывод в консоль
        logging.FileHandler('review_analyzer.log', encoding='utf-8')  # Вывод в файл
    ]
)
logger = logging.getLogger(__name__)

# Инициализация spaCy и pymorphy3
nlp = spacy.load("ru_core_news_sm", disable=["ner"])
morph = pymorphy3.MorphAnalyzer()

# Инициализация модели для анализа тональности
model_name = "seara/rubert-tiny2-russian-sentiment"
sentiment_analyzer = None
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)
    logger.info(f"Модель {model_name} успешно загружена.")
except Exception as e:
    logger.warning(f"Ошибка загрузки модели {model_name}: {e}. Используется словарный анализ.")
    sentiment_analyzer = None

# Расширенный словарь тональности
SENTIMENT_DICT = {
    "хороший": ("положительное", 0.7), "отличный": ("положительное", 0.9), "прекрасный": ("положительное", 0.8),
    "удобный": ("положительное", 0.6), "качественный": ("положительное", 0.7), "быстрый": ("положительное", 0.6),
    "надежный": ("положительное", 0.7), "вкусный": ("положительное", 0.7), "красивый": ("положительное", 0.6),
    "приятный": ("положительное", 0.6), "замечательный": ("положительное", 0.8), "свежий": ("положительное", 0.6),
    "вежливый": ("положительное", 0.7), "плохой": ("отрицательное", -0.7), "ужасный": ("отрицательное", -0.9),
    "дефектный": ("отрицательное", -0.8), "неудобный": ("отрицательное", -0.6), "медленный": ("отрицательное", -0.6),
    "ненадежный": ("отрицательное", -0.7), "невкусный": ("отрицательное", -0.7), "сломанный": ("отрицательное", -0.8),
    "грязный": ("отрицательное", -0.7), "дорогой": ("отрицательное", -0.7), "безвкусный": ("отрицательное", -0.7),
    "прогорклый": ("отрицательное", -0.8), "протухший": ("отрицательное", -0.9), "трындец": ("отрицательное", -0.9),
    "худший": ("отрицательное", -0.8), "хамоватый": ("отрицательное", -0.7), "неквалифицированный": ("отрицательное", -0.7),
    "высокий": ("нейтральное", 0.0), "просто": ("нейтральное", 0.0), "развивается": ("нейтральное", 0.0),
    "хранится": ("нейтральное", 0.0), "хранении": ("нейтральное", 0.0), "обслуживание": ("нейтральное", 0.0),
    "персонал": ("нейтральное", 0.0), "топливо": ("нейтральное", 0.0), "нет": ("отрицательное", -0.5),
    "нету": ("отрицательное", -0.5), "очень": ("усилитель", 1.5), "слегка": ("ослабитель", 0.5),
    "совершенно": ("усилитель", 1.3), "абсолютно": ("усилитель", 1.4), "немного": ("ослабитель", 0.7),
    "дрянь": ("отрицательное", -0.8), "обман": ("отрицательное", -0.7), "воруют": ("отрицательное", -0.9),
    "хмурый": ("отрицательное", -0.6), "недосмотр": ("отрицательное", -0.6), "отвратительно": ("отрицательное", -0.9),
    "воровство": ("отрицательное", -0.9), "обманывают": ("отрицательное", -0.8), "не рекомендую": ("отрицательное", -0.7),
    "жесть": ("отрицательное", -0.7), "стыдобень": ("отрицательное", -0.7), "недолив": ("отрицательное", -0.8),
    "хамство": ("отрицательное", -0.7), "дороговато": ("отрицательное", -0.8), "мало": ("отрицательное", -0.6),
    "маленький": ("отрицательное", -0.5), "маловато": ("отрицательное", -0.6), "странный": ("отрицательное", -0.6),
    "высокая": ("отрицательное", -0.6), "непонятный": ("отрицательное", -0.6)
}

class ReviewAnalyzer:
    def __init__(self, use_preprocessing: bool = True, positive_threshold: int = 4, negative_threshold: int = 2):
        self.positive_keywords: Set[str] = set()
        self.negative_keywords: Set[str] = set()
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.use_preprocessing = use_preprocessing
        self.invalid_phrases = {
            "безобидный вкус", "изящный делаться", "вкусный блок", "допорывать",
            "прислать развакуум", "возврат делать отказываться", "косарь на ветер",
            "пахнуть быльгет", "звый раз", "два тык", "дотошный аккумулятор", "шень 224"
        }
        self.sentiment_cache = OrderedDict()
        self.max_cache_size = 10000
        self.domain_hints = [
            "рыба", "мясо", "еда", "экран", "камера", "креветка", "минтай", "гребешок", "вкус", "качество", "размер",
            "топливо", "сервис", "обслуживание", "персонал", "баллы", "АЗС", "бензин", "дизель", "карта", "цена",
            "приложение", "состав", "упаковка", "икра"
        ]
        self.review_data = []
        self.stop_words = set(["и", "в", "на", "с", "по", "для", "к", "у", "из", "о", "а", "но", "или"])
        self.emoji_dict = {
            "👍": "положительный_эмодзи", "👌": "положительный_эмодзи", "😊": "положительное_эмоция",
            "🙂": "положительное_эмоция", "😍": "положительное_эмоция", "😢": "отрицательное_эмоция",
            "😡": "отрицательное_эмоция", "😠": "отрицательное_эмоция", "😋": "положительное_эмоция",
            "！": "восклицание", "‼": "двойное_восклицание", "？": "вопросительный_знак",
            "🌻": "цветок_эмодзи", "🚘": "машина_эмодзи", "👎": "отрицательный_эмодзи",
            "😞": "отрицательное_эмоция", "🤔": "нейтральное_эмоция", "💡": "идея_эмодзи"
        }
        self.load_cache()

    def save_cache(self):
        with open("sentiment_cache.json", "w", encoding="utf-8") as f:
            json.dump(dict(self.sentiment_cache), f, ensure_ascii=False)

    def load_cache(self):
        try:
            with open("sentiment_cache.json", "r", encoding="utf-8") as f:
                self.sentiment_cache = OrderedDict(json.load(f))
        except FileNotFoundError:
            self.sentiment_cache = OrderedDict()

    def log_safe(self, text: str) -> str:
        return self.replace_emojis(text)

    def replace_emojis(self, text: str) -> str:
        for emoji, label in self.emoji_dict.items():
            text = text.replace(emoji, f" {label} ")
        return text.strip()

    def preprocess_text(self, text: str) -> str:
        if not self.use_preprocessing or not text.strip():
            return text
        if speller is None:
            logger.warning("pyaspeller недоступен, предобработка текста отключена.")
            return text
        try:
            text = self.replace_emojis(text)
            corrected = speller.spelled(text)
            typo_corrections = {"при хари": "при хранении", "хари": "хранении"}
            for typo, correction in typo_corrections.items():
                corrected = corrected.replace(typo, correction)
            doc = nlp(corrected)
            normalized_tokens = [
                morph.parse(token.text)[0].normal_form
                for token in doc
                if not token.is_punct and not token.is_stop and token.text.lower() not in self.stop_words
            ]
            return " ".join(normalized_tokens).strip()
        except Exception as e:
            logger.warning(f"Ошибка при предобработке текста: {str(e)}")
            return text

    def split_sentences(self, text: str) -> List[str]:
        text = re.sub(r'[!?.]+\)+|\)+|[:;]-?\)+', '.', text)
        text = re.sub(r'(\.+|\!+|\?+)', r'. ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def split_mixed_sentence(self, sentence: str) -> List[str]:
        parts = re.split(r'\s+(но|а|или|хотя|зато)\s+', sentence.lower())
        return [part.strip() for part in parts if part.strip()] if len(parts) > 1 else [sentence]

    def check_modifiers_with_dependencies(self, sentence: str) -> float:
        doc = nlp(sentence)
        sentiment_modifier = 1.0
        negation_count = 0
        intensifier = 1.0
        for token in doc:
            if token.lemma_ in ["не", "нет", "ни", "едва"] and token.dep_ in ["neg"]:
                negation_count += 1
            elif token.lemma_ in ["очень", "крайне", "сильно", "абсолютно", "совершенно"] and token.dep_ in ["advmod"]:
                intensifier = 1.5
            elif token.lemma_ in ["слегка", "немного", "чуть", "еле"] and token.dep_ in ["advmod"]:
                intensifier = 0.5
            elif token.lemma_ in ["вряд ли", "едва ли"] and token.dep_ in ["advmod"]:
                negation_count += 1
        sentiment_modifier = -intensifier if negation_count % 2 == 1 else intensifier
        if "не очень" in sentence.lower():
            sentiment_modifier *= 0.3
        logger.debug(f"Модификатор для '{self.log_safe(sentence)}': {sentiment_modifier}, negation_count={negation_count}")
        return sentiment_modifier

    def analyze_sentiment_transformers(self, texts: List[str]) -> List[Tuple[str, float]]:
        if not texts:
            return [("нейтральное", 0.0)]
        cached_results = [self.sentiment_cache.get(text, None) for text in texts]
        to_process = [text for i, text in enumerate(texts) if cached_results[i] is None]
        results = []
        if to_process and sentiment_analyzer:
            try:
                batch_results = sentiment_analyzer(to_process, batch_size=16)
                for text, result in zip(to_process, batch_results):
                    label = result['label'].lower()
                    score = result['score']
                    logger.info(f"Модель вернула для текста '{self.log_safe(text)}': label={label}, score={score}")
                    doc = nlp(text)
                    has_negative = any(token.lemma_ in [k for k, (s, _) in SENTIMENT_DICT.items() if s == "отрицательное"] for token in doc)
                    has_positive = any(token.lemma_ in [k for k, (s, _) in SENTIMENT_DICT.items() if s == "положительное"] for token in doc)
                    negative_boost = 0
                    if has_negative:
                        negative_words = [token.lemma_ for token in doc if SENTIMENT_DICT.get(token.lemma_, (None, None))[0] == "отрицательное"]
                        if negative_words:
                            negative_boost = -0.7 * len(negative_words) * min(abs(SENTIMENT_DICT[word][1]) for word in negative_words)
                        else:
                            negative_boost = 0
                    if label == "positive" and score > 0.6:
                        base_score = score
                    elif label == "negative" and score > 0.6:
                        base_score = -score
                    elif label == "neutral":
                        base_score = negative_boost if has_negative and negative_boost < -0.5 else 0.6 if has_positive else 0.0
                    else:
                        base_score = 0.0
                    modifier = self.check_modifiers_with_dependencies(text)
                    adjusted_score = base_score * modifier
                    if "!" in text:
                        adjusted_score *= 1.2
                    elif "..." in text or "?" in text:
                        adjusted_score *= 0.8
                    sentiment = "положительное" if adjusted_score > 0.4 else "отрицательное" if adjusted_score < -0.4 else "нейтральное"
                    if sentiment != "нейтральное" and abs(adjusted_score) > 0.8:
                        for token in doc:
                            if token.pos_ in ["ADJ", "ADV"] and token.lemma_ not in SENTIMENT_DICT:
                                self.update_sentiment_dict(token.lemma_, sentiment, adjusted_score)
                    if len(self.sentiment_cache) >= self.max_cache_size:
                        self.sentiment_cache.popitem(last=False)
                    self.sentiment_cache[text] = (sentiment, adjusted_score)
                    results.append((sentiment, adjusted_score))
            except Exception as e:
                logger.warning(f"Ошибка анализа тональности: {e}. Используется словарный метод.")
                for text in to_process:
                    result = self.fallback_sentiment_analysis(text)
                    if len(self.sentiment_cache) >= self.max_cache_size:
                        self.sentiment_cache.popitem(last=False)
                    self.sentiment_cache[text] = result
                    results.append(result)
        final_results = []
        result_idx = 0
        for i, text in enumerate(texts):
            if cached_results[i] is not None:
                final_results.append(cached_results[i])
            else:
                final_results.append(results[result_idx])
                result_idx += 1
        return final_results

    def fallback_sentiment_analysis(self, text: str) -> Tuple[str, float]:
        doc = nlp(text.lower())
        sentiment_scores = {"положительное": 0, "отрицательное": 0}
        for token in doc:
            sentiment, score = SENTIMENT_DICT.get(token.lemma_, ("нейтральное", 0.0))
            if sentiment in ["положительное", "отрицательное"]:
                sentiment_scores[sentiment] += score
        modifier = self.check_modifiers_with_dependencies(text)
        total_score = (sentiment_scores["положительное"] + sentiment_scores["отрицательное"]) * modifier
        return ("положительное", total_score) if total_score > 0.4 else ("отрицательное", total_score) if total_score < -0.4 else ("нейтральное", 0.0)

    def extract_aspects(self, sentence: str) -> List[Tuple[str, str, float, str]]:
        doc = nlp(sentence)
        aspects = []
        invalid_words = {"оченк", "поробовать", "спасного", "заскучаться", "добовство", "хари"}
        MAX_ASPECT_LENGTH = 3  # Ограничим длину аспекта до 3 слов

        if sentence.lower().strip() in ["нет", "нету"]:
            sentiment, score = self.get_sentiment("нет")
            aspects.append(("нет", sentiment, score, sentence))
            logger.info(f"Извлечён аспект 'нет': тональность={sentiment}, скор={score}")
            return aspects

        sentiment_results = self.analyze_sentiment_transformers([sentence])[0]
        sentiment, score = sentiment_results
        clauses = self.split_mixed_sentence(sentence)

        for clause in clauses:
            doc_clause = nlp(clause)
            for token in doc_clause:
                # Извлекаем аспекты только для существительных, прилагательных и глаголов
                if (token.pos_ in ["NOUN", "ADJ", "VERB"] and 
                    token.lemma_ not in invalid_words and 
                    (token.lemma_ in self.domain_hints or token.lemma_ in [k for k, _ in SENTIMENT_DICT.items()])):
                    modifiers = []
                    negation = False
                    children_count = 0
                    for child in token.children:
                        if (child.dep_ in ["amod", "compound", "advmod", "nmod", "obj"] and 
                            child.lemma_ not in invalid_words and 
                            children_count < MAX_ASPECT_LENGTH - 1):
                            modifiers.append(child.lemma_)
                            children_count += 1
                        if child.lemma_ in ["не", "нет", "ни", "едва"] and child.dep_ in ["neg"]:
                            negation = True
                    if token.dep_ in ["amod", "nmod", "obj"]:
                        parent = token.head
                        if parent.pos_ in ["NOUN", "VERB"] and parent.lemma_ not in invalid_words:
                            modifiers.append(parent.lemma_)
                    
                    # Формируем аспект
                    aspect_phrase = " ".join(sorted([mod for mod in modifiers if mod] + [token.lemma_])).strip()
                    
                    # Фильтруем аспекты
                    if (not aspect_phrase or 
                        aspect_phrase.lower() in self.invalid_phrases or 
                        len(aspect_phrase.split()) > MAX_ASPECT_LENGTH or
                        aspect_phrase.lower() == "рыба"):  # Исключаем аспект "рыба", так как это сам продукт
                        continue
                    
                    aspect_sentiment = sentiment
                    aspect_score = score
                    if negation:
                        aspect_sentiment = "отрицательное" if sentiment == "положительное" else "положительное"
                        aspect_score = -score
                    
                    aspects.append((aspect_phrase, aspect_sentiment, aspect_score, sentence))
                    logger.debug(f"Извлечён аспект '{aspect_phrase}' с тональностью '{aspect_sentiment}' из предложения: {sentence}")
        
        return aspects

    def analyze_review_sentences(self, review_text: str) -> List[Tuple[str, str, float, List[Tuple[str, str, float, str]]]]:
        review_text = self.preprocess_text(review_text)
        sentences = self.split_sentences(review_text)
        result = []
        sentiments = self.analyze_sentiment_transformers(sentences)
        for sentence, (sentiment, score) in zip(sentences, sentiments):
            try:
                aspects = self.extract_aspects(sentence)
                result.append((sentence, sentiment, score, aspects))
                logger.info(f"Анализ предложения '{self.log_safe(sentence)}': тональность={sentiment}, скор={score}, аспекты={aspects}")
            except Exception as e:
                logger.error(f"Ошибка при анализе предложения '{self.log_safe(sentence)}': {str(e)}")
                result.append((sentence, "нейтральное", 0.0, []))
        return result

    def validate_csv(self, df: pd.DataFrame) -> bool:
        required_columns_set1 = {"Достоинства", "Недостатки", "Оценка"}
        required_columns_set2 = {"Текст отзыва", "Оценка"}
        columns = set(df.columns)
        if not (required_columns_set1.issubset(columns) or required_columns_set2.issubset(columns)):
            logger.error(f"CSV-файл не содержит необходимых столбцов. Ожидаются: {required_columns_set1} или {required_columns_set2}, найдены: {columns}")
            raise ValueError("CSV-файл должен содержать либо столбцы 'Достоинства', 'Недостатки' и 'Оценка', либо 'Текст отзыва' и 'Оценка'")
        
        if "Оценка" in df.columns:
            # Преобразуем некорректные оценки (0) в 1
            invalid_zero_ratings = df["Оценка"].apply(lambda x: isinstance(x, (int, float)) and x == 0)
            if invalid_zero_ratings.any():
                logger.warning(f"Обнаружены оценки 0 в строках: {df[invalid_zero_ratings].index.tolist()}. Преобразуем их в 1.")
                df.loc[invalid_zero_ratings, "Оценка"] = 1
            
            # Проверяем, что все оценки находятся в диапазоне от 1 до 5
            invalid_ratings = df["Оценка"].apply(lambda x: not (isinstance(x, (int, float)) and 1 <= x <= 5))
            if invalid_ratings.any():
                logger.error(f"Некорректные значения в столбце 'Оценка' после преобразования: {df[invalid_ratings]['Оценка'].tolist()}")
                raise ValueError("Все значения в столбце 'Оценка' должны быть числами от 1 до 5")
        return True

    def normalize_aspect(self, aspect: str) -> str:
        words = sorted(set(aspect.lower().split()))
        # Приводим "вкусный" к "вкус" и "креветка размер" к "размер"
        normalized_words = []
        for word in words:
            if word == "вкусный":
                normalized_words.append("вкус")
            elif word == "креветка" and "размер" in words:
                continue  # Пропускаем "креветка", если есть "размер"
            #elif word == product_name.lower():  # Пропускаем название продукта
                continue
            else:
                normalized_words.append(word)
        
        # Если аспект уже существует, используем его
        aspect_str = " ".join(normalized_words)
        for existing_aspect in self.positive_keywords | self.negative_keywords:
            existing_words = set(existing_aspect.lower().split())
            if existing_words.issubset(set(normalized_words)) and len(existing_words) <= len(normalized_words):
                return existing_aspect
        return aspect_str

    def analyze_reviews(self, csv_path: str) -> Dict[str, Union[str, Counter]]:
        logger.debug(f"Начало анализа файла: {csv_path}")
        try:
            # Попробуем разные кодировки для чтения файла
            encodings = ['utf-8', 'cp1251', 'latin1']
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_path, encoding=encoding)
                    logger.debug(f"Файл {csv_path} успешно прочитан с кодировкой {encoding}")
                    break
                except UnicodeDecodeError:
                    logger.debug(f"Не удалось прочитать файл {csv_path} с кодировкой {encoding}, пробуем следующую")
                    continue
            if df is None:
                logger.error(f"Не удалось прочитать файл {csv_path} с доступными кодировками: {encodings}")
                return {
                    "Плюсы": "Ошибка кодировки",
                    "Минусы": "Ошибка кодировки",
                    "Ключевые слова (положительные)": "Ошибка кодировки",
                    "Ключевые слова (отрицательные)": "Ошибка кодировки",
                    "Общие ключевые слова": "Ошибка кодировки",
                    "Положительные отзывы": "Ошибка кодировки",
                    "Отрицательные отзывы": "Ошибка кодировки",
                    "Нейтральные отзывы": "Ошибка кодировки",
                    "positive_aspects": Counter(),
                    "negative_aspects": Counter()
                }

            # Проверка на пустой файл
            if df.empty:
                logger.warning(f"CSV-файл {csv_path} пустой")
                return {
                    "Плюсы": "Файл пуст",
                    "Минусы": "Файл пуст",
                    "Ключевые слова (положительные)": "",
                    "Ключевые слова (отрицательные)": "",
                    "Общие ключевые слова": "",
                    "Положительные отзывы": "0",
                    "Отрицательные отзывы": "0",
                    "Нейтральные отзывы": "0",
                    "positive_aspects": Counter(),
                    "negative_aspects": Counter()
                }

            # Валидация структуры CSV
            self.validate_csv(df)
            logger.debug(f"Структура CSV-файла {csv_path} валидна")

            # Инициализация счётчиков
            positive_aspects = Counter()
            negative_aspects = Counter()
            positive_count = negative_count = neutral_count = 0
            processed_texts_set = set()

            # Сохранение данных для дальнейшего анализа
            self.review_data = []
            for _, row in df.iterrows():
                pros = row.get("Достоинства", row.get("Текст отзыва", ""))
                cons = row.get("Недостатки", "")
                # Обрабатываем возможные NaN в столбце "Оценка"
                rating = row["Оценка"]
                if pd.isna(rating):
                    logger.warning(f"Обнаружено NaN значение в столбце 'Оценка' в строке {row.name}. Устанавливаем рейтинг 3.")
                    rating = 3.0
                else:
                    rating = float(rating)
                date = row.get("Дата", "")
                username = row.get("Пользователь", "Аноним")
                self.review_data.append({
                    "original_pros": str(pros),
                    "original_cons": str(cons),
                    "rating": rating,
                    "date": date,
                    "username": username
                })

            # Анализ отзывов
            for _, row in df.iterrows():
                pros_text = str(row.get("Достоинства", row.get("Текст отзыва", "")))
                cons_text = str(row.get("Недостатки", ""))
                rating = float(row["Оценка"])  # После валидации здесь уже не должно быть NaN
                logger.debug(f"Обработка отзыва: Достоинства='{pros_text}', Недостатки='{cons_text}', Оценка={rating}")

                for text in [pros_text, cons_text]:
                    if text.strip():
                        processed_texts_set.add(text)
                        sentences = self.analyze_review_sentences(text)
                        for _, sentiment, _, aspects in sentences:
                            if aspects:
                                for aspect_phrase, aspect_sentiment, _, _ in aspects:
                                    # Нормализуем аспект перед добавлением
                                    normalized_aspect = self.normalize_aspect(aspect_phrase)
                                    # Проверяем тональность аспекта
                                    aspect_text_sentiment, _ = self.analyze_sentiment_transformers([normalized_aspect])[0]
                                    if aspect_text_sentiment == "отрицательное":
                                        negative_aspects[normalized_aspect] += 1
                                        self.negative_keywords.add(normalized_aspect)
                                    elif aspect_sentiment == "положительное":
                                        positive_aspects[normalized_aspect] += 1
                                        self.positive_keywords.add(normalized_aspect)
                                    elif aspect_sentiment == "отрицательное":
                                        negative_aspects[normalized_aspect] += 1
                                        self.negative_keywords.add(normalized_aspect)

                if rating >= self.positive_threshold:
                    positive_count += 1
                elif rating <= self.negative_threshold:
                    negative_count += 1
                else:
                    neutral_count += 1

            # Фильтрация неинформативных аспектов
            for aspect in list(positive_aspects.keys()):
                if aspect.lower() in ["нет недостатков", "нет"]:
                    del positive_aspects[aspect]
                    self.positive_keywords.discard(aspect)
            for aspect in list(negative_aspects.keys()):
                if aspect.lower() in ["нет", "нету"]:
                    sentiment, _ = self.analyze_sentiment_transformers([aspect])[0]
                    if sentiment == "положительное":
                        positive_aspects[aspect] += negative_aspects[aspect]
                        self.positive_keywords.add(aspect)
                        del negative_aspects[aspect]
                        self.negative_keywords.discard(aspect)

            # Удаление дублирующихся аспектов
            common_aspects = set(positive_aspects.keys()).intersection(set(negative_aspects.keys()))
            for aspect in common_aspects:
                sentiment, _ = self.analyze_sentiment_transformers([aspect])[0]
                if sentiment == "положительное":
                    del negative_aspects[aspect]
                    self.negative_keywords.discard(aspect)
                elif sentiment == "отрицательное":
                    del positive_aspects[aspect]
                    self.positive_keywords.discard(aspect)
                else:
                    pos_count = positive_aspects[aspect]
                    neg_count = negative_aspects[aspect]
                    if pos_count > neg_count:
                        del negative_aspects[aspect]
                        self.negative_keywords.discard(aspect)
                    else:
                        del positive_aspects[aspect]
                        self.positive_keywords.discard(aspect)

            main_positives = "\n".join([f"{aspect} ({count})" for aspect, count in positive_aspects.most_common(5)]) if positive_aspects else "Нет положительных отзывов"
            main_negatives = "\n".join([f"{aspect} ({count})" for aspect, count in negative_aspects.most_common(5)]) if negative_aspects else "Нет отрицательных отзывов"

            # Сохранение кэша после анализа
            self.save_cache()

            result = {
                "Плюсы": main_positives,
                "Минусы": main_negatives,
                "Ключевые слова (положительные)": ", ".join(sorted(self.positive_keywords)[:10]),
                "Ключевые слова (отрицательные)": ", ".join(sorted(self.negative_keywords)[:10]),
                "Общие ключевые слова": ", ".join(sorted(set(self.positive_keywords).union(self.negative_keywords))[:15]),
                "Положительные отзывы": str(positive_count),
                "Отрицательные отзывы": str(negative_count),
                "Нейтральные отзывы": str(neutral_count),
                "positive_aspects": positive_aspects,
                "negative_aspects": negative_aspects
            }
            logger.debug(f"Результат анализа: {result}")
            return result

        except FileNotFoundError as e:
            logger.error(f"Файл {csv_path} не найден: {str(e)}")
            return {
                "Плюсы": "Файл не найден",
                "Минусы": "Файл не найден",
                "Ключевые слова (положительные)": "Файл не найден",
                "Ключевые слова (отрицательные)": "Файл не найден",
                "Общие ключевые слова": "Файл не найден",
                "Положительные отзывы": "Файл не найден",
                "Отрицательные отзывы": "Файл не найден",
                "Нейтральные отзывы": "Файл не найден",
                "positive_aspects": Counter(),
                "negative_aspects": Counter()
            }
        except UnicodeDecodeError as e:
            logger.error(f"Ошибка кодировки при чтении файла {csv_path}: {str(e)}")
            return {
                "Плюсы": "Ошибка кодировки",
                "Минусы": "Ошибка кодировки",
                "Ключевые слова (положительные)": "Ошибка кодировки",
                "Ключевые слова (отрицательные)": "Ошибка кодировки",
                "Общие ключевые слова": "Ошибка кодировки",
                "Положительные отзывы": "Ошибка кодировки",
                "Отрицательные отзывы": "Ошибка кодировки",
                "Нейтральные отзывы": "Ошибка кодировки",
                "positive_aspects": Counter(),
                "negative_aspects": Counter()
            }
        except ValueError as e:
            logger.error(f"Ошибка валидации CSV-файла {csv_path}: {str(e)}")
            return {
                "Плюсы": f"Ошибка валидации: {str(e)}",
                "Минусы": f"Ошибка валидации: {str(e)}",
                "Ключевые слова (положительные)": f"Ошибка валидации: {str(e)}",
                "Ключевые слова (отрицательные)": f"Ошибка валидации: {str(e)}",
                "Общие ключевые слова": f"Ошибка валидации: {str(e)}",
                "Положительные отзывы": f"Ошибка валидации: {str(e)}",
                "Отрицательные отзывы": f"Ошибка валидации: {str(e)}",
                "Нейтральные отзывы": f"Ошибка валидации: {str(e)}",
                "positive_aspects": Counter(),
                "negative_aspects": Counter()
            }
        except Exception as e:
            logger.error(f"Неизвестная ошибка при анализе файла {csv_path}: {str(e)}")
            return {
                "Плюсы": "Неизвестная ошибка",
                "Минусы": "Неизвестная ошибка",
                "Ключевые слова (положительные)": "Неизвестная ошибка",
                "Ключевые слова (отрицательные)": "Неизвестная ошибка",
                "Общие ключевые слова": "Неизвестная ошибка",
                "Положительные отзывы": "Неизвестная ошибка",
                "Отрицательные отзывы": "Неизвестная ошибка",
                "Нейтральные отзывы": "Неизвестная ошибка",
                "positive_aspects": Counter(),
                "negative_aspects": Counter()
            }

    def aggregate_reviews(self, csv_paths: List[str]) -> Dict[str, Union[str, Counter]]:
        all_keywords: List[str] = []
        total_positive_count = total_negative_count = total_neutral_count = 0
        all_positive_aspects = Counter()
        all_negative_aspects = Counter()

        with Pool(processes=4) as pool:
            results = pool.map(self.analyze_reviews, csv_paths)

        for result in results:
            if result["Плюсы"] not in ["Ошибка", "Ошибка кодировки", "Файл не найден", "Неизвестная ошибка"]:
                positive_count = int(result["Положительные отзывы"])
                negative_count = int(result["Отрицательные отзывы"])
                neutral_count = int(result["Нейтральные отзывы"])
                total_positive_count += positive_count
                total_negative_count += negative_count
                total_neutral_count += neutral_count
                all_positive_aspects.update(result["positive_aspects"])
                all_negative_aspects.update(result["negative_aspects"])
                all_keywords.extend(result["Общие ключевые слова"].split(", "))

        main_positives = "\n".join([f"{aspect} ({count})" for aspect, count in all_positive_aspects.most_common(5)]) if all_positive_aspects else "Нет положительных отзывов"
        main_negatives = "\n".join([f"{aspect} ({count})" for aspect, count in all_negative_aspects.most_common(5)]) if all_negative_aspects else "Нет отрицательных отзывов"
        top_keywords = ", ".join(sorted(set(all_keywords))[:15]) if all_keywords else "Нет ключевых слов"

        return {
            "Плюсы (все сайты)": main_positives,
            "Минусы (все сайты)": main_negatives,
            "Ключевые слова (положительные, все сайты)": ", ".join(sorted(self.positive_keywords)[:10]),
            "Ключевые слова (отрицательные, все сайты)": ", ".join(sorted(self.negative_keywords)[:10]),
            "Общие ключевые слова (все сайты)": top_keywords,
            "Положительные отзывы (все сайты)": str(total_positive_count),
            "Отрицательные отзывы (все сайты)": str(total_negative_count),
            "Нейтральные отзывы (все сайты)": str(total_neutral_count),
            "positive_aspects": all_positive_aspects,
            "negative_aspects": all_negative_aspects
        }

    def get_sentiment(self, word: str) -> Tuple[str, float]:
        return SENTIMENT_DICT.get(word, ("нейтральное", 0.0))

    def update_sentiment_dict(self, word: str, sentiment: str, score: float):
        if word in SENTIMENT_DICT:
            current_sentiment, current_score = SENTIMENT_DICT[word]
            new_score = (current_score + score) / 2
            SENTIMENT_DICT[word] = (sentiment, new_score)
        else:
            SENTIMENT_DICT[word] = (sentiment, score)
        logger.info(f"Обновлён словарь тональности: {word} -> {SENTIMENT_DICT[word]}")

    def find_representative_examples(self, aspect: str, aspect_sentiment: str, max_examples: int = 2) -> List[str]:
        examples = []
        aspect_words = set(aspect.lower().split())
        positive_examples = []
        negative_examples = []
        seen_examples = set()  # Для исключения дубликатов

        for review in self.review_data:
            pros_text = review['original_pros'].lower() if review['original_pros'] else ''
            cons_text = review['original_cons'].lower() if review['original_cons'] else ''
            review_text = pros_text + ' ' + cons_text

            # Проверяем, есть ли все слова аспекта в тексте отзыва
            if not all(word in review_text for word in aspect_words):
                continue

            sentiment, _ = self.analyze_sentiment_transformers([review_text])[0]
            # Проверяем соответствие тональности
            if (aspect_sentiment == "положительное" and sentiment != "положительное") or \
            (aspect_sentiment == "отрицательное" and sentiment != "отрицательное"):
                continue

            example_text = review['original_pros'] if pros_text else review['original_cons']
            example = f'"{example_text}" ({review["username"]})'

            # Пропускаем дублирующиеся примеры
            if example in seen_examples:
                continue
            seen_examples.add(example)

            logger.debug(f"Проверяем отзыв для аспекта '{aspect}': текст='{example_text}', тональность={sentiment}")

            if sentiment == "положительное":
                positive_examples.append(example)
            elif sentiment == "отрицательное":
                negative_examples.append(example)

        if aspect_sentiment == "положительное":
            examples.extend(positive_examples[:max_examples])
        elif aspect_sentiment == "отрицательное":
            examples.extend(negative_examples[:max_examples])

        if not examples:
            logger.warning(f"Не удалось найти примеры для аспекта '{aspect}' с тональностью '{aspect_sentiment}'")
        else:
            logger.info(f"Найдены примеры для аспекта '{aspect}': {examples}")

        return examples[:max_examples]

    def analyze_trends(self) -> str:
        if not self.review_data or not any(review['date'] for review in self.review_data):
            logger.warning("Данные о датах отсутствуют в review_data.")
            return "Данные о датах отсутствуют.\n"

        # Словарь для преобразования русских названий месяцев
        month_mapping = {
            "янв": "01", "января": "01",
            "фев": "02", "февраля": "02",
            "мар": "03", "марта": "03",
            "апр": "04", "апреля": "04",
            "май": "05", "мая": "05",
            "июн": "06", "июня": "06",
            "июл": "07", "июля": "07",
            "авг": "08", "августа": "08",
            "сен": "09", "сентября": "09",
            "окт": "10", "октября": "10",
            "ноя": "11", "ноября": "11",
            "дек": "12", "декабря": "12"
        }

        reviews_with_dates = []
        date_formats = [
            "%d.%m.%Y",  # 27.04.2024
            "%Y-%m-%d",  # 2023-10-15
            "%d/%m/%Y",  # 15/10/2023
            "%Y/%m/%d",  # 2023/10/15
            "%d-%m-%Y",  # 15-10-2023
        ]

        for review in self.review_data:
            if not review['date']:
                continue

            date_str = str(review['date']).strip()
            date = None

            for date_format in date_formats:
                try:
                    date = pd.to_datetime(date_str, format=date_format, errors='coerce', dayfirst=True)
                    if pd.notna(date):
                        break
                except Exception:
                    continue

            if pd.isna(date):
                parts = date_str.split()
                if len(parts) == 3:
                    day, month_str, year = parts
                    month_str = month_str.lower()
                    if month_str in month_mapping:
                        month = month_mapping[month_str]
                        day = day.zfill(2)
                        normalized_date_str = f"{day}.{month}.{year}"
                        try:
                            date = pd.to_datetime(normalized_date_str, format="%d.%m.%Y", errors='coerce')
                        except Exception as e:
                            logger.warning(f"Не удалось преобразовать нормализованную дату '{normalized_date_str}': {str(e)}")

            if pd.notna(date):
                reviews_with_dates.append((date, review['rating']))
            else:
                logger.warning(f"Не удалось преобразовать дату '{date_str}' в строке {review}. Пропускаем.")

        if not reviews_with_dates:
            logger.error("Не удалось преобразовать ни одну дату для анализа трендов.")
            return "Не удалось преобразовать даты для анализа.\n"

        df = pd.DataFrame(reviews_with_dates, columns=['date', 'rating'])
        # Агрегируем по годам вместо месяцев
        df['year'] = df['date'].dt.to_period('Y')
        yearly_stats = df.groupby('year')['rating'].agg(['mean', 'count']).reset_index()
        yearly_stats['year'] = yearly_stats['year'].astype(str)

        trend_report = "Тренды по годам:\n"
        trend_report += "-" * 30 + "\n"
        for _, row in yearly_stats.iterrows():
            trend_report += f"{row['year']}: Средняя оценка {row['mean']:.1f} (отзывов: {row['count']})\n"

        if len(yearly_stats) >= 2:
            # Проверяем общий тренд с помощью линейной регрессии
            yearly_stats['index'] = range(len(yearly_stats))
            slope, _ = np.polyfit(yearly_stats['index'], yearly_stats['mean'], 1)
            if slope > 0.1:
                trend_report += f"\nНаблюдается улучшение: средняя оценка выросла с {yearly_stats['mean'].iloc[0]:.1f} до {yearly_stats['mean'].iloc[-1]:.1f}.\n"
            elif slope < -0.1:
                trend_report += f"\nНаблюдается ухудшение: средняя оценка снизилась с {yearly_stats['mean'].iloc[0]:.1f} до {yearly_stats['mean'].iloc[-1]:.1f}.\n"
            else:
                trend_report += "\nОценки относительно стабильны.\n"

        return trend_report + "\n"

    def generate_detailed_report(self, analysis_result: Dict[str, Union[str, Counter]], product_name: str = "продукт", site: str = "Не указан") -> str:
        positive_aspects = analysis_result["positive_aspects"]
        negative_aspects = analysis_result["negative_aspects"]
        positive_count = int(analysis_result["Положительные отзывы"])
        negative_count = int(analysis_result["Отрицательные отзывы"])
        neutral_count = int(analysis_result["Нейтральные отзывы"])
        total_reviews = positive_count + negative_count + neutral_count

        # Начало отчёта
        current_time = datetime.now().strftime("%d.%m.%Y %H:%M")
        report = ""
        if site != "Не указан":
            report += f"Отчёт для {site}:\n\n"
        report += "Отчёт по анализу отзывов\n"
        report += "Автор: Никита Челышев\n"
        report += f"Дата: {current_time}\n"
        report += f"Продукт: {product_name}\n"
        report += "=" * 50 + "\n\n"

        # 1. Основные преимущества
        report += "1. Основные преимущества\n"
        report += "-" * 30 + "\n"
        if not positive_aspects:
            report += "Положительные аспекты отсутствуют.\n\n"
        else:
            for aspect, count in positive_aspects.most_common(5):
                # Пропускаем неинформативные аспекты
                if len(aspect.split()) > 3 or aspect.lower() in ["нет", "нету"]:
                    continue
                examples = self.find_representative_examples(aspect, "положительное")
                # Исключаем неинформативные примеры
                if examples and all("нет" in ex.lower() for ex in examples):
                    examples = []
                examples_str = ", ".join(examples) if examples else "Примеры не найдены."
                formatted_aspect = ' '.join(word.capitalize() for word in aspect.split())
                report += f"- {formatted_aspect} ({count} упоминаний)\n"
                report += f"  Примеры: {examples_str}\n\n"

        # 2. Основные недостатки
        report += "2. Основные недостатки\n"
        report += "-" * 30 + "\n"
        if not negative_aspects:
            report += "Отрицательные аспекты отсутствуют.\n\n"
        else:
            for aspect, count in negative_aspects.most_common(5):
                # Пропускаем неинформативные аспекты
                if len(aspect.split()) > 3 or aspect.lower() in ["нет", "нету"]:
                    continue
                examples = self.find_representative_examples(aspect, "отрицательное")
                # Исключаем неинформативные примеры
                if examples and all("нет" in ex.lower() for ex in examples):
                    examples = []
                examples_str = ", ".join(examples) if examples else "Примеры не найдены."
                formatted_aspect = ' '.join(word.capitalize() for word in aspect.split())
                report += f"- {formatted_aspect} ({count} упоминаний)\n"
                report += f"  Примеры: {examples_str}\n\n"

        # 3. Ключевые слова
        report += "3. Ключевые слова\n"
        report += "-" * 30 + "\n"
        # Фильтруем ключевые слова
        positive_keywords = []
        negative_keywords = []
        for kw in sorted(self.positive_keywords):
            if len(kw.split()) <= 3:
                kw_sentiment, _ = self.analyze_sentiment_transformers([kw])[0]
                if kw_sentiment == "положительное":
                    positive_keywords.append(kw)
        positive_keywords = positive_keywords[:10]

        for kw in sorted(self.negative_keywords):
            if len(kw.split()) <= 3:
                kw_sentiment, _ = self.analyze_sentiment_transformers([kw])[0]
                if kw_sentiment == "отрицательное":
                    negative_keywords.append(kw)
        negative_keywords = negative_keywords[:10]

        common_keywords = sorted(set(positive_keywords).union(negative_keywords))[:15]
        report += f"Положительные: {', '.join(positive_keywords) if positive_keywords else 'Нет'}\n"
        report += f"Отрицательные: {', '.join(negative_keywords) if negative_keywords else 'Нет'}\n"
        report += f"Общие: {', '.join(common_keywords) if common_keywords else 'Нет'}\n\n"

        # 4. Общая статистика
        report += "4. Общая статистика\n"
        report += "-" * 30 + "\n"
        positive_percentage = (positive_count / total_reviews * 100) if total_reviews > 0 else 0
        negative_percentage = (negative_count / total_reviews * 100) if total_reviews > 0 else 0
        neutral_percentage = (neutral_count / total_reviews * 100) if total_reviews > 0 else 0
        average_rating = sum(review['rating'] for review in self.review_data) / total_reviews if total_reviews > 0 else 0
        report += f"Всего отзывов: {total_reviews}\n"
        report += f"Средняя оценка: {average_rating:.1f}/5\n"
        report += f"Положительные отзывы (4-5): {positive_count} ({positive_percentage:.1f}%)\n"
        report += f"Отрицательные отзывы (1-2): {negative_count} ({negative_percentage:.1f}%)\n"
        report += f"Нейтральные отзывы (3): {neutral_count} ({neutral_percentage:.1f}%)\n\n"

        # 5. Анализ трендов
        report += "5. Анализ трендов\n"
        report += "-" * 30 + "\n"
        trend_analysis = self.analyze_trends()
        report += trend_analysis

        # 6. Общее впечатление
        report += "6. Общее впечатление\n"
        report += "-" * 30 + "\n"
        top_positive_aspects = [aspect for aspect, _ in positive_aspects.most_common(3) if len(aspect.split()) <= 3 and aspect.lower() != product_name.lower()]
        top_negative_aspects = [aspect for aspect, _ in negative_aspects.most_common(3) if len(aspect.split()) <= 3 and aspect.lower() != product_name.lower()]
        if positive_percentage > negative_percentage:
            impression = f"Пользователи в целом довольны продуктом '{product_name}'.\n"
            impression += f"Основные преимущества: {', '.join(top_positive_aspects).lower() if top_positive_aspects else 'не указаны'}.\n"
            impression += f"Положительные отзывы составляют {positive_percentage:.1f}%.\n"
            if top_negative_aspects:
                impression += f"Недостатки: {', '.join(top_negative_aspects).lower()} ({negative_percentage:.1f}% отзывов).\n"
        else:
            impression = f"Впечатления о продукте '{product_name}' смешанные.\n"
            impression += f"Преимущества: {', '.join(top_positive_aspects).lower() if top_positive_aspects else 'не указаны'} ({positive_percentage:.1f}% отзывов).\n"
            impression += f"Недостатки: {', '.join(top_negative_aspects).lower() if top_negative_aspects else 'не указаны'} ({negative_percentage:.1f}% отзывов).\n"
        report += impression + "\n"

        # 7. Рекомендации
        report += "7. Рекомендации\n"
        report += "-" * 30 + "\n"
        report += f"Средняя оценка продукта: {average_rating:.1f}/5.\n"
        if average_rating >= 4.0:
            report += "Продукт получает высокую оценку.\n"
        elif average_rating >= 3.0:
            report += "Продукт воспринимается умеренно.\n"
        else:
            report += "Продукт имеет значительные проблемы.\n"
        if negative_aspects:
            top_negatives = [aspect for aspect, _ in negative_aspects.most_common(2) if len(aspect.split()) <= 3 and aspect.lower() != product_name.lower()]
            if top_negatives:
                report += f"Рекомендуется улучшить: {', '.join(top_negatives).lower()}.\n"
                # Добавляем конкретные рекомендации на основе ключевых слов
                negative_text = " ".join(top_negatives).lower()
                if "тухлый" in negative_text or "прогорклый" in negative_text or "вонючий" in negative_text:
                    report += "Рекомендация: обратить внимание на контроль качества и свежести продукта.\n"
                elif "размер" in negative_text or "мелкий" in negative_text:
                    report += "Рекомендация: увеличить размер креветок или уточнить информацию о размере в описании.\n"
                elif "цена" in negative_text or "дорогой" in negative_text:
                    report += "Рекомендация: пересмотреть ценовую политику или предложить акции.\n"
                elif "упаковка" in negative_text:
                    report += "Рекомендация: улучшить качество упаковки или её удобство.\n"
                else:
                    report += "Рекомендация: провести дополнительный анализ для выявления причин недовольства.\n"
            else:
                report += "Рекомендуется провести дополнительный анализ для выявления проблем.\n"
        else:
            report += "Рекомендаций по улучшению нет.\n"

        # Сохранение результатов анализа в JSON
        result_data = {
            "positive_aspects": dict(positive_aspects),
            "negative_aspects": dict(negative_aspects),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "average_rating": average_rating,
            "positive_keywords": list(self.positive_keywords),
            "negative_keywords": list(self.negative_keywords)
        }
        with open("analysis_results.json", "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=4)

        return report

# Юнит-тесты
if __name__ == "__main__":
    import unittest

    class TestReviewAnalyzer(unittest.TestCase):
        def setUp(self):
            self.analyzer = ReviewAnalyzer(use_preprocessing=False)

        def test_preprocess_text(self):
            text = "Хороший продукт 👍"
            expected = "хороший продукт положительный_эмодзи"
            result = self.analyzer.preprocess_text(text)
            self.assertEqual(result, expected)

        def test_analyze_sentiment_transformers(self):
            texts = ["Отличный продукт!", "Ужасный сервис"]
            results = self.analyzer.analyze_sentiment_transformers(texts)
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0][0], "положительное")
            self.assertEqual(results[1][0], "отрицательное")

        def test_split_mixed_sentence(self):
            sentence = "Экран яркий, но быстро разряжается"
            expected = ["экран яркий", "быстро разряжается"]
            result = self.analyzer.split_mixed_sentence(sentence)
            self.assertEqual(result, expected)

        def test_validate_csv(self):
            data = pd.DataFrame({
                "Текст отзыва": ["Хороший продукт", "Плохой сервис"],
                "Оценка": [5, 1]
            })
            self.assertTrue(self.analyzer.validate_csv(data))
            invalid_data = pd.DataFrame({
                "Текст отзыва": ["Хороший продукт"],
                "Оценка": [6]
            })
            with self.assertRaises(ValueError):
                self.analyzer.validate_csv(invalid_data)

        def test_extract_aspects(self):
            sentence = "Экран яркий, но батарея слабая"
            aspects = self.analyzer.extract_aspects(sentence)
            self.assertTrue(len(aspects) > 0)
            self.assertIn(aspects[0][1], ["положительное", "отрицательное", "нейтральное"])

        def test_analyze_review_sentences(self):
            review_text = "Продукт отличный, но доставка медленная."
            result = self.analyzer.analyze_review_sentences(review_text)
            self.assertTrue(len(result) > 0)
            self.assertIn(result[0][1], ["положительное", "отрицательное", "нейтральное"])

    unittest.main()