import pandas as pd
import spacy
from collections import Counter
import logging
import sys
from typing import List, Dict, Set, Union, Tuple
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pymorphy3
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Попытка импорта YandexSpeller с обработкой ошибки
try:
    from pyaspeller import YandexSpeller
    speller = YandexSpeller()
except ImportError:
    logging.warning("Модуль pyaspeller недоступен. Предобработка текста будет отключена.")
    speller = None

# Настройка логирования с поддержкой UTF-8
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    encoding='utf-8-sig'  # Убедимся, что используется UTF-8
)
logger = logging.getLogger(__name__)

# Замена эмодзи и специальных символов для логирования
def replace_emoji_for_logging(text: str) -> str:
    emoji_dict = {
        "👍": "[положительный_эмодзи]",
        "👌": "[положительный_эмодзи]",
        "😊": "[положительное_эмоция]",
        "🙂": "[положительное_эмоция]",
        "😍": "[положительное_эмоция]",
        "😢": "[отрицательное_эмоция]",
        "😡": "[отрицательное_эмоция]",
        "😠": "[отрицательное_эмоция]",
        "😋": "[положительное_эмоция]",
        "！": "[восклицание]",
        "‼": "[двойное_восклицание]",
        "？": "[вопросительный_знак]",
        "🌻": "[цветок_эмодзи]",
        "🚘": "[машина_эмодзи]"
    }
    for emoji, label in emoji_dict.items():
        text = text.replace(emoji, f" {label} ")
    return text

# Инициализация spaCy для русского языка
nlp = spacy.load("ru_core_news_sm", disable=["ner", "lemmatizer"])

# Инициализация pymorphy3 для нормализации слов
morph = pymorphy3.MorphAnalyzer()

# Инициализация модели для анализа тональности
model_name = "seara/rubert-tiny2-russian-sentiment"
sentiment_analyzer = None
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    logger.info(f"Модель {model_name} успешно загружена.")
except Exception as e:
    logger.warning(f"Ошибка загрузки модели {model_name}: {e}. Используется словарный анализ.")
    sentiment_analyzer = None

# Расширенный словарь тональности с усилением негативных слов
SENTIMENT_DICT = {
    "хороший": ("положительное", 0.7),
    "отличный": ("положительное", 0.9),
    "прекрасный": ("положительное", 0.8),
    "удобный": ("положительное", 0.6),
    "качественный": ("положительное", 0.7),
    "быстрый": ("положительное", 0.6),
    "надежный": ("положительное", 0.7),
    "вкусный": ("положительное", 0.7),
    "красивый": ("положительное", 0.6),
    "приятный": ("положительное", 0.6),
    "замечательный": ("положительное", 0.8),
    "свежий": ("положительное", 0.6),
    "вежливый": ("положительное", 0.7),
    "плохой": ("отрицательное", -0.7),
    "ужасный": ("отрицательное", -0.9),
    "дефектный": ("отрицательное", -0.8),
    "неудобный": ("отрицательное", -0.6),
    "медленный": ("отрицательное", -0.6),
    "ненадежный": ("отрицательное", -0.7),
    "невкусный": ("отрицательное", -0.7),
    "сломанный": ("отрицательное", -0.8),
    "грязный": ("отрицательное", -0.7),
    "дорогой": ("отрицательное", -0.6),
    "безвкусный": ("отрицательное", -0.7),
    "прогорклый": ("отрицательное", -0.8),
    "протухший": ("отрицательное", -0.9),
    "трындец": ("отрицательное", -0.9),
    "худший": ("отрицательное", -0.8),
    "хамоватый": ("отрицательное", -0.7),
    "неквалифицированный": ("отрицательное", -0.7),
    "высокий": ("нейтральное", 0.0),
    "просто": ("нейтральное", 0.0),
    "развивается": ("нейтральное", 0.0),
    "хранится": ("нейтральное", 0.0),
    "хранении": ("нейтральное", 0.0),
    "обслуживание": ("нейтральное", 0.0),
    "персонал": ("нейтральное", 0.0),
    "топливо": ("нейтральное", 0.0),
    "нет": ("отрицательное", -0.5),
    "нету": ("отрицательное", -0.5),
    "очень": ("усилитель", 1.5),
    "слегка": ("ослабитель", 0.5),
    "совершенно": ("усилитель", 1.3),
    "абсолютно": ("усилитель", 1.4),
    "немного": ("ослабитель", 0.7),
    "дрянь": ("отрицательное", -0.8),
    "обман": ("отрицательное", -0.7),
    "воруют": ("отрицательное", -0.9),
    "хмурый": ("отрицательное", -0.6),
    "недосмотр": ("отрицательное", -0.6),
    "отвратительно": ("отрицательное", -0.9),
    "воровство": ("отрицательное", -0.9),
    "обманывают": ("отрицательное", -0.8),
    "не рекомендую": ("отрицательное", -0.7),
    "жесть": ("отрицательное", -0.7),
    "стыдобень": ("отрицательное", -0.7),
    "недолив": ("отрицательное", -0.8),
    "хамство": ("отрицательное", -0.7),
    "дорого": ("отрицательное", -0.7),  # Усилен вес
    "дороговато": ("отрицательное", -0.8),
    "мало": ("отрицательное", -0.6),  # Добавлен вес
    "маленький": ("отрицательное", -0.5),  # Добавлен вес
    "маленькие": ("отрицательное", -0.5),
    "маловато": ("отрицательное", -0.6),
    "странный": ("отрицательное", -0.6),  # Для "вкус странный"
    "плохой": ("отрицательное", -0.7)
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
        self.sentiment_cache = {}
        self.domain_hints = ["рыба", "мясо", "еда", "экран", "камера", "креветка", "минтай", "гребешок", "вкус", "качество", "размер", "топливо", "сервис", "обслуживание", "персонал", "баллы", "АЗС", "бензин", "дизель", "карта", "цена", "приложение", "состав", "упаковка", "икра"]

    def preprocess_text(self, text: str) -> str:
        if not self.use_preprocessing or not text.strip():
            return text
        if speller is None:
            logger.warning("pyaspeller не доступен, предобработка текста отключена.")
            return text
        try:
            emoji_dict = {
                "👍": "положительный_эмодзи",
                "👌": "положительный_эмодзи",
                "😊": "положительное_эмоция",
                "🙂": "положительное_эмоция",
                "😍": "положительное_эмоция",
                "😢": "отрицательное_эмоция",
                "😡": "отрицательное_эмоция",
                "😠": "отрицательное_эмоция",
                "😋": "положительное_эмоция",
                "！": "восклицание",
                "‼": "двойное_восклицание",
                "？": "вопросительный_знак",
                "🌻": "цветок_эмодзи",
                "🚘": "машина_эмодзи"
            }
            for emoji, label in emoji_dict.items():
                text = text.replace(emoji, f" {label} ")
            corrected = speller.spelled(text)
            typo_corrections = {
                "при хари": "при хранении",
                "хари": "хранении",
            }
            for typo, correction in typo_corrections.items():
                corrected = corrected.replace(typo, correction)
            doc = nlp(corrected)
            normalized_tokens = []
            token_cache = {}
            for token in doc:
                if token.is_punct or token.is_stop:
                    normalized_tokens.append(token.text)
                    continue
                if token.text in token_cache:
                    normalized_tokens.append(token_cache[token.text])
                    continue
                parsed_word = morph.parse(token.text)[0]
                normal_form = parsed_word.normal_form
                token_cache[token.text] = normal_form
                normalized_tokens.append(normal_form)
            return " ".join(normalized_tokens).strip()
        except Exception as e:
            logger.warning(f"Ошибка при исправлении опечаток: {str(e)}")
            return text

    def split_sentences(self, text: str) -> List[str]:
        text = re.sub(r'[!?.]+\)+|\)+|[:;]-?\)+', '.', text)
        text = re.sub(r'(\.+|\!+|\?+)', r'. ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return sentences

    def split_mixed_sentence(self, sentence: str) -> List[str]:
        # Разбиение предложений типа "хорошо, но дорого"
        parts = re.split(r'\s+(но|а|или)\s+', sentence.lower())
        if len(parts) > 1:
            return [part.strip() for part in parts if part.strip()]
        return [sentence]

    def check_modifiers_with_dependencies(self, sentence: str) -> float:
        doc = nlp(sentence)
        sentiment_modifier = 1.0
        negation_count = 0
        intensifier = 1.0

        for token in doc:
            if token.lemma_ in ["не", "нет", "ни", "едва"] and token.dep_ in ["neg"]:
                head = token.head
                if head.pos_ in ["ADJ", "ADV", "VERB"]:
                    negation_count += 1
                for child in head.children:
                    if child.lemma_ in ["не", "нет"] and child.dep_ in ["neg"]:
                        negation_count += 1
            elif token.lemma_ in ["очень", "крайне", "сильно", "абсолютно", "совершенно"] and token.dep_ in ["advmod"]:
                intensifier = 1.5
            elif token.lemma_ in ["слегка", "немного", "чуть", "еле"] and token.dep_ in ["advmod"]:
                intensifier = 0.5
            elif token.lemma_ in ["вряд ли", "едва ли"] and token.dep_ in ["advmod"]:
                negation_count += 1

        if negation_count % 2 == 1:
            sentiment_modifier = -intensifier
        else:
            sentiment_modifier = intensifier
        logger.debug(f"Модификатор для '{replace_emoji_for_logging(sentence)}': {sentiment_modifier}, negation_count={negation_count}")
        return sentiment_modifier

    def analyze_sentiment_transformers(self, text: str) -> Tuple[str, float]:
        if not text.strip():
            return 'нейтральное', 0.0

        if text in self.sentiment_cache:
            return self.sentiment_cache[text]

        if sentiment_analyzer:
            try:
                result = sentiment_analyzer(text)[0]
                label = result['label'].lower()
                score = result['score']
                logger.info(f"Модель вернула для текста '{replace_emoji_for_logging(text)}': label={label}, score={score}")

                doc = nlp(text)
                token_count = len([token for token in doc if not token.is_punct])
                MIN_CONFIDENCE_THRESHOLD = 0.6  # Уменьшен до 0.6 для большей чувствительности

                # Проверка на ключевые слова для переопределения neutral
                has_negative = any(token.lemma_ in [k for k, (s, _) in SENTIMENT_DICT.items() if s == "отрицательное"] for token in doc)
                has_positive = any(token.lemma_ in [k for k, (s, _) in SENTIMENT_DICT.items() if s == "положительное"] for token in doc)
                negative_boost = 0
                if has_negative:
                    negative_words = [token.lemma_ for token in doc if SENTIMENT_DICT.get(token.lemma_, (None, None))[0] == "отрицательное"]
                    negative_boost = -0.7 * len(negative_words) * min(abs(SENTIMENT_DICT.get(word, (None, 0.0))[1]) for word in negative_words if SENTIMENT_DICT.get(word, (None, None))[0] == "отрицательное")
                    for i in range(len(negative_words) - 1):
                        if negative_words[i] in ["очень", "мега"] and negative_words[i + 1] in ["мало", "дорого", "маленький"]:
                            negative_boost *= 1.5  # Усиление комбинаций

                if label == "positive" and score > MIN_CONFIDENCE_THRESHOLD:
                    base_score = score
                elif label == "negative" and score > MIN_CONFIDENCE_THRESHOLD:
                    base_score = -score
                elif label == "neutral":
                    if has_negative and negative_boost < -0.5:
                        base_score = negative_boost
                    elif has_positive:
                        base_score = 0.6
                    else:
                        base_score = 0.0
                else:
                    base_score = 0.0

                modifier = self.check_modifiers_with_dependencies(text)
                adjusted_score = base_score * modifier

                if "!" in text:
                    adjusted_score *= 1.2
                elif "..." in text or "?" in text:
                    adjusted_score *= 0.8

                if score < MIN_CONFIDENCE_THRESHOLD and abs(adjusted_score) < 0.5:
                    fallback_sentiment, fallback_score = self.fallback_sentiment_analysis(text)
                    combined_score = (score * 0.4) + (abs(fallback_score) * 0.6) * (-1 if fallback_sentiment == "отрицательное" else 1)
                    adjusted_score = combined_score

                sentiment = "положительное" if adjusted_score > 0.4 else "отрицательное" if adjusted_score < -0.4 else "нейтральное"  # Уменьшены пороги
                self.sentiment_cache[text] = (sentiment, adjusted_score)
                return sentiment, adjusted_score
            except Exception as e:
                logger.warning(f"Ошибка анализа тональности с моделью: {e}. Переход к словарному анализу.")
                return self.fallback_sentiment_analysis(text)
        else:
            return self.fallback_sentiment_analysis(text)

    def fallback_sentiment_analysis(self, text: str) -> Tuple[str, float]:
        doc = nlp(text.lower())
        sentiment_scores = {"положительное": 0, "отрицательное": 0}
        for token in doc:
            sentiment, score = self.get_sentiment(token.lemma_)
            if sentiment in ["положительное", "отрицательное"]:
                sentiment_scores[sentiment] += score
        modifier = self.check_modifiers_with_dependencies(text)
        total_score = sentiment_scores["положительное"] + sentiment_scores["отрицательное"]
        total_score *= modifier
        if total_score > 0.4:
            return "положительное", total_score
        elif total_score < -0.4:
            return "отрицательное", total_score
        else:
            return "нейтральное", 0.0

    def extract_aspects(self, sentence: str) -> List[Tuple[str, str, float]]:
        doc = nlp(sentence)
        aspects = []
        invalid_words = {"оченк", "поробовать", "спасного", "заскучаться", "добовство", "хари"}

        if sentence.lower().strip() in ["нет", "нету"]:
            sentiment, score = self.get_sentiment("нет")
            aspects.append(("нет", sentiment, score))
            logger.info(f"Извлечён аспект 'нет': тональность={sentiment}, скор={score}")
            return aspects

        sentiment, score = self.analyze_sentiment_transformers(sentence)

        # Ограничение длины аспекта
        MAX_ASPECT_LENGTH = 4

        # Разбиение смешанных предложений
        clauses = self.split_mixed_sentence(sentence)
        for clause in clauses:
            doc_clause = nlp(clause)
            for token in doc_clause:
                if (token.pos_ in ["NOUN", "ADJ", "VERB"] and token.lemma_ not in invalid_words and 
                    (token.lemma_ in self.domain_hints or token.lemma_ in [k for k, _ in SENTIMENT_DICT.items()])):
                    aspect_phrase = token.lemma_
                    modifiers = []
                    negation = False
                    children_count = 0

                    for child in token.children:
                        if child.dep_ in ["amod", "compound", "advmod"] and child.lemma_ not in invalid_words and children_count < MAX_ASPECT_LENGTH - 1:
                            modifiers.append(child.lemma_)
                            children_count += 1
                        if child.lemma_ in ["не", "нет", "ни", "едва"] and child.dep_ in ["neg"]:
                            negation = True

                    aspect_phrase = " ".join(modifiers + [aspect_phrase]).strip()
                    if not aspect_phrase or aspect_phrase.lower() in self.invalid_phrases or len(aspect_phrase.split()) > MAX_ASPECT_LENGTH:
                        continue

                    aspect_sentiment = sentiment
                    aspect_score = score

                    if negation or any(mod in [k for k, (s, _) in SENTIMENT_DICT.items() if s == "отрицательное"] for mod in modifiers):
                        aspect_sentiment = "отрицательное"
                        aspect_score = -abs(aspect_score) if aspect_score > 0 else aspect_score

                    aspects.append((aspect_phrase, aspect_sentiment, aspect_score))
                    logger.info(f"Извлечён аспект: '{replace_emoji_for_logging(aspect_phrase)}', тональность: {aspect_sentiment}, скор: {aspect_score}")

            if not aspects and sentiment != "нейтральное" and len(sentence.split()) <= MAX_ASPECT_LENGTH:
                aspects.append((sentence.strip(), sentiment, score))
                logger.info(f"Извлечён аспект (из всего предложения): '{replace_emoji_for_logging(sentence.strip())}', тональность: {sentiment}, скор: {score}")

        return aspects

    def analyze_review_sentences(self, review_text: str) -> List[Tuple[str, str, float, List[Tuple[str, str, float]]]]:
        review_text = self.preprocess_text(review_text)
        sentences = self.split_sentences(review_text)
        result = []
        for sentence in sentences:
            doc = nlp(sentence)
            clauses = []
            current_clause = []
            for token in doc:
                current_clause.append(token.text)
                if token.dep_ in ["cc", "punct"] and token.lemma_ in ["и", "но", "а", "или", ","]:
                    if current_clause:
                        clauses.append(" ".join(current_clause).strip())
                        current_clause = []
            if current_clause:
                clauses.append(" ".join(current_clause).strip())

            for clause in clauses:
                sentiment, score = self.analyze_sentiment_transformers(clause)
                aspects = self.extract_aspects(clause)
                result.append((clause, sentiment, score, aspects))
                logger.info(f"Анализ предложения '{replace_emoji_for_logging(clause)}': тональность={sentiment}, скор={score}, аспекты={aspects}")
        return result

    def analyze_reviews(self, csv_path: str) -> Dict[str, str]:
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
            positive_count = 0
            negative_count = 0
            positive_aspects = Counter()
            negative_aspects = Counter()
            self.positive_keywords.clear()
            self.negative_keywords.clear()
            processed_texts_set = set()

            if all(col in df.columns for col in ['Достоинства', 'Недостатки', 'Оценка']):
                texts = [(str(row['Достоинства']) if pd.notna(row['Достоинства']) else "",
                         str(row['Недостатки']) if pd.notna(row['Недостатки']) else "",
                         int(row['Оценка']) if pd.notna(row['Оценка']) else 3)
                        for _, row in df.iterrows()]
                for pros_text, cons_text, rating in texts:
                    if pros_text.strip():
                        if pros_text.strip().lower() in ["нет", "нету"]:
                            negative_aspects["нет"] += 1
                            self.negative_keywords.add("нет")
                        else:
                            processed_texts_set.add(pros_text)
                            pros_sentences = self.analyze_review_sentences(pros_text)
                            for _, sentiment, _, aspects in pros_sentences:
                                if sentiment == "положительное":
                                    for aspect_phrase, aspect_sentiment, _ in aspects:
                                        if aspect_sentiment == "положительное":
                                            positive_aspects[aspect_phrase] += 1
                                            self.positive_keywords.add(aspect_phrase)
                                        elif aspect_sentiment == "отрицательное":
                                            negative_aspects[aspect_phrase] += 1
                                            self.negative_keywords.add(aspect_phrase)
                                elif sentiment == "отрицательное":
                                    for aspect_phrase, aspect_sentiment, _ in aspects:
                                        if aspect_sentiment == "отрицательное":
                                            negative_aspects[aspect_phrase] += 1
                                            self.negative_keywords.add(aspect_phrase)
                    if cons_text.strip():
                        if cons_text.strip().lower() in ["нет", "нету"]:
                            positive_aspects["нет недостатков"] += 1
                            self.positive_keywords.add("нет недостатков")
                        else:
                            processed_texts_set.add(cons_text)
                            cons_sentences = self.analyze_review_sentences(cons_text)
                            for _, sentiment, _, aspects in cons_sentences:
                                if sentiment == "положительное":
                                    for aspect_phrase, aspect_sentiment, _ in aspects:
                                        if aspect_sentiment == "положительное":
                                            positive_aspects[aspect_phrase] += 1
                                            self.positive_keywords.add(aspect_phrase)
                                elif sentiment == "отрицательное":
                                    for aspect_phrase, aspect_sentiment, _ in aspects:
                                        if aspect_sentiment == "отрицательное":
                                            negative_aspects[aspect_phrase] += 1
                                            self.negative_keywords.add(aspect_phrase)
                    if rating >= self.positive_threshold:
                        positive_count += 1
                    elif rating <= self.negative_threshold:
                        negative_count += 1

            elif all(col in df.columns for col in ['Текст отзыва', 'Оценка']):
                texts = [(str(row['Текст отзыва']), int(row['Оценка']) if pd.notna(row['Оценка']) else 3)
                        for _, row in df.iterrows()]
                for text, rating in texts:
                    if text.strip():
                        processed_texts_set.add(text)
                        sentences = self.analyze_review_sentences(text)
                        for _, sentiment, _, aspects in sentences:
                            if sentiment == "положительное":
                                for aspect_phrase, aspect_sentiment, _ in aspects:
                                    if aspect_sentiment == "положительное":
                                        positive_aspects[aspect_phrase] += 1
                                        self.positive_keywords.add(aspect_phrase)
                            elif sentiment == "отрицательное":
                                for aspect_phrase, aspect_sentiment, _ in aspects:
                                    if aspect_sentiment == "отрицательное":
                                        negative_aspects[aspect_phrase] += 1
                                        self.negative_keywords.add(aspect_phrase)
                    if rating >= self.positive_threshold:
                        positive_count += 1
                    elif rating <= self.negative_threshold:
                        negative_count += 1

            else:
                raise ValueError("CSV-файл должен содержать либо столбцы 'Достоинства', 'Недостатки' и 'Оценка', либо 'Текст отзыва' и 'Оценка'")

            common_aspects = set(positive_aspects.keys()).intersection(set(negative_aspects.keys()))
            for aspect in common_aspects:
                sentiment, _ = self.analyze_sentiment_transformers(aspect)
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

            return {
                "Плюсы": main_positives,
                "Минусы": main_negatives,
                "Ключевые слова (положительные)": ", ".join(sorted(self.positive_keywords)[:10]),
                "Ключевые слова (отрицательные)": ", ".join(sorted(self.negative_keywords)[:10]),
                "Общие ключевые слова": ", ".join(sorted(set(self.positive_keywords).union(self.negative_keywords))[:15]),
                "Положительные отзывы": str(positive_count),
                "Отрицательные отзывы": str(negative_count)
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
                "Отрицательные отзывы": "Ошибка кодировки"
            }
        except Exception as e:
            logger.error(f"Ошибка при анализе файла {csv_path}: {str(e)}")
            return {
                "Плюсы": "Ошибка",
                "Минусы": "Ошибка",
                "Ключевые слова (положительные)": "Ошибка",
                "Ключевые слова (отрицательные)": "Ошибка",
                "Общие ключевые слова": "Ошибка",
                "Положительные отзывы": "Ошибка",
                "Отрицательные отзывы": "Ошибка"
            }

    def aggregate_reviews(self, csv_paths: List[str]) -> Dict[str, str]:
        all_keywords: List[str] = []
        total_positive_count = 0
        total_negative_count = 0
        all_positive_aspects = Counter()
        all_negative_aspects = Counter()

        def process_single_file(csv_path):
            return self.analyze_reviews(csv_path)

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_single_file, csv_paths))

        for result in results:
            if result["Плюсы"] != "Ошибка" and result["Плюсы"] != "Ошибка кодировки":
                positive_count = int(result["Положительные отзывы"])
                negative_count = int(result["Отрицательные отзывы"])
                total_positive_count += positive_count
                total_negative_count += negative_count
                for line in result["Плюсы"].split("\n"):
                    if "(" in line:
                        aspect, count = line.split(" (")
                        count = int(count[:-1])
                        all_positive_aspects[aspect] += count
                for line in result["Минусы"].split("\n"):
                    if "(" in line:
                        aspect, count = line.split(" (")
                        count = int(count[:-1])
                        all_negative_aspects[aspect] += count
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
            "Отрицательные отзывы (все сайты)": str(total_negative_count)
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