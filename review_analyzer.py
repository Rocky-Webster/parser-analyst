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
from datetime import datetime
import json

# Попытка импорта YandexSpeller
try:
    from pyaspeller import YandexSpeller
    speller = YandexSpeller()
except ImportError:
    logging.warning("Модуль pyaspeller недоступен. Предобработка текста будет отключена.")
    speller = None

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('review_analyzer.log', encoding='utf-8')
    ],
    encoding='utf-8-sig'
)
logger = logging.getLogger(__name__)

# Расширенная замена эмодзи
def replace_emoji_for_logging(text: str) -> str:
    emoji_dict = {
        "👍": "[положительный_эмодзи]", "👌": "[положительный_эмодзи]", "😊": "[положительное_эмоция]",
        "🙂": "[положительное_эмоция]", "😍": "[положительное_эмоция]", "😢": "[отрицательное_эмоция]",
        "😡": "[отрицательное_эмоция]", "😠": "[отрицательное_эмоция]", "😋": "[положительное_эмоция]",
        "！": "восклицание", "‼": "двойное_восклицание", "？": "вопросительный_знак",
        "🌻": "цветок_эмодзи", "🚘": "машина_эмодзи", "👎": "[отрицательный_эмодзи]",
        "😞": "[отрицательное_эмоция]", "🤔": "[нейтральное_эмоция]", "💡": "[идея_эмодзи]"
    }
    for emoji, label in emoji_dict.items():
        text = text.replace(emoji, f" {label} ")
    return text.strip()

# Инициализация spaCy и pymorphy3
nlp = spacy.load("ru_core_news_sm", disable=["ner"])
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
        self.sentiment_cache = {}
        self.domain_hints = [
            "рыба", "мясо", "еда", "экран", "камера", "креветка", "минтай", "гребешок", "вкус", "качество", "размер",
            "топливо", "сервис", "обслуживание", "персонал", "баллы", "АЗС", "бензин", "дизель", "карта", "цена",
            "приложение", "состав", "упаковка", "икра"
        ]
        self.review_data = []
        self.stop_words = set(["и", "в", "на", "с", "по", "для", "к", "у", "из", "о", "а", "но", "или"])

    def preprocess_text(self, text: str) -> str:
        if not self.use_preprocessing or not text.strip():
            return text
        if speller is None:
            logger.warning("pyaspeller не доступен, предобработка текста отключена.")
            return text
        try:
            emoji_dict = {
                "👍": "положительный_эмодзи", "👌": "положительный_эмодзи", "😊": "положительное_эмоция",
                "🙂": "положительное_эмоция", "😍": "положительное_эмоция", "😢": "отрицательное_эмоция",
                "😡": "отрицательное_эмоция", "😠": "отрицательное_эмоция", "😋": "положительное_эмоция",
                "！": "восклицание", "‼": "двойное_восклицание", "？": "вопросительный_знак",
                "🌻": "цветок_эмодзи", "🚘": "машина_эмодзи", "👎": "отрицательный_эмодзи",
                "😞": "отрицательное_эмоция", "🤔": "нейтральное_эмоция", "💡": "идея_эмодзи"
            }
            for emoji, label in emoji_dict.items():
                text = text.replace(emoji, f" {label} ")
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
        parts = re.split(r'\s+(но|а|или)\s+', sentence.lower())
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
                has_negative = any(token.lemma_ in [k for k, (s, _) in SENTIMENT_DICT.items() if s == "отрицательное"] for token in doc)
                has_positive = any(token.lemma_ in [k for k, (s, _) in SENTIMENT_DICT.items() if s == "положительное"] for token in doc)
                negative_boost = 0
                if has_negative:
                    negative_words = [token.lemma_ for token in doc if SENTIMENT_DICT.get(token.lemma_, (None, None))[0] == "отрицательное"]
                    negative_boost = -0.7 * len(negative_words) * min(abs(SENTIMENT_DICT.get(word, (None, 0.0))[1]) for word in negative_words if SENTIMENT_DICT.get(word))
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
                self.sentiment_cache[text] = (sentiment, adjusted_score)
                return sentiment, adjusted_score
            except Exception as e:
                logger.warning(f"Ошибка анализа тональности: {e}. Используется словарный метод.")
                return self.fallback_sentiment_analysis(text)
        return self.fallback_sentiment_analysis(text)

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
        MAX_ASPECT_LENGTH = 5

        if sentence.lower().strip() in ["нет", "нету"]:
            sentiment, score = self.get_sentiment("нет")
            aspects.append(("нет", sentiment, score, sentence))
            logger.info(f"Извлечён аспект 'нет': тональность={sentiment}, скор={score}")
            return aspects

        sentiment, score = self.analyze_sentiment_transformers(sentence)
        clauses = self.split_mixed_sentence(sentence)

        for clause in clauses:
            doc_clause = nlp(clause)
            for token in doc_clause:
                if (token.pos_ in ["NOUN", "ADJ", "VERB"] and token.lemma_ not in invalid_words and 
                    (token.lemma_ in self.domain_hints or token.lemma_ in [k for k, _ in SENTIMENT_DICT.items()])):
                    modifiers = []
                    negation = False
                    children_count = 0
                    for child in token.children:
                        if child.dep_ in ["amod", "compound", "advmod", "nmod", "obj"] and child.lemma_ not in invalid_words and children_count < MAX_ASPECT_LENGTH - 1:
                            modifiers.append(child.lemma_)
                            children_count += 1
                        if child.lemma_ in ["не", "нет", "ни", "едва"] and child.dep_ in ["neg"]:
                            negation = True
                    if token.dep_ in ["amod", "nmod", "obj"]:
                        parent = token.head
                        if parent.pos_ in ["NOUN", "VERB"] and parent.lemma_ not in invalid_words:
                            modifiers.append(parent.lemma_)
                    aspect_phrase = " ".join(sorted([mod for mod in modifiers if mod] + [token.lemma_])).strip()
                    if not aspect_phrase or aspect_phrase.lower() in self.invalid_phrases or len(aspect_phrase.split()) > MAX_ASPECT_LENGTH:
                        continue
                    aspect_sentiment = sentiment
                    aspect_score = score
                    if negation or any(mod in [k for k, (s, _) in SENTIMENT_DICT.items() if s == "отрицательное"] for mod in modifiers):
                        aspect_sentiment = "отрицательное"
                        aspect_score = -abs(aspect_score) if aspect_score > 0 else aspect_score
                    elif any(mod in [k for k, (s, _) in SENTIMENT_DICT.items() if s == "положительное"] for mod in modifiers):
                        aspect_sentiment = "положительное"
                        aspect_score = abs(aspect_score)
                    aspects.append((aspect_phrase, aspect_sentiment, aspect_score, clause))
                    logger.info(f"Извлечён аспект: '{replace_emoji_for_logging(aspect_phrase)}', тональность: {aspect_sentiment}, скор: {aspect_score}, из текста: '{clause}'")
            if not aspects and sentiment != "нейтральное" and len(sentence.split()) <= MAX_ASPECT_LENGTH:
                aspects.append((sentence.strip(), sentiment, score, sentence))
                logger.info(f"Извлечён аспект (из всего предложения): '{replace_emoji_for_logging(sentence.strip())}', тональность: {sentiment}, скор: {score}")
        return aspects

    def analyze_review_sentences(self, review_text: str) -> List[Tuple[str, str, float, List[Tuple[str, str, float, str]]]]:
        review_text = self.preprocess_text(review_text)
        sentences = self.split_sentences(review_text)
        result = []
        for sentence in sentences:
            try:
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
            except Exception as e:
                logger.error(f"Ошибка при анализе предложения '{replace_emoji_for_logging(sentence)}': {str(e)}")
                result.append((sentence, "нейтральное", 0.0, []))
        return result

    def analyze_reviews(self, csv_path: str) -> Dict[str, Union[str, Counter]]:
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
            positive_count = negative_count = neutral_count = 0
            positive_aspects = Counter()
            negative_aspects = Counter()
            self.positive_keywords.clear()
            self.negative_keywords.clear()
            self.review_data = []
            processed_texts_set = set()

            if all(col in df.columns for col in ['Достоинства', 'Недостатки', 'Оценка']):
                texts = [(str(row['Достоинства']) if pd.notna(row['Достоинства']) else "",
                         str(row['Недостатки']) if pd.notna(row['Недостатки']) else "",
                         int(row['Оценка']) if pd.notna(row['Оценка']) else 3,
                         str(row.get('Имя пользователя', 'Аноним')))
                        for _, row in df.iterrows()]
                for pros_text, cons_text, rating, username in texts:
                    self.review_data.append({
                        'username': username,
                        'pros': pros_text,
                        'cons': cons_text,
                        'rating': rating,
                        'original_pros': pros_text,
                        'original_cons': cons_text
                    })
                    if pros_text.strip():
                        if pros_text.strip().lower() in ["нет", "нету"]:
                            negative_aspects["нет"] += 1
                            self.negative_keywords.add("нет")
                        else:
                            processed_texts_set.add(pros_text)
                            pros_sentences = self.analyze_review_sentences(pros_text)
                            for _, sentiment, _, aspects in pros_sentences:
                                if aspects:
                                    for aspect_phrase, aspect_sentiment, _, _ in aspects:
                                        if aspect_sentiment == "положительное":
                                            positive_aspects[aspect_phrase] += 1
                                            self.positive_keywords.add(aspect_phrase)
                                        elif aspect_sentiment == "отрицательное":
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
                                if aspects:
                                    for aspect_phrase, aspect_sentiment, _, _ in aspects:
                                        if aspect_sentiment == "положительное":
                                            positive_aspects[aspect_phrase] += 1
                                            self.positive_keywords.add(aspect_phrase)
                                        elif aspect_sentiment == "отрицательное":
                                            negative_aspects[aspect_phrase] += 1
                                            self.negative_keywords.add(aspect_phrase)
                    if rating >= self.positive_threshold:
                        positive_count += 1
                    elif rating <= self.negative_threshold:
                        negative_count += 1
                    else:
                        neutral_count += 1
            elif all(col in df.columns for col in ['Текст отзыва', 'Оценка']):
                texts = [(str(row['Текст отзыва']), int(row['Оценка']) if pd.notna(row['Оценка']) else 3,
                         str(row.get('Имя пользователя', 'Аноним')))
                        for _, row in df.iterrows()]
                for text, rating, username in texts:
                    self.review_data.append({
                        'username': username,
                        'pros': text,
                        'cons': '',
                        'rating': rating,
                        'original_pros': text,
                        'original_cons': ''
                    })
                    if text.strip():
                        processed_texts_set.add(text)
                        sentences = self.analyze_review_sentences(text)
                        for _, sentiment, _, aspects in sentences:
                            if aspects:
                                for aspect_phrase, aspect_sentiment, _, _ in aspects:
                                    if aspect_sentiment == "положительное":
                                        positive_aspects[aspect_phrase] += 1
                                        self.positive_keywords.add(aspect_phrase)
                                    elif aspect_sentiment == "отрицательное":
                                        negative_aspects[aspect_phrase] += 1
                                        self.negative_keywords.add(aspect_phrase)
                    if rating >= self.positive_threshold:
                        positive_count += 1
                    elif rating <= self.negative_threshold:
                        negative_count += 1
                    else:
                        neutral_count += 1
            else:
                raise ValueError("CSV-файл должен содержать либо столбцы 'Достоинства', 'Недостатки' и 'Оценка', либо 'Текст отзыва' и 'Оценка'")

            # Фильтрация неинформативных аспектов
            for aspect in list(positive_aspects.keys()):
                if aspect.lower() in ["нет недостатков", "нет"]:
                    del positive_aspects[aspect]
                    self.positive_keywords.discard(aspect)
            for aspect in list(negative_aspects.keys()):
                if aspect.lower() in ["нет", "нету"]:
                    del negative_aspects[aspect]
                    self.negative_keywords.discard(aspect)

            # Удаление дублирующихся аспектов
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
                "Отрицательные отзывы": str(negative_count),
                "Нейтральные отзывы": str(neutral_count),
                "positive_aspects": positive_aspects,
                "negative_aspects": negative_aspects
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
        except Exception as e:
            logger.error(f"Ошибка при анализе файла {csv_path}: {str(e)}")
            return {
                "Плюсы": "Ошибка",
                "Минусы": "Ошибка",
                "Ключевые слова (положительные)": "Ошибка",
                "Ключевые слова (отрицательные)": "Ошибка",
                "Общие ключевые слова": "Ошибка",
                "Положительные отзывы": "Ошибка",
                "Отрицательные отзывы": "Ошибка",
                "Нейтральные отзывы": "Ошибка",
                "positive_aspects": Counter(),
                "negative_aspects": Counter()
            }

    def aggregate_reviews(self, csv_paths: List[str]) -> Dict[str, Union[str, Counter]]:
        all_keywords: List[str] = []
        total_positive_count = total_negative_count = total_neutral_count = 0
        all_positive_aspects = Counter()
        all_negative_aspects = Counter()

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(self.analyze_reviews, csv_paths))

        for result in results:
            if result["Плюсы"] not in ["Ошибка", "Ошибка кодировки"]:
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

        for review in self.review_data:
            pros_text = review['original_pros'].lower() if review['original_pros'] else ''
            cons_text = review['original_cons'].lower() if review['original_cons'] else ''
            review_text = pros_text + ' ' + cons_text

            if not all(word in review_text for word in aspect_words):
                continue

            sentiment, _ = self.analyze_sentiment_transformers(review_text)
            example_text = review['original_pros'] if pros_text else review['original_cons']
            example = f'"{example_text}" ({review["username"]})'

            if sentiment == "положительное":
                positive_examples.append(example)
            elif sentiment == "отрицательное":
                negative_examples.append(example)

        if aspect_sentiment == "положительное":
            examples.extend(positive_examples[:max_examples])
            if len(examples) < max_examples and negative_examples:
                examples.append(negative_examples[0])
        elif aspect_sentiment == "отрицательное":
            examples.extend(negative_examples[:max_examples])
            if len(examples) < max_examples and positive_examples:
                examples.append(positive_examples[0])
        return examples[:max_examples]

    def generate_detailed_report(self, analysis_result: Dict[str, Union[str, Counter]], product_name: str = "продукт") -> str:
        positive_aspects = analysis_result["positive_aspects"]
        negative_aspects = analysis_result["negative_aspects"]
        positive_count = int(analysis_result["Положительные отзывы"])
        negative_count = int(analysis_result["Отрицательные отзывы"])
        neutral_count = int(analysis_result["Нейтральные отзывы"])
        total_reviews = positive_count + negative_count + neutral_count

        current_time = datetime.now().strftime("%d.%m.%Y %H:%M")
        report = f"Отчёт по анализу отзывов\nАвтор: Никита Челышев\nДата: {current_time}\nПродукт: {product_name}\n\n"
        report += "=" * 50 + "\n\n"

        report += "1. Основные преимущества\n"
        report += "-" * 30 + "\n"
        if not positive_aspects:
            report += "Положительные аспекты отсутствуют.\n\n"
        else:
            for aspect, count in positive_aspects.most_common(5):
                examples = self.find_representative_examples(aspect, "положительное")
                formatted_aspect = ' '.join(word.capitalize() for word in aspect.split())
                report += f"- {formatted_aspect} ({count} упоминаний)\n"
                report += f"  Примеры: {', '.join(examples) if examples else 'Примеры не найдены.'}\n\n"

        report += "2. Основные недостатки\n"
        report += "-" * 30 + "\n"
        if not negative_aspects:
            report += "Отрицательные аспекты отсутствуют.\n\n"
        else:
            for aspect, count in negative_aspects.most_common(5):
                examples = self.find_representative_examples(aspect, "отрицательное")
                formatted_aspect = ' '.join(word.capitalize() for word in aspect.split())
                report += f"- {formatted_aspect} ({count} упоминаний)\n"
                report += f"  Примеры: {', '.join(examples) if examples else 'Примеры не найдены.'}\n\n"

        report += "3. Ключевые слова\n"
        report += "-" * 30 + "\n"
        report += f"Положительные: {', '.join(sorted(self.positive_keywords)[:10])}\n"
        report += f"Отрицательные: {', '.join(sorted(self.negative_keywords)[:10])}\n"
        report += f"Общие: {analysis_result['Общие ключевые слова']}\n\n"

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

        report += "5. Общее впечатление\n"
        report += "-" * 30 + "\n"
        top_positive_aspects = [aspect for aspect, _ in positive_aspects.most_common(3)]
        top_negative_aspects = [aspect for aspect, _ in negative_aspects.most_common(3)]
        if positive_percentage > negative_percentage:
            report += f"Пользователи в целом довольны продуктом '{product_name}'.\n"
            report += f"Основные преимущества: {', '.join(top_positive_aspects).lower()}.\n"
            report += f"Положительные отзывы составляют {positive_percentage:.1f}%.\n"
            if top_negative_aspects:
                report += f"Недостатки: {', '.join(top_negative_aspects).lower()} ({negative_percentage:.1f}% отзывов).\n"
        else:
            report += f"Впечатления о продукте '{product_name}' смешанные.\n"
            report += f"Преимущества: {', '.join(top_positive_aspects).lower()} ({positive_percentage:.1f}% отзывов).\n"
            report += f"Недостатки: {', '.join(top_negative_aspects).lower()} ({negative_percentage:.1f}% отзывов).\n"

        report += "\n6. Рекомендации\n"
        report += "-" * 30 + "\n"
        report += f"Средняя оценка продукта: {average_rating:.1f}/5.\n"
        if average_rating >= 4.0:
            report += f"Продукт получает высокую оценку.\n"
        elif average_rating >= 3.0:
            report += f"Продукт воспринимается умеренно.\n"
        else:
            report += f"Продукт имеет значительные проблемы.\n"
        if negative_aspects:
            top_negatives = [aspect for aspect, _ in negative_aspects.most_common(2)]
            report += f"Рекомендуется улучшить: {', '.join(top_negatives).lower()}.\n"

        report += "\n" + "=" * 50 + "\n"

        # Экспорт результатов для визуализации
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

