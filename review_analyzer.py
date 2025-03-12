import pandas as pd
import spacy
from collections import Counter
import logging
from typing import List, Dict, Set, Union
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

# Настройка логирования
logger = logging.getLogger(__name__)

# Инициализация spaCy для русского языка с отключением ненужных компонентов
nlp = spacy.load("ru_core_news_sm", disable=["ner", "lemmatizer"])

# Инициализация pymorphy3 для нормализации слов
morph = pymorphy3.MorphAnalyzer()

# Инициализация модели для анализа тональности
model_name = "cointegrated/rubert-tiny-sentiment-balanced"
sentiment_analyzer = None
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
except Exception as e:
    logger.warning(f"Error loading model {model_name}: {e}. Falling back to dictionary-based sentiment analysis.")

# Расширенный словарь тональности для русского языка
SENTIMENT_DICT = {
    "хороший": "положительное",
    "отличный": "положительное",
    "прекрасный": "положительное",
    "удобный": "положительное",
    "качественный": "положительное",
    "быстрый": "положительное",
    "надежный": "положительное",
    "вкусный": "положительное",
    "красивый": "положительное",
    "приятный": "положительное",
    "замечательный": "положительное",
    "свежий": "положительное",
    "плохой": "отрицательное",
    "ужасный": "отрицательное",
    "дефектный": "отрицательное",
    "неудобный": "отрицательное",
    "медленный": "отрицательное",
    "ненадежный": "отрицательное",
    "невкусный": "отрицательное",
    "сломанный": "отрицательное",
    "грязный": "отрицательное",
    "дорогой": "отрицательное",
    "безвкусный": "отрицательное",
    "прогорклый": "отрицательное",
    "протухший": "отрицательное",
    "трындец": "отрицательное",
    "худший": "отрицательное",
    "высокий": "нейтральное",
    "просто": "нейтральное",
    "развивается": "нейтральное",
    "хранится": "нейтральное",
    "хранении": "нейтральное",
}

class ReviewAnalyzer:
    def __init__(self, positive_threshold: int = 4, negative_threshold: int = 2, use_preprocessing: bool = True):
        self.positive_keywords: Set[str] = set()
        self.negative_keywords: Set[str] = set()
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.use_preprocessing = use_preprocessing
        # Список некорректных или неинформативных фраз
        self.invalid_phrases = {
            "безобидный вкус", "изящный делаться", "вкусный блок", "допорывать",
            "прислать развакуум", "возврат делать отказываться", "косарь на ветер",
            "пахнуть быльгет", "звый раз", "два тык", "дотошный аккумулятор", "шень 224"
        }
        # Кэш для результатов анализа тональности
        self.sentiment_cache = {}

    def preprocess_text(self, text: str) -> str:
        """Исправляет опечатки в тексте, нормализует слова и обрабатывает эмодзи."""
        if not self.use_preprocessing:
            return text
        if speller is None:
            logger.warning("pyaspeller не доступен, предобработка текста отключена.")
            return text
        try:
            # Преобразуем эмодзи в текстовые метки
            emoji_dict = {
                "😊": "положительное_эмоция",
                "🙂": "положительное_эмоция",
                "😍": "положительное_эмоция",
                "😢": "отрицательное_эмоция",
                "😡": "отрицательное_эмоция",
                "😠": "отрицательное_эмоция",
            }
            for emoji, label in emoji_dict.items():
                text = text.replace(emoji, f" {label} ")

            # Исправляем опечатки
            corrected = speller.spelled(text)
            # Дополнительная замена известных опечаток
            corrected = corrected.replace("при хари", "при хранении")
            corrected = corrected.replace("хари", "хранении")
            # Токенизация текста
            doc = nlp(corrected)
            # Нормализация слов
            normalized_tokens = []
            token_cache = {}  # Локальный кэш для нормализации
            for token in doc:
                if token.is_punct or token.is_stop:
                    normalized_tokens.append(token.text)
                    continue
                # Нормализуем только уникальные слова
                if token.text in token_cache:
                    normalized_tokens.append(token_cache[token.text])
                    continue
                parsed_word = morph.parse(token.text)[0]
                normal_form = parsed_word.normal_form
                if not parsed_word.word:
                    logger.debug(f"Пропущено некорректное слово: {token.text}")
                    continue
                token_cache[token.text] = normal_form
                normalized_tokens.append(normal_form)
            return " ".join(normalized_tokens)
        except Exception as e:
            logger.warning(f"Ошибка при исправлении опечаток: {str(e)}")
            return text

    def split_sentences(self, text: str) -> List[str]:
        """Разбивает текст на предложения с помощью spaCy, с предварительной очисткой."""
        text = re.sub(r'[!?.]+\)+|\)+|[:;]-?\)+', '.', text)  # Учитываем смайлики вроде :) или ;-)
        text = re.sub(r'(\.+|\!+|\?+)', r'. ', text)  # Заменяем многократные знаки на одну точку
        text = re.sub(r'\s+', ' ', text).strip()  # Убираем лишние пробелы
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return sentences

    def check_modifiers(self, sentence: str) -> float:
        """Проверяет наличие модификаторов (отрицаний, усилителей, ослабителей) и возвращает корректировочный коэффициент."""
        doc = nlp(sentence)
        sentiment_modifier = 1.0
        negation = False
        intensifier = 1.0

        for token in doc:
            if token.lemma_ in ["не", "нет", "ни"] and token.dep_ in ["neg"]:
                negation = True
            elif token.lemma_ in ["очень", "крайне", "сильно"] and token.dep_ in ["advmod"]:
                intensifier = 1.5
            elif token.lemma_ in ["слегка", "немного"] and token.dep_ in ["advmod"]:
                intensifier = 0.5

        if negation:
            sentiment_modifier = -1 * intensifier
        else:
            sentiment_modifier = intensifier

        return sentiment_modifier

    def analyze_sentiment_transformers(self, text: str) -> tuple[str, float]:
        """Анализирует тональность текста с помощью transformers, учитывая контекст и модификаторы."""
        if not text.strip():
            return 'нейтральное', 0.0

        if text in self.sentiment_cache:
            return self.sentiment_cache[text]

        if sentiment_analyzer:
            try:
                result = sentiment_analyzer(text)[0]
                label = result['label'].lower()
                score = result['score']
                doc = nlp(text)
                token_count = len([token for token in doc if not token.is_punct])
                MIN_CONFIDENCE_THRESHOLD = 0.7 if token_count > 5 else 0.5

                if label == "positive" and score > MIN_CONFIDENCE_THRESHOLD:
                    sentiment = "положительное"
                    base_score = score
                elif label == "negative" and score > MIN_CONFIDENCE_THRESHOLD:
                    sentiment = "отрицательное"
                    base_score = -score
                else:
                    sentiment = "нейтральное"
                    base_score = 0.0

                modifier = self.check_modifiers(text)
                adjusted_score = base_score * modifier
                if adjusted_score > 0:
                    sentiment = "положительное"
                elif adjusted_score < 0:
                    sentiment = "отрицательное"
                else:
                    sentiment = "нейтральное"

                self.sentiment_cache[text] = (sentiment, adjusted_score)
                return sentiment, adjusted_score
            except Exception as e:
                logger.warning(f"Error analyzing sentiment with model: {e}. Falling back to dictionary-based sentiment analysis.")
                return self.fallback_sentiment_analysis(text)
        else:
            return self.fallback_sentiment_analysis(text)

    def fallback_sentiment_analysis(self, text: str) -> tuple[str, float]:
        """Резервный анализ тональности с использованием словаря."""
        doc = nlp(text.lower())
        sentiment_scores = {"положительное": 0, "отрицательное": 0}
        for token in doc:
            sentiment = self.get_sentiment(token.lemma_)
            if sentiment:
                sentiment_scores[sentiment] += 1
        modifier = self.check_modifiers(text)
        if sentiment_scores["положительное"] > sentiment_scores["отрицательное"]:
            return "положительное", 0.5 * modifier
        elif sentiment_scores["отрицательное"] > sentiment_scores["положительное"]:
            return "отрицательное", -0.5 * modifier
        else:
            return "нейтральное", 0.0

    def extract_aspects(self, sentence: str, domain_hints: List[str] = None) -> List[tuple]:
        """Извлекает аспекты с использованием нейросети для проверки тональности."""
        doc = nlp(sentence)
        aspects = []
        invalid_words = {"оченк", "поробовать", "спасного", "заскучатся", "добовство", "хари"}
        invalid_phrases = ["лучше времени", "разом себя", "при хари"]

        domain = "техника" if domain_hints and any(hint in ["экран", "камера", "процессор"] for hint in domain_hints) else "еда" if domain_hints and any(hint in ["рыба", "мясо", "еда", "креветка", "минтай"] for hint in domain_hints) else "общий"

        for token in doc:
            if token.pos_ in ["NOUN", "VERB", "ADJ"] and token.lemma_ not in invalid_words:
                aspect = token.lemma_
                modifiers = []
                negation = False
                for child in token.children:
                    if child.dep_ in ["amod", "compound", "advmod"] and child.lemma_ not in invalid_words:
                        modifiers.append(child.text)
                    if child.lemma_ in ["не", "нет", "ни"] and child.dep_ in ["neg"]:
                        negation = True
                aspect_phrase = " ".join(modifiers + [aspect])
                if any(phrase in aspect_phrase.lower() for phrase in invalid_phrases) or aspect_phrase.lower() in self.invalid_phrases:
                    continue

                sentiment, score = self.analyze_sentiment_transformers(aspect_phrase)
                if negation:
                    if sentiment == "положительное" and any(mod in ["вкусный", "хороший", "замечательный"] for mod in modifiers):
                        sentiment = "отрицательное"
                        score = -score
                    elif sentiment == "отрицательное":
                        sentiment = "положительное"
                        score = -score
                if domain == "еда" and aspect in ["хранится"] and "не" in sentence.lower():
                    sentiment = "отрицательное"
                    score = -abs(score)
                elif any(word in aspect_phrase.lower() for word in ["вкусный", "хороший", "свежий", "замечательный"]) and sentiment != "отрицательное":
                    sentiment = "положительное"
                    score = abs(score)
                elif any(word in aspect_phrase.lower() for word in ["безвкусный", "прогорклый", "протухший", "худший", "трындец"]):
                    sentiment = "отрицательное"
                    score = -abs(score)

                aspects.append((aspect_phrase, sentiment, score))

        return aspects

    def analyze_review_sentences(self, review_text: str, domain_hints: List[str] = None) -> List[tuple]:
        """Анализирует тональность каждого предложения в отзыве и извлекает аспекты."""
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

            clause_sentiments = self.analyze_batch_sentiments(clauses)
            for clause, (sentiment, score) in zip(clauses, clause_sentiments):
                aspects = self.extract_aspects(clause, domain_hints)
                result.append((clause, sentiment, score, aspects))

        return result

    def analyze_batch_sentiments(self, texts: List[str]) -> List[tuple]:
        """Пакетный анализ тональности текстов."""
        if not texts:
            return [('нейтральное', 0.0) for _ in texts]

        results = []
        uncached_texts = []
        uncached_indices = []
        for i, text in enumerate(texts):
            if text in self.sentiment_cache:
                results.append(self.sentiment_cache[text])
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                results.append(None)

        if uncached_texts:
            if sentiment_analyzer:
                try:
                    batch_results = sentiment_analyzer(uncached_texts)
                    for idx, res in zip(uncached_indices, batch_results):
                        label = res['label'].lower()
                        score = res['score']
                        doc = nlp(uncached_texts[idx - uncached_indices[0]])
                        token_count = len([token for token in doc if not token.is_punct])
                        MIN_CONFIDENCE_THRESHOLD = 0.7 if token_count > 5 else 0.5

                        if label == "positive" and score > MIN_CONFIDENCE_THRESHOLD:
                            sentiment = "положительное"
                            base_score = score
                        elif label == "negative" and score > MIN_CONFIDENCE_THRESHOLD:
                            sentiment = "отрицательное"
                            base_score = -score
                        else:
                            sentiment = "нейтральное"
                            base_score = 0.0
                        modifier = self.check_modifiers(uncached_texts[idx - uncached_indices[0]])
                        adjusted_score = base_score * modifier
                        if adjusted_score > 0:
                            sentiment = "положительное"
                        elif adjusted_score < 0:
                            sentiment = "отрицательное"
                        else:
                            sentiment = "нейтральное"
                        results[idx] = (sentiment, adjusted_score)
                        self.sentiment_cache[uncached_texts[idx - uncached_indices[0]]] = (sentiment, adjusted_score)
                except Exception as e:
                    logger.warning(f"Error in batch sentiment analysis: {e}. Falling back to dictionary-based sentiment analysis.")
                    for idx, text in zip(uncached_indices, uncached_texts):
                        sentiment, score = self.fallback_sentiment_analysis(text)
                        results[idx] = (sentiment, score)
                        self.sentiment_cache[text] = (sentiment, score)
            else:
                for idx, text in zip(uncached_indices, uncached_texts):
                    sentiment, score = self.fallback_sentiment_analysis(text)
                    results[idx] = (sentiment, score)
                    self.sentiment_cache[text] = (sentiment, score)

        return results

    def analyze_reviews(self, csv_path: str) -> Dict[str, str]:
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
            site_keywords: List[str] = []
            positive_count = 0
            negative_count = 0
            positive_aspects = Counter()
            negative_aspects = Counter()
            self.positive_keywords.clear()
            self.negative_keywords.clear()
            processed_texts_set = set()

            domain_hints = []
            if all(col in df.columns for col in ['Достоинства', 'Недостатки', 'Оценка']):
                texts = [(str(row['Достоинства']) if pd.notna(row['Достоинства']) else "",
                          str(row['Недостатки']) if pd.notna(row['Недостатки']) else "",
                          int(row['Оценка']) if pd.notna(row['Оценка']) else 3)
                         for _, row in df.iterrows()]
                for pros_text, cons_text, _ in texts:
                    if pros_text.strip() and pros_text.strip().lower() != "нет":
                        doc = nlp(pros_text.lower())
                        for token in doc:
                            if token.lemma_ in ["рыба", "мясо", "еда", "экран", "камера", "креветка", "минтай"]:
                                domain_hints.append(token.lemma_)
                    if cons_text.strip() and cons_text.strip().lower() != "нет":
                        doc = nlp(cons_text.lower())
                        for token in doc:
                            if token.lemma_ in ["рыба", "мясо", "еда", "экран", "камера", "креветка", "минтай"]:
                                domain_hints.append(token.lemma_)
            elif all(col in df.columns for col in ['Текст отзыва', 'Оценка']):
                texts = [str(row['Текст отзыва']) for _, row in df.iterrows()]
                for text in texts:
                    doc = nlp(text.lower())
                    for token in doc:
                        if token.lemma_ in ["рыба", "мясо", "еда", "экран", "камера", "креветка", "минтай"]:
                            domain_hints.append(token.lemma_)

            domain_hints = list(set(domain_hints))

            if all(col in df.columns for col in ['Достоинства', 'Недостатки', 'Оценка']):
                texts = [(str(row['Достоинства']) if pd.notna(row['Достоинства']) else "",
                          str(row['Недостатки']) if pd.notna(row['Недостатки']) else "",
                          int(row['Оценка']) if pd.notna(row['Оценка']) else 3)
                         for _, row in df.iterrows()]
                pros_texts, cons_texts, ratings = zip(*texts)
                pros_processed = self.process_texts(pros_texts, domain_hints)
                cons_processed = self.process_texts(cons_texts, domain_hints)

                for (pros_text, cons_text, rating), pros_keywords, cons_keywords in zip(texts, pros_processed, cons_processed):
                    site_keywords.extend(pros_keywords + cons_keywords)
                    if pros_text.strip() and pros_text.strip().lower() != "нет" and pros_text not in processed_texts_set:
                        processed_texts_set.add(pros_text)
                        pros_sentences = self.analyze_review_sentences(pros_text, domain_hints)
                        pros_overall_sentiment, _ = self.analyze_sentiment_transformers(pros_text)
                        for sentence, sentiment, _, aspects in pros_sentences:
                            if sentiment == "положительное":
                                if aspects:
                                    for aspect_phrase, aspect_sentiment, _ in aspects:
                                        if aspect_sentiment == "положительное":
                                            positive_aspects[aspect_phrase] += 1
                                            self.positive_keywords.add(aspect_phrase)
                                        elif aspect_sentiment == "отрицательное":
                                            negative_aspects[aspect_phrase] += 1
                                            self.negative_keywords.add(aspect_phrase)
                                else:
                                    positive_aspects[sentence] += 1
                                    self.positive_keywords.add(sentence)
                            elif sentiment == "отрицательное":
                                if aspects:
                                    for aspect_phrase, aspect_sentiment, _ in aspects:
                                        if aspect_sentiment == "положительное":
                                            positive_aspects[aspect_phrase] += 1
                                            self.positive_keywords.add(aspect_phrase)
                                        elif aspect_sentiment == "отрицательное":
                                            negative_aspects[aspect_phrase] += 1
                                            self.negative_keywords.add(aspect_phrase)
                                else:
                                    negative_aspects[sentence] += 1
                                    self.negative_keywords.add(sentence)
                        if pros_overall_sentiment == "положительное" and not positive_aspects:
                            positive_aspects[pros_text] += 1
                            self.positive_keywords.add(pros_text)
                        elif pros_overall_sentiment == "отрицательное" and not negative_aspects:
                            negative_aspects[pros_text] += 1
                            self.negative_keywords.add(pros_text)

                    if cons_text.strip() and cons_text.strip().lower() != "нет" and cons_text not in processed_texts_set:
                        processed_texts_set.add(cons_text)
                        cons_sentences = self.analyze_review_sentences(cons_text, domain_hints)
                        cons_overall_sentiment, _ = self.analyze_sentiment_transformers(cons_text)
                        for sentence, sentiment, _, aspects in cons_sentences:
                            if sentiment == "положительное":
                                if aspects:
                                    for aspect_phrase, aspect_sentiment, _ in aspects:
                                        if aspect_sentiment == "положительное":
                                            positive_aspects[aspect_phrase] += 1
                                            self.positive_keywords.add(aspect_phrase)
                                        elif aspect_sentiment == "отрицательное":
                                            negative_aspects[aspect_phrase] += 1
                                            self.negative_keywords.add(aspect_phrase)
                                else:
                                    positive_aspects[sentence] += 1
                                    self.positive_keywords.add(sentence)
                            elif sentiment == "отрицательное":
                                if aspects:
                                    for aspect_phrase, aspect_sentiment, _ in aspects:
                                        if aspect_sentiment == "положительное":
                                            positive_aspects[aspect_phrase] += 1
                                            self.positive_keywords.add(aspect_phrase)
                                        elif aspect_sentiment == "отрицательное":
                                            negative_aspects[aspect_phrase] += 1
                                            self.negative_keywords.add(aspect_phrase)
                                else:
                                    negative_aspects[sentence] += 1
                                    self.negative_keywords.add(sentence)
                        if cons_overall_sentiment == "положительное" and not positive_aspects:
                            positive_aspects[cons_text] += 1
                            self.positive_keywords.add(cons_text)
                        elif cons_overall_sentiment == "отрицательное" and not negative_aspects:
                            negative_aspects[cons_text] += 1
                            self.negative_keywords.add(cons_text)

                    if rating >= self.positive_threshold:
                        positive_count += 1
                    elif rating <= self.negative_threshold:
                        negative_count += 1

            elif all(col in df.columns for col in ['Текст отзыва', 'Оценка']):
                texts = [str(row['Текст отзыва']) for _, row in df.iterrows()]
                ratings = [int(row['Оценка']) if pd.notna(row['Оценка']) else 3 for _, row in df.iterrows()]
                processed_texts = self.process_texts(texts, domain_hints)

                for text, rating, keywords in zip(texts, ratings, processed_texts):
                    site_keywords.extend(keywords)
                    if text not in processed_texts_set:
                        processed_texts_set.add(text)
                        sentences = self.analyze_review_sentences(text, domain_hints)
                        text_overall_sentiment, _ = self.analyze_sentiment_transformers(text)
                        for sentence, sentiment, _, aspects in sentences:
                            if sentiment == "положительное":
                                if aspects:
                                    for aspect_phrase, aspect_sentiment, _ in aspects:
                                        if aspect_sentiment == "положительное":
                                            positive_aspects[aspect_phrase] += 1
                                            self.positive_keywords.add(aspect_phrase)
                                        elif aspect_sentiment == "отрицательное":
                                            negative_aspects[aspect_phrase] += 1
                                            self.negative_keywords.add(aspect_phrase)
                                else:
                                    positive_aspects[sentence] += 1
                                    self.positive_keywords.add(sentence)
                            elif sentiment == "отрицательное":
                                if aspects:
                                    for aspect_phrase, aspect_sentiment, _ in aspects:
                                        if aspect_sentiment == "положительное":
                                            positive_aspects[aspect_phrase] += 1
                                            self.positive_keywords.add(aspect_phrase)
                                        elif aspect_sentiment == "отрицательное":
                                            negative_aspects[aspect_phrase] += 1
                                            self.negative_keywords.add(aspect_phrase)
                                else:
                                    negative_aspects[sentence] += 1
                                    self.negative_keywords.add(sentence)
                        if text_overall_sentiment == "положительное" and not positive_aspects:
                            positive_aspects[text] += 1
                            self.positive_keywords.add(text)
                        elif text_overall_sentiment == "отрицательное" and not negative_aspects:
                            negative_aspects[text] += 1
                            self.negative_keywords.add(text)
                    if rating >= self.positive_threshold:
                        positive_count += 1
                    elif rating <= self.negative_threshold:
                        negative_count += 1

            else:
                raise ValueError("CSV-файл должен содержать либо столбцы 'Достоинства', 'Недостатки' и 'Оценка', либо 'Текст отзыва' и 'Оценка'")

            # Удаляем пересечения между положительными и отрицательными ключевыми словами
            common_keywords = self.positive_keywords.intersection(self.negative_keywords)
            for keyword in common_keywords:
                sentiment = self.get_sentiment(keyword)
                if sentiment == "положительное":
                    self.negative_keywords.discard(keyword)
                elif sentiment == "отрицательное":
                    self.positive_keywords.discard(keyword)
                else:
                    pos_count = sum(1 for aspect, count in positive_aspects.items() if keyword in aspect for _ in range(count))
                    neg_count = sum(1 for aspect, count in negative_aspects.items() if keyword in aspect for _ in range(count))
                    if pos_count > neg_count:
                        self.negative_keywords.discard(keyword)
                    else:
                        self.positive_keywords.discard(keyword)

            main_positives = "\n".join([f"{aspect} (положительное) ({count})" for aspect, count in positive_aspects.most_common(5)]) if positive_aspects else "Нет положительных отзывов"
            main_negatives = "\n".join([f"{aspect} (отрицательное) ({count})" for aspect, count in negative_aspects.most_common(5)]) if negative_aspects else "Нет отрицательных отзывов"
            top_keywords = self.get_top_keywords(site_keywords, 15)

            return {
                "Плюсы": main_positives,
                "Минусы": main_negatives,
                "Ключевые слова (положительные)": ", ".join(sorted(self.positive_keywords)[:10]),
                "Ключевые слова (отрицательные)": ", ".join(sorted(self.negative_keywords)[:10]),
                "Общие ключевые слова": ", ".join(top_keywords.split(", ")),
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
        """Агрегированный анализ отзывов из всех CSV-файлов с использованием многопоточности."""
        all_keywords: List[str] = []
        total_positive_count = 0
        total_negative_count = 0
        all_positive_aspects = Counter()
        all_negative_aspects = Counter()

        def process_single_file(csv_path):
            result = self.analyze_reviews(csv_path)
            keywords = result["Общие ключевые слова"].split(", ")
            positive_count = int(result["Положительные отзывы"]) if result["Положительные отзывы"].isdigit() else 0
            negative_count = int(result["Отрицательные отзывы"]) if result["Отрицательные отзывы"].isdigit() else 0
            return keywords, positive_count, negative_count

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_single_file, csv_paths))

        for keywords, positive_count, negative_count in results:
            all_keywords.extend(keywords)
            total_positive_count += positive_count
            total_negative_count += negative_count

        for csv_path in csv_paths:
            df = pd.read_csv(csv_path, encoding='utf-8')
            if all(col in df.columns for col in ['Достоинства', 'Недостатки', 'Оценка']):
                texts = [(str(row['Достоинства']) if pd.notna(row['Достоинства']) else "",
                          str(row['Недостатки']) if pd.notna(row['Недостатки']) else "",
                          int(row['Оценка']) if pd.notna(row['Оценка']) else 3)
                         for _, row in df.iterrows()]
                for pros_text, cons_text, _ in texts:
                    if pros_text.strip() and pros_text.strip().lower() != "нет":
                        pros_sentences = self.analyze_review_sentences(pros_text)
                        for sentence, sentiment, _, aspects in pros_sentences:
                            if sentiment == "положительное":
                                for aspect_phrase, aspect_sentiment, _ in aspects:
                                    if aspect_sentiment == "положительное":
                                        all_positive_aspects[aspect_phrase] += 1
                                        self.positive_keywords.add(aspect_phrase)
                            elif sentiment == "отрицательное":
                                for aspect_phrase, aspect_sentiment, _ in aspects:
                                    if aspect_sentiment == "отрицательное":
                                        all_negative_aspects[aspect_phrase] += 1
                                        self.negative_keywords.add(aspect_phrase)
                    if cons_text.strip() and cons_text.strip().lower() != "нет":
                        cons_sentences = self.analyze_review_sentences(cons_text)
                        for sentence, sentiment, _, aspects in cons_sentences:
                            if sentiment == "положительное":
                                for aspect_phrase, aspect_sentiment, _ in aspects:
                                    if aspect_sentiment == "положительное":
                                        all_positive_aspects[aspect_phrase] += 1
                                        self.positive_keywords.add(aspect_phrase)
                            elif sentiment == "отрицательное":
                                for aspect_phrase, aspect_sentiment, _ in aspects:
                                    if aspect_sentiment == "отрицательное":
                                        all_negative_aspects[aspect_phrase] += 1
                                        self.negative_keywords.add(aspect_phrase)

            elif all(col in df.columns for col in ['Текст отзыва', 'Оценка']):
                texts = [str(row['Текст отзыва']) for _, row in df.iterrows()]
                for text in texts:
                    sentences = self.analyze_review_sentences(text)
                    for sentence, sentiment, _, aspects in sentences:
                        if sentiment == "положительное":
                            for aspect_phrase, aspect_sentiment, _ in aspects:
                                if aspect_sentiment == "положительное":
                                    all_positive_aspects[aspect_phrase] += 1
                                    self.positive_keywords.add(aspect_phrase)
                        elif sentiment == "отрицательное":
                            for aspect_phrase, aspect_sentiment, _ in aspects:
                                if aspect_sentiment == "отрицательное":
                                    all_negative_aspects[aspect_phrase] += 1
                                    self.negative_keywords.add(aspect_phrase)

        main_positives = "\n".join([f"{aspect} (положительное) ({count})" for aspect, count in all_positive_aspects.most_common(5)]) if all_positive_aspects else "Нет положительных отзывов"
        main_negatives = "\n".join([f"{aspect} (отрицательное) ({count})" for aspect, count in all_negative_aspects.most_common(5)]) if all_negative_aspects else "Нет отрицательных отзывов"
        top_keywords = self.get_top_keywords(all_keywords, 15)

        return {
            "Плюсы (все сайты)": main_positives,
            "Минусы (все сайты)": main_negatives,
            "Ключевые слова (положительные, все сайты)": ", ".join(sorted(self.positive_keywords)[:10]),
            "Ключевые слова (отрицательные, все сайты)": ", ".join(sorted(self.negative_keywords)[:10]),
            "Общие ключевые слова (все сайты)": ", ".join(top_keywords.split(", ")),
            "Положительные отзывы (все сайты)": str(total_positive_count),
            "Отрицательные отзывы (все сайты)": str(total_negative_count)
        }

    def process_texts(self, texts: List[str], domain_hints: List[str] = None) -> List[List[str]]:
        """Пакетная обработка текстов с помощью spaCy с фильтрацией шумных слов и учётом домена."""
        try:
            keywords_list = []
            cyrillic_pattern = re.compile(r'^[а-яА-ЯёЁ]+$')
            texts = [self.preprocess_text(text.lower()) for text in texts]

            domain = "техника" if domain_hints and any(hint in ["экран", "камера", "процессор"] for hint in domain_hints) else "еда" if domain_hints and any(hint in ["рыба", "мясо", "еда"] for hint in domain_hints) else "общий"
            
            tech_words = {"экран", "камера", "процессор", "тормозит", "глючный", "яркий"}
            food_words = {"вкусный", "свежий", "прогорклый", "хранится", "пахнет", "минтай", "креветка"}

            action_verbs = {"похвалить", "купить", "приготовить", "съесть"}

            for doc in nlp.pipe(texts, disable=["ner"]):
                keywords = []
                for token in doc:
                    lemma = token.lemma_
                    if (not token.is_stop and not token.is_punct and
                        token.pos_ in ["NOUN", "ADJ", "ADV"] and
                        len(lemma) >= 3 and
                        cyrillic_pattern.match(lemma)):
                        if lemma in action_verbs:
                            continue
                        if domain == "еда" and lemma in tech_words:
                            continue
                        if domain == "техника" and lemma in food_words:
                            continue
                        keywords.append(lemma)
                    elif token.pos_ == "VERB" and lemma in food_words and domain == "еда":
                        keywords.append(lemma)
                keywords_list.append(keywords)
            return keywords_list
        except Exception as e:
            logger.warning(f"Ошибка при пакетной обработке текстов: {str(e)}")
            return [[] for _ in texts]

    def get_sentiment(self, word: str) -> str:
        """Определяет тональность слова с помощью словаря SENTIMENT_DICT."""
        return SENTIMENT_DICT.get(word, "")

    def summarize_reviews(self, reviews: List[Dict[str, Union[str, List[str]]]]) -> str:
        """Суммирует отзывы с учётом тональности и значимости."""
        try:
            if not reviews:
                return "Нет данных для суммирования"
            all_keywords = [keyword for review in reviews for keyword in review["keywords"]]
            keyword_counts = Counter(all_keywords)
            top_keywords = [(word, count) for word, count in keyword_counts.most_common(10) if count >= 1]

            summary = []
            for word, count in top_keywords:
                sentiment = self.get_sentiment(word)
                sentiment_label = f" ({sentiment})" if sentiment else ""
                summary.append(f"{word}{sentiment_label} ({count})")
            return "\n".join(summary) if summary else "Нет ключевых слов для суммирования"
        except Exception as e:
            logger.warning(f"Ошибка при суммировании отзывов: {str(e)}")
            return "Ошибка при суммировании"

    def get_top_keywords(self, keywords: List[str], n: int = 15) -> str:
        """Возвращает топ-n ключевых слов по частоте."""
        try:
            if not keywords:
                return "Нет ключевых слов"
            keyword_counts = Counter(keywords)
            top_keywords = [word for word, count in keyword_counts.most_common(n) if count >= 1]
            return ", ".join(top_keywords) if top_keywords else "Нет ключевых слов"
        except Exception as e:
            logger.warning(f"Ошибка при получении топ-ключевых слов: {str(e)}")
            return "Ошибка"
