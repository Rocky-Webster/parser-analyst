from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from utils.browser import setup_browser
from utils.file_utils import save_to_csv
from utils.logger import setup_logger
import os
import re
import spacy
from collections import Counter

logger = setup_logger()

# Загружаем модель spaCy для русского языка
nlp = spacy.load("ru_core_news_sm")

class Site2Parser:
    def parse(self, link=None, output_dir="data"):
        """Парсинг отзывов со второго сайта (iRecommend), включая переход по страницам и выделение ключевых слов."""

        if not link:
            logger.error("Ссылка для парсинга не передана!")
            return

        logger.info(f"Начинаем парсинг отзывов с сайта iRecommend: {link}")
        driver = None
        try:
            driver = setup_browser()
            driver.get(link)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "list-comments"))
            )

            # Извлекаем название товара
            try:
                product_name_elem = driver.find_element(By.CSS_SELECTOR, 'span.fn[itemprop="name"]')
                product_name = product_name_elem.text.strip()
                safe_product_name = re.sub(r'[<>:"/\\|?*]+', '', product_name).replace(' ', '_')
                save_filename = f"{safe_product_name}_irecommend.csv"
            except Exception:
                save_filename = "irecommend_reviews.csv"

            output_path = os.path.join(output_dir, save_filename).replace("\\", "/")

            parsed_reviews = []

            # Сначала парсим начальную страницу
            page_urls = [link]

            # Собираем ссылки пагинации
            try:
                pager = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "ul.pager"))
                )
                page_links = pager.find_elements(By.TAG_NAME, "a")
                page_urls.extend([link.get_attribute("href") for link in page_links])
            except Exception as e:
                logger.warning(f"Не удалось найти пагинацию: {e}")
                page_urls = [link]

            # Обходим все страницы
            page_num = 1
            for page_url in page_urls:
                logger.info(f"Обрабатываем страницу {page_num}")
                driver.get(page_url)
                
                # Ожидание загрузки отзывов и прокрутка
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                try:
                    reviews_section = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "list-comments"))
                    )
                except:
                    logger.warning("Блок отзывов не найден на странице!")
                    break

                reviews = reviews_section.find_elements(By.CLASS_NAME, 'item')
                logger.info(f"Найдено {len(reviews)} отзывов на странице {page_num}")

                for review in reviews:
                    try:
                        username = review.find_element(By.CLASS_NAME, 'authorName').text.strip()
                        stars = review.find_elements(By.CLASS_NAME, 'star')
                        rating = sum(1 for star in stars if star.find_elements(By.CLASS_NAME, 'on'))
                        date = review.find_element(By.CLASS_NAME, 'created').text.strip()
                        review_title = review.find_element(By.CLASS_NAME, 'reviewTitle').text.strip()

                        # Выделение ключевых слов с помощью spaCy
                        doc = nlp(review_title.lower())  # Приводим текст к нижнему регистру
                        keywords = []
                        for token in doc:
                            if not token.is_stop and not token.is_punct and token.pos_ in ["NOUN", "VERB", "ADJ"]:  # Существительные, глаголы, прилагательные
                                keywords.append(token.lemma_)  # Используем лемму для нормализации

                        # Берем 3 наиболее частых ключевых слова (или меньше, если их меньше)
                        keyword_counts = Counter(keywords)
                        top_keywords = [word for word, count in keyword_counts.most_common(3) if count > 1]  # Только слова, встречающиеся > 1 раза
                        keywords_str = ", ".join(top_keywords) if top_keywords else "Нет ключевых слов"

                        parsed_reviews.append({
                            "Имя пользователя": username,
                            "Оценка": rating,
                            "Дата": date,
                            "Текст отзыва": review_title,
                            "Ключевые слова": keywords_str  # Выделенные ключевые слова
                        })
                    except Exception as e:
                        logger.warning(f"Ошибка при обработке отзыва: {e}")
                        continue

                page_num += 1

            # Сохранение данных
            if parsed_reviews:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                save_to_csv(parsed_reviews, output_path)
                logger.info(f"Сохранено {len(parsed_reviews)} отзывов в {output_path}")
            else:
                logger.warning("Нет данных для сохранения!")
            return parsed_reviews

        except Exception as e:
            logger.error(f"Ошибка при парсинге: {e}")
            return []
        finally:
            if driver:
                driver.quit()

if __name__ == "__main__":
    parser = Site2Parser()
    parser.parse()