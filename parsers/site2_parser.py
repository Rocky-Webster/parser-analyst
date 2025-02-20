from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from utils.browser import setup_browser
from utils.file_utils import save_to_csv
from utils.logger import setup_logger
import os
import re

logger = setup_logger()

class Site2Parser:
    def parse(self, link=None, output_dir="data"):
        """Парсинг отзывов со второго сайта (iRecommend), включая переход по страницам."""

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

            output_path = os.path.join(output_dir, save_filename)

            parsed_reviews = []
            page_num = 1  # Счетчик страниц

            while True:
                try:
                    logger.info(f"Обрабатываем страницу {page_num}")

                    # Ожидание загрузки отзывов (проверяем оба возможных контейнера)
                    try:
                        reviews_section = WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.CLASS_NAME, "list-comments"))
                        )
                    except:
                        try:
                            reviews_section = WebDriverWait(driver, 10).until(
                                EC.presence_of_element_located((By.CLASS_NAME, "item-list"))
                            )
                        except:
                            logger.warning("Блок отзывов не найден на странице!")
                            break  # Прекращаем парсинг, если отзывы не найдены

                    reviews = reviews_section.find_elements(By.CLASS_NAME, 'item')

                    for review in reviews:
                        try:
                            # Имя пользователя
                            username = review.find_element(By.CLASS_NAME, 'authorName').text.strip()

                            # Оценка (количество закрашенных звездочек)
                            stars = review.find_elements(By.CLASS_NAME, 'star')
                            rating = sum(1 for star in stars if star.find_elements(By.CLASS_NAME, 'on'))

                            # Дата отзыва
                            date = review.find_element(By.CLASS_NAME, 'created').text.strip()

                            # Текст отзыва (заголовок отзыва)
                            review_title = review.find_element(By.CLASS_NAME, 'reviewTitle').text.strip()

                            parsed_reviews.append({
                                "Имя пользователя": username,
                                "Оценка": rating,
                                "Дата": date,
                                "Текст отзыва": review_title,
                            })
                        except Exception as e:
                            logger.warning(f"Ошибка при обработке отзыва: {e}")
                            continue

                    # Попытка найти кнопку "Следующая страница"
                    next_page_element = driver.find_elements(By.CLASS_NAME, 'pager-next') or driver.find_elements(By.CLASS_NAME, 'pager-last')
                    if next_page_element and next_page_element[0].find_elements(By.TAG_NAME, 'a'):
                        next_page_url = next_page_element[0].find_element(By.TAG_NAME, 'a').get_attribute('href')
                        logger.info(f"Переход на следующую страницу: {next_page_url}")
                        driver.get(next_page_url)
                        WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.CLASS_NAME, "list-comments"))
                        )
                        page_num += 1
                    else:
                        logger.info("Достигнут конец списка страниц.")
                        break

                except Exception as e:
                    logger.warning(f"Ошибка при обработке страницы: {e}")
                    break

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