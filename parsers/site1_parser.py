from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from utils.browser import setup_browser
from utils.file_utils import save_to_csv
from utils.logger import setup_logger
from config import SITE1_DATA_PATH
import os
import re

logger = setup_logger()

class Site1Parser:
    def parse(self, link=None, output_dir=None):
        """Парсинг отзывов с первого сайта (Отзовик) через Selenium с перелистыванием страниц."""

        if not link:
            logger.error("Ссылка для парсинга не передана!")
            return
        
        logger.info(f"Начинаем парсинг отзывов с сайта Отзовик: {link}")
        driver = None
        try:
            driver = setup_browser()
            driver.get(link)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'item'))
            )

            # Извлекаем название товара
            try:
                product_name_elem = driver.find_element(By.CSS_SELECTOR, 'span.fn[itemprop="name"]')
                product_name = product_name_elem.text.strip()
                safe_product_name = re.sub(r'[<>:"/\\|?*]+', '', product_name).replace(' ', '_')
                save_filename = f"{safe_product_name}_otzovik.csv"
            except Exception:
                save_filename = "otzovik_reviews.csv"

            save_path = os.path.join(output_dir if output_dir else SITE1_DATA_PATH, save_filename)

            parsed_reviews = []

            while True:
                try:
                    reviews = driver.find_elements(By.CLASS_NAME, 'item')
                    
                    for review in reviews:
                        try:
                            username = review.find_element(By.CSS_SELECTOR, 'span[itemprop="name"]').text.strip()
                            
                            # Репутация необязательна
                            reputation = ""
                            try:
                                reputation = review.find_element(By.CLASS_NAME, 'karma-line').text.strip()
                            except Exception:
                                pass
                            
                            date = review.find_element(By.CLASS_NAME, 'review-postdate').text.strip()
                            rating = review.find_element(By.CLASS_NAME, 'rating-score').find_element(By.TAG_NAME, 'span').text.strip()
                            pros = review.find_element(By.CLASS_NAME, 'review-plus').text.replace('Достоинства:', '').strip()
                            cons = review.find_element(By.CLASS_NAME, 'review-minus').text.replace('Недостатки:', '').strip()

                            parsed_reviews.append({
                                "Имя пользователя": username,
                                "Репутация": reputation,
                                "Дата": date,
                                "Оценка": rating,
                                "Достоинства": pros,
                                "Недостатки": cons,
                            })
                        except Exception as e:
                            logger.warning(f"Ошибка при обработке отзыва: {e}")

                    next_page_element = driver.find_elements(By.CLASS_NAME, 'next')
                    if next_page_element and next_page_element[0].get_attribute('href'):
                        next_page_url = next_page_element[0].get_attribute('href')
                        logger.info(f"Переход на следующую страницу: {next_page_url}")
                        driver.get(next_page_url)
                        WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.CLASS_NAME, 'item'))
                        )
                    else:
                        logger.info("Достигнут конец списка страниц.")
                        break

                except Exception as e:
                    logger.warning(f"Ошибка при обработке страницы: {e}")
                    break

            if parsed_reviews:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                save_to_csv(parsed_reviews, save_path)
                logger.info(f"Сохранено {len(parsed_reviews)} отзывов в {save_path}")
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
    parser = Site1Parser()
    parser.parse()