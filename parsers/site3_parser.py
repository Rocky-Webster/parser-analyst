from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from utils.browser import setup_browser
from utils.file_utils import save_to_csv
from utils.logger import setup_logger
from config import SITE3_DATA_PATH
import logging
import urllib.parse
import os
import re
from selenium.common.exceptions import TimeoutException, NoSuchElementException

logger = setup_logger()

class Site3Parser:
    def parse(self, link=None, output_dir=None):
        if not link:
            logger.error("Ссылка для парсинга не передана!")
            return
        
        if "ozon.ru" not in link or "/reviews/" not in link:
            logger.error("Введена некорректная ссылка на страницу отзывов Ozon!")
            return

        driver = None
        try:
            driver = setup_browser()
            driver.get(link)
            logger.info(f"Начинаем парсинг отзывов с сайта Озон: {link}")

            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, 'x4q_30'))
            )

            try:
                product_name_elem = driver.find_element(By.CLASS_NAME, 'l2v_27')
                product_name = product_name_elem.text.strip()
                safe_product_name = re.sub(r'[<>:"/\\|?*]+', '', product_name).replace(' ', '_')
                save_filename = f"{safe_product_name}_ozon.csv"
            except Exception:
                save_filename = "ozon_reviews.csv"

            save_path = os.path.join(output_dir if output_dir else SITE3_DATA_PATH, save_filename)

            parsed_reviews = []
            seen_reviews = set()
            base_url = link
            max_pages = 5
            last_page_key = None

            for page in range(1, max_pages + 1):
                if page == 1:
                    current_url = base_url
                else:
                    parsed_base_url = urllib.parse.urlparse(base_url)
                    query_params = urllib.parse.parse_qs(parsed_base_url.query)
                    query_params['page'] = [str(page)]
                    query_params['sort'] = ['published_at_desc']
                    if last_page_key:
                        query_params['page_key'] = [last_page_key]
                    current_url = f"{parsed_base_url.scheme}://{parsed_base_url.netloc}{parsed_base_url.path}?{urllib.parse.urlencode(query_params, doseq=True)}"
                
                logger.info(f"Парсим страницу {page}")
                driver.get(current_url)
                try:
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_all_elements_located((By.CLASS_NAME, 'x4q_30'))
                    )
                except TimeoutException as e:
                    logger.error(f"Тайм-аут при ожидании отзывов на странице {page}: {e}")
                    break

                reviews = driver.find_elements(By.CLASS_NAME, 'x4q_30')
                logger.info(f"Найдено отзывов: {len(reviews)}")
                
                if not reviews:
                    break
                
                for review in reviews:
                    try:
                        username_elem = review.find_element(By.CLASS_NAME, 'pp3_30')
                        WebDriverWait(driver, 5).until(lambda d: username_elem.text.strip() != "")
                        username = username_elem.text.strip() or username_elem.get_attribute('innerText').strip()

                        date_elem = review.find_element(By.CLASS_NAME, 'ps4_30')
                        WebDriverWait(driver, 5).until(lambda d: date_elem.text.strip() != "")
                        date = date_elem.text.strip() or date_elem.get_attribute('innerText').strip()

                        review_text = ""
                        try:
                            text_container = review.find_element(By.CLASS_NAME, 'p5s_30')
                            review_text_elem = text_container.find_element(By.CLASS_NAME, 'sp5_30')
                            WebDriverWait(driver, 5).until(lambda d: review_text_elem.text.strip() != "")
                            review_text = review_text_elem.text.strip() or review_text_elem.get_attribute('innerText').strip()
                        except Exception:
                            pass  # Если текста нет, оставляем пустую строку

                        rating_container_parent = review.find_element(By.CLASS_NAME, 's3p_30')
                        rating_container = rating_container_parent.find_element(By.CLASS_NAME, 'a5d24-a')
                        stars = rating_container.find_elements(By.TAG_NAME, 'svg')
                        rating = sum(1 for star in stars if 'rgb(255, 168, 0)' in star.get_attribute('style'))

                        review_key = (username, date, review_text, rating)
                        if review_key not in seen_reviews:
                            seen_reviews.add(review_key)
                            parsed_reviews.append({
                                "Имя пользователя": username,
                                "Дата": date,
                                "Оценка": rating,
                                "Текст отзыва": review_text,
                            })
                    except Exception as e:
                        logger.warning(f"Ошибка при обработке отзыва: {e}")

                try:
                    next_button = driver.find_element(By.CLASS_NAME, 'qp7_30')
                    next_url = next_button.get_attribute('href')
                    if not next_url:
                        break
                    parsed_url = urllib.parse.urlparse(next_url)
                    query_params = urllib.parse.parse_qs(parsed_url.query)
                    page_key = query_params.get('page_key', [''])[0]
                    if page_key:
                        last_page_key = page_key
                except NoSuchElementException:
                    break
                except Exception as e:
                    logger.warning(f"Не удалось извлечь page_key: {e}")
                    break

            if parsed_reviews:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                save_to_csv(parsed_reviews, save_path)
                logger.info(f"Сохранено {len(parsed_reviews)} отзывов в {save_path}")
            else:
                logger.warning("Нет данных для сохранения!")

            return parsed_reviews

        except Exception as e:
            logger.error(f"Ошибка при парсинге: {str(e)}", exc_info=True)
            return []
        finally:
            if driver:
                driver.quit()

if __name__ == "__main__":
    parser = Site3Parser()
    parser.parse()