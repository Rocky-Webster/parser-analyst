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
import time
from selenium.common.exceptions import TimeoutException, NoSuchElementException

logger = setup_logger()

class Site3Parser:
    def __init__(self):
        self.selectors = {
            'reviews_tab': "//div[contains(text(), 'ОТЗЫВЫ') or contains(text(), 'Отзывы') or contains(@href, '/reviews/')]",
            'product_name': (
                "//a[contains(@class, 'n0m') and contains(@href, '/product/')] | "
                "//div[contains(@class, 'nm0')]//a[contains(@href, '/product/')] | "
                "//h1 | "
                "//h2[contains(@class, 'title')] | "
                "//div[contains(@class, 'title')]//h1 | "
                "//div[contains(@class, 'title')]//h2 | "
                "//div[contains(@class, 'product')]//h1 | "
                "//div[contains(@class, 'product')]//h2 | "
                "//span[contains(@itemprop, 'name') and not(ancestor::div[@data-review-uuid])] | "
                "//div[contains(@class, 'breadcrumb')]//span[last()] | "
                "//div[contains(@class, 'header')]//h1"
            ),
            'review_container': (
                "//div[@data-review-uuid] | "
                "//div[contains(@data-widget, 'webReviewItem')] | "
                "//div[contains(@data-widget, 'review')] | "
                "//div[contains(@class, 'review') and .//div[contains(text(), 'Комментарий') or .//svg]]"
            ),
            'username': ".//div[contains(@class, 'pu8_31')]//span[contains(@class, 'p8u_31')]",
            'date': ".//div[contains(@class, 'x5p_31')]",
            'review_text': ".//span[contains(@class, 'p7x_31')]",
            'rating_stars': ".//div[contains(@class, 'p6x_31')]//svg",  # Ищем все звёзды
            'pagination': (
                "//div[contains(@class, 'vp8_31') or contains(@class, 'pw6') or contains(@class, 'pagination') or contains(@data-widget, 'webPagination')]//a[contains(@href, '/reviews/') and contains(@href, 'page=')] | "
                "//a[.//div[contains(text(), 'Дальше')]] | "
                "//a[contains(text(), 'Дальше')] | "
                "//a[contains(text(), 'след')] | "
                "//a[contains(text(), 'Вперёд')] | "
                "//a[contains(@aria-label, 'Next')] | "
                "//a[contains(@aria-label, 'Следующая')] | "
                "//div[contains(@class, 'vp8_31') or contains(@class, 'pw6')]//a[not(contains(@class, 'v7p_31')) and contains(@href, '/reviews/')] | "
                "//a[./*[local-name()='svg'] and contains(@href, '/reviews/')] | "
                "//a[contains(@data-page, 'next')] | "
                "//button[contains(text(), 'Дальше')] | "
                "//button[contains(text(), 'след')] | "
                "//button[contains(@aria-label, 'Next')]"
            )
        }
        self.max_pages = 5

    def find_element_safely(self, parent, selector, by=By.XPATH, timeout=5):
        try:
            if by == By.XPATH:
                elements = parent.find_elements(by, selector)
                if elements:
                    return elements[0]
            else:
                elements = parent.find_elements(by, selector)
                if elements:
                    return elements[0]
            logger.debug(f"Элемент не найден по селектору: {selector}")
            return None
        except Exception as e:
            logger.debug(f"Ошибка при поиске элемента с селектором {selector}: {e}")
            return None

    def find_elements_safely(self, parent, selector, by=By.XPATH):
        try:
            elements = parent.find_elements(by, selector)
            logger.debug(f"Найдено {len(elements)} элементов с селектором {selector}")
            return elements
        except Exception as e:
            logger.debug(f"Не удалось найти элементы с селектором {selector}: {e}")
            return []

    def get_text_safely(self, element):
        if element is None:
            return ""
        try:
            text = element.text.strip()
            if not text:
                text = element.get_attribute('innerText').strip()
            if not text:
                text = element.get_attribute('textContent').strip()
            return text
        except Exception:
            return ""

    def get_rating(self, review, driver):
        try:
            # Прокручиваем к блоку рейтинга для уверенности
            rating_container = self.find_element_safely(review, ".//div[contains(@class, 'p6x_31')]")
            if rating_container:
                driver.execute_script("arguments[0].scrollIntoView(true);", rating_container)
                time.sleep(0.5)  # Даём время на рендеринг

            stars = self.find_elements_safely(review, self.selectors['rating_stars'])
            if stars:
                rating = 0
                for star in stars:
                    style = star.get_attribute('style') or ''
                    logger.debug(f"Стиль звезды: {style}")
                    # Учитываем вариации форматирования цвета
                    if 'rgba(255, 165, 0, 1)' in style or 'rgb(255, 165, 0)' in style.replace(' ', ''):
                        rating += 1
                logger.debug(f"Найдено {len(stars)} звёзд, заполнено: {rating}")
                if rating > 0:
                    return rating
                else:
                    logger.warning("Звёзды найдены, но ни одна не заполнена (все серые?)")
                    return 0

            logger.warning("Звёзды рейтинга не найдены")
            return 0
        except Exception as e:
            logger.debug(f"Ошибка при определении рейтинга: {e}")
            return 0

    def click_element_safely(self, driver, element):
        try:
            driver.execute_script("arguments[0].click();", element)
            return True
        except Exception as e:
            logger.debug(f"Не удалось кликнуть по элементу: {e}")
            return False

    def parse(self, link=None, output_dir=None):
        if not link:
            logger.error("Не указана ссылка для парсинга!")
            return

        if "ozon.ru" not in link:
            logger.error("Некорректная ссылка для сайта Ozon!")
            return

        driver = None
        try:
            driver = setup_browser()
            driver.get(link)
            logger.info(f"Начинаем парсинг отзывов с сайта Ozon: {link}")
            time.sleep(5)

            product_name = None
            main_page_product_name = None

            try:
                product_name_elem = self.find_element_safely(driver, self.selectors['product_name'])
                if product_name_elem:
                    main_page_product_name = self.get_text_safely(product_name_elem)
                    logger.info(f"Найдено название товара на главной странице: {main_page_product_name}")
                else:
                    logger.warning("Элемент с названием товара не найден на главной странице")
            except Exception as e:
                logger.warning(f"Ошибка при поиске названия товара на главной странице: {e}")

            if "/reviews/" not in link:
                try:
                    reviews_tab = self.find_element_safely(driver, self.selectors['reviews_tab'])
                    if reviews_tab:
                        self.click_element_safely(driver, reviews_tab)
                        logger.info("Переходим на вкладку отзывов")
                        time.sleep(3)
                    else:
                        logger.warning("Вкладка отзывов не найдена")
                except Exception as e:
                    logger.warning(f"Ошибка при переходе на вкладку отзывов: {e}")

            try:
                product_name_elem = self.find_element_safely(driver, self.selectors['product_name'])
                if product_name_elem:
                    product_name = self.get_text_safely(product_name_elem)
                    logger.info(f"Найдено название товара на странице отзывов: {product_name}")
                elif main_page_product_name:
                    product_name = main_page_product_name
                    logger.info(f"Используем название товара с главной страницы: {product_name}")
                else:
                    logger.warning("Название товара не найдено ни на одной из страниц")
            except Exception as e:
                logger.warning(f"Ошибка при поиске названия товара на странице отзывов: {e}")
                if main_page_product_name:
                    product_name = main_page_product_name
                    logger.info(f"Используем название товара с главной страницы: {product_name}")
                else:
                    logger.warning("Название товара не найдено ни на одной из страниц")

            if product_name:
                safe_product_name = re.sub(r'[<>:"/\\|?*]+', '', product_name)
                safe_product_name = re.sub(r'[\n\r]+', ' ', safe_product_name)
                safe_product_name = re.sub(r'^\d+\s*', '', safe_product_name)
                safe_product_name = safe_product_name.strip().replace(' ', '_')
                save_filename = f"{safe_product_name}_ozon.csv"
            else:
                save_filename = "ozon_reviews.csv"

            save_path = os.path.join(output_dir if output_dir else SITE3_DATA_PATH, save_filename)
            logger.info(f"Имя файла для сохранения: {save_filename}")

            parsed_reviews = []
            seen_reviews = set()
            current_page = 1

            current_url = driver.current_url
            if "/reviews/" not in current_url:
                try:
                    reviews_links = self.find_elements_safely(driver, "//a[contains(@href, '/reviews/')]")
                    if reviews_links:
                        reviews_url = reviews_links[0].get_attribute('href')
                        if reviews_url:
                            driver.get(reviews_url)
                            logger.info(f"Переходим на страницу отзывов: {reviews_url}")
                            time.sleep(3)
                except Exception as e:
                    logger.warning(f"Ошибка при переходе на страницу отзывов: {e}")

            while current_page <= self.max_pages:
                logger.info(f"Парсим страницу {current_page}")

                try:
                    pagination_container = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'p8v_31') or contains(@class, 't1r_31')]"))
                    )
                    logger.info("Блок пагинации загружен")
                    driver.execute_script("arguments[0].scrollIntoView(true);", pagination_container)
                    time.sleep(1)
                except TimeoutException:
                    logger.warning("Блок пагинации не найден после ожидания")
                    break

                driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 2);")
                time.sleep(1)
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)

                try:
                    frames = driver.find_elements(By.TAG_NAME, "iframe")
                    for frame in frames:
                        if "reviews" in frame.get_attribute('src').lower():
                            driver.switch_to.frame(frame)
                            logger.info("Переключились на iframe с отзывами")
                            break
                except Exception as e:
                    logger.debug(f"Не удалось переключиться на iframe: {e}")

                reviews = self.find_elements_safely(driver, self.selectors['review_container'])
                if not reviews:
                    reviews = self.find_elements_safely(
                        driver,
                        "div[data-review-uuid], div[data-widget*='webReview'], div[data-widget*='review'], div[class*='review']",
                        By.CSS_SELECTOR
                    )

                logger.info(f"Найдено {len(reviews)} отзывов")

                if not reviews:
                    logger.warning("На этой странице отзывы не найдены")
                    try:
                        screenshot_path = os.path.join(output_dir if output_dir else SITE3_DATA_PATH, f"debug_page_{current_page}.png")
                        driver.save_screenshot(screenshot_path)
                        logger.info(f"Сохранён отладочный скриншот: {screenshot_path}")
                        html_path = os.path.join(output_dir if output_dir else SITE3_DATA_PATH, f"debug_page_{current_page}.html")
                        with open(html_path, 'w', encoding='utf-8') as f:
                            f.write(driver.page_source)
                        logger.info(f"Сохранён исходный код страницы для отладки: {html_path}")
                    except Exception as e:
                        logger.warning(f"Не удалось сохранить отладочную информацию: {e}")
                else:
                    for review in reviews:
                        try:
                            username_elem = self.find_element_safely(review, self.selectors['username'])
                            username = self.get_text_safely(username_elem) or "Анонимный пользователь"
                            logger.debug(f"Извлечённое имя пользователя: {username}")

                            date_elem = self.find_element_safely(review, self.selectors['date'])
                            date = self.get_text_safely(date_elem) or "Дата не указана"
                            logger.debug(f"Извлечённая дата: {date}")

                            review_text_elem = self.find_element_safely(review, self.selectors['review_text'])
                            review_text = self.get_text_safely(review_text_elem) or ""
                            logger.debug(f"Извлечённый текст отзыва: {review_text[:50]}...")

                            rating = self.get_rating(review, driver)
                            logger.debug(f"Извлечённый рейтинг: {rating}")

                            review_key = (username, date, review_text[:50])
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

                next_page_found = False
                try:
                    pagination_links = self.find_elements_safely(
                        pagination_container,
                        ".//a[contains(@href, '/reviews/') and (contains(@class, 'vp6_31') or contains(text(), 'Дальше'))]"
                    )
                    if pagination_links:
                        logger.info(f"Найдено {len(pagination_links)} ссылок пагинации")
                        current_page_num = current_page
                        next_page_link = None
                        for link in pagination_links:
                            href = link.get_attribute('href') or ""
                            link_text = self.get_text_safely(link).lower()
                            if "дальше" in link_text or "next" in link_text:
                                next_page_link = link
                                logger.debug(f"Найдена кнопка 'Дальше': {href}")
                                break
                            if 'page=' in href:
                                try:
                                    page_num = int(re.search(r'page=(\d+)', href).group(1))
                                    if page_num == current_page + 1:
                                        next_page_link = link
                                        logger.debug(f"Найдена следующая страница {page_num}: {href}")
                                        break
                                except (ValueError, AttributeError):
                                    continue
                        if next_page_link:
                            if self.click_element_safely(driver, next_page_link):
                                logger.info(f"Переходим на следующую страницу")
                                next_page_found = True
                                time.sleep(3)
                            else:
                                logger.warning("Не удалось кликнуть по ссылке следующей страницы")
                                break
                        else:
                            logger.info("Ссылка на следующую страницу не найдена")
                            break
                    else:
                        logger.warning("Ссылки пагинации не найдены в контейнере")
                        break
                except Exception as e:
                    logger.warning(f"Ошибка при обработке пагинации: {e}")
                    break

                if not next_page_found:
                    logger.info("Следующая страница не найдена, парсинг завершён")
                    break

                current_page += 1

            try:
                driver.switch_to.default_content()
            except Exception:
                pass

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