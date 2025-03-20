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
        # Обновлённые селекторы для повышения устойчивости к изменениям классов
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
            'username': (
                ".//span[@itemprop='name'] | "
                ".//div[contains(@class, 'user')] | "
                ".//div[contains(@class, 'author')] | "
                ".//span[contains(@class, 'v5p')] | "
                ".//div[contains(@class, 'vp5')]//span | "
                ".//div[./preceding-sibling::div[contains(text(), 'Автор')]]"
            ),
            'date': (
                ".//span[@itemprop='datePublished'] | "
                ".//div[contains(@class, 'date')] | "
                ".//div[contains(@class, 'p3y')] | "
                ".//div[./preceding-sibling::div[contains(text(), 'Дата')]]"
            ),
            'review_text': (
                ".//div[@itemprop='description'] | "
                ".//div[contains(@class, 'text')] | "
                ".//div[contains(text(), 'Комментарий')]//following-sibling::div | "
                ".//span[contains(@class, 'y4p')] | "
                ".//div[contains(@class, 'p5y')]//span | "
                ".//div[./preceding-sibling::div[contains(text(), 'Комментарий') or contains(text(), 'Отзыв')]]"
            ),
            'rating_stars': (
                ".//div[contains(@data-widget, 'webStars')] | "
                ".//div[@itemprop='ratingValue'] | "
                ".//div[contains(@class, 'y3p')]//svg | "
                ".//div[./svg[contains(@style, 'color')]]"
            ),
            'pagination': (
                "//div[contains(@class, 'pw6') or contains(@class, 'pagination') or contains(@data-widget, 'webPagination')]//a[contains(@href, '/reviews/') and contains(@href, 'page=')] | "
                "//a[contains(text(), 'Дальше')] | "
                "//a[contains(text(), 'след')] | "
                "//a[contains(@aria-label, 'Next')] | "
                "//div[contains(@class, 'pw6')]//a[not(contains(@class, 'pw4')) and contains(@href, '/reviews/')]"
            )
        }

    def find_element_safely(self, parent, selector, by=By.XPATH, timeout=5):
        """Безопасный поиск элемента с обработкой таймаута и исключений"""
        try:
            if by == By.XPATH:
                elements = parent.find_elements(by, selector)
                if elements:
                    return elements[0]
            else:
                elements = parent.find_elements(by, selector)
                if elements:
                    return elements[0]
            return None
        except Exception as e:
            logger.debug(f"Не удалось найти элемент с селектором {selector}: {e}")
            return None

    def find_elements_safely(self, parent, selector, by=By.XPATH):
        """Безопасный поиск элементов с обработкой исключений"""
        try:
            return parent.find_elements(by, selector)
        except Exception as e:
            logger.debug(f"Не удалось найти элементы с селектором {selector}: {e}")
            return []

    def get_text_safely(self, element):
        """Безопасное извлечение текста из элемента"""
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
        """Извлечение рейтинга с использованием нескольких методов"""
        try:
            # Метод 1: Подсчёт SVG-звёзд (новая структура)
            stars_elems = self.find_elements_safely(review, self.selectors['rating_stars'])
            if stars_elems:
                for stars_elem in stars_elems:
                    active_stars = self.find_elements_safely(
                        stars_elem,
                        ".//svg[contains(@style, 'color') and .//path[contains(@d, 'M9.358')]]"
                    )
                    if active_stars:
                        return len(active_stars)

            # Метод 2: Поиск рейтинга через itemprop
            rating_elem = self.find_element_safely(review, ".//*[@itemprop='ratingValue']")
            if rating_elem:
                rating_text = self.get_text_safely(rating_elem)
                if rating_text and rating_text.isdigit():
                    return int(rating_text)
                rating_value = rating_elem.get_attribute('content')
                if rating_value and rating_value.isdigit():
                    return int(rating_value)

            # Метод 3: Поиск текста, упоминающего звёзды
            all_text = self.get_text_safely(review)
            if all_text:
                match = re.search(r'(\d+)(?:\s*из\s*5|\s*звезд)', all_text)
                if match:
                    return int(match.group(1))

            # По умолчанию возвращаем 4, если ничего не найдено
            return 4
        except Exception as e:
            logger.debug(f"Ошибка при определении рейтинга: {e}")
            return 4

    def click_element_safely(self, driver, element):
        """Безопасный клик по элементу с использованием JavaScript"""
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

            # Даём странице время на загрузку
            time.sleep(5)

            # Шаг 1: Получаем название товара
            product_name = None
            main_page_product_name = None

            # Сначала пытаемся получить название товара с главной страницы
            try:
                product_name_elem = self.find_element_safely(driver, self.selectors['product_name'])
                if product_name_elem:
                    main_page_product_name = self.get_text_safely(product_name_elem)
                    logger.info(f"Найдено название товара на главной странице: {main_page_product_name}")
                else:
                    logger.warning("Элемент с названием товара не найден на главной странице")
            except Exception as e:
                logger.warning(f"Ошибка при поиске названия товара на главной странице: {e}")

            # Переходим на вкладку отзывов, если мы ещё не там
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

            # Пытаемся снова получить название товара на странице отзывов, если не нашли ранее
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

            # Формируем имя файла
            if product_name:
                safe_product_name = re.sub(r'[<>:"/\\|?*]+', '', product_name).replace(' ', '_')
                save_filename = f"{safe_product_name}_ozon.csv"
            else:
                save_filename = "ozon_reviews.csv"

            save_path = os.path.join(output_dir if output_dir else SITE3_DATA_PATH, save_filename)

            # Шаг 2: Парсим отзывы
            parsed_reviews = []
            seen_reviews = set()
            max_pages = 5
            current_page = 1

            # Убедимся, что мы на странице отзывов
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

            # Цикл по страницам
            while current_page <= max_pages:
                logger.info(f"Парсим страницу {current_page}")

                # Даём странице загрузиться и ждём блок пагинации
                try:
                    WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'pw6')]"))
                    )
                    logger.info("Блок пагинации загружен")
                except TimeoutException:
                    logger.warning("Блок пагинации не найден после ожидания")

                # Прокручиваем страницу, чтобы загрузить все отзывы
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 2);")
                time.sleep(1)
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)

                # Переключаемся на iframe, если он есть
                try:
                    frames = driver.find_elements(By.TAG_NAME, "iframe")
                    for frame in frames:
                        if "reviews" in frame.get_attribute('src').lower():
                            driver.switch_to.frame(frame)
                            logger.info("Переключились на iframe с отзывами")
                            break
                except Exception as e:
                    logger.debug(f"Не удалось переключиться на iframe: {e}")

                # Ищем отзывы с использованием нескольких методов
                reviews = []

                # Метод 1: Основной селектор
                reviews = self.find_elements_safely(driver, self.selectors['review_container'])

                # Метод 2: Альтернативный CSS-селектор
                if not reviews:
                    reviews = self.find_elements_safely(
                        driver,
                        "div[data-review-uuid], div[data-widget*='webReview'], div[data-widget*='review'], div[class*='review']",
                        By.CSS_SELECTOR
                    )

                # Метод 3: Ищем блоки с звёздами
                if not reviews:
                    stars_containers = self.find_elements_safely(driver, self.selectors['rating_stars'])
                    if stars_containers:
                        for container in stars_containers:
                            parent = container.find_element(By.XPATH, "./ancestor::div[@data-review-uuid or contains(@data-widget, 'review')]")
                            if parent and parent not in reviews:
                                reviews.append(parent)

                # Метод 4: Ищем через itemprop
                if not reviews:
                    reviews = self.find_elements_safely(driver, "//*[@itemprop='review']")

                # Метод 5: Ищем текст, указывающий на отзыв
                if not reviews:
                    reviews = self.find_elements_safely(
                        driver,
                        "//div[contains(text(), 'Комментарий') or contains(text(), 'Достоинства') or contains(text(), 'Недостатки')]//ancestor::div[3]"
                    )

                logger.info(f"Найдено {len(reviews)} отзывов")

                # Отладка, если отзывы не найдены
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
                    # Обрабатываем найденные отзывы
                    for review in reviews:
                        try:
                            # Получаем имя пользователя
                            username_elem = self.find_element_safely(review, self.selectors['username'])
                            username = self.get_text_safely(username_elem) or "Анонимный пользователь"

                            # Получаем дату
                            date_elem = self.find_element_safely(review, self.selectors['date'])
                            date = self.get_text_safely(date_elem) or "Дата не указана"

                            # Получаем текст отзыва
                            review_text_elem = self.find_element_safely(review, self.selectors['review_text'])
                            review_text = self.get_text_safely(review_text_elem) or ""

                            # Извлекаем достоинства и недостатки для резерва
                            pros_elem = self.find_element_safely(review, ".//div[contains(text(), 'Достоинства')]//following-sibling::div")
                            cons_elem = self.find_element_safely(review, ".//div[contains(text(), 'Недостатки')]//following-sibling::div")
                            comment_elem = self.find_element_safely(review, ".//div[contains(text(), 'Комментарий')]//following-sibling::div")

                            pros = self.get_text_safely(pros_elem)
                            cons = self.get_text_safely(cons_elem)
                            comment = self.get_text_safely(comment_elem)

                            # Если текст отзыва пуст, формируем его из достоинств, недостатков и комментария
                            if not review_text and (pros or cons or comment):
                                review_text = ""
                                if pros:
                                    review_text += f"Достоинства: {pros}\n"
                                if cons:
                                    review_text += f"Недостатки: {cons}\n"
                                if comment:
                                    review_text += f"Комментарий: {comment}"

                            # Получаем рейтинг
                            rating = self.get_rating(review, driver)

                            # Проверяем на дубликаты
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

                # Переходим на следующую страницу
                next_page_found = False
                try:
                    pagination_links = self.find_elements_safely(driver, self.selectors['pagination'])
                    if pagination_links:
                        logger.info(f"Найдено {len(pagination_links)} ссылок пагинации")
                        for link in pagination_links:
                            page_num_text = self.get_text_safely(link)
                            href = link.get_attribute('href') or ""
                            try:
                                # Извлекаем номер страницы из href
                                if 'page=' in href:
                                    page_num = int(re.search(r'page=(\d+)', href).group(1))
                                    if page_num == current_page + 1:
                                        self.click_element_safely(driver, link)
                                        logger.info(f"Переходим на страницу {page_num}")
                                        next_page_found = True
                                        time.sleep(3)
                                        break
                            except (ValueError, AttributeError):
                                # Проверяем наличие кнопки "Дальше"
                                if "Дальше" in page_num_text.lower() or "след" in page_num_text.lower() or "next" in page_num_text.lower() or "→" in page_num_text:
                                    self.click_element_safely(driver, link)
                                    logger.info("Переходим на следующую страницу")
                                    next_page_found = True
                                    time.sleep(3)
                                    break
                    else:
                        logger.warning("Ссылки пагинации не найдены")
                except Exception as e:
                    logger.warning(f"Ошибка при переходе на следующую страницу: {e}")

                if not next_page_found:
                    logger.info("Следующая страница не найдена, парсинг завершён")
                    break

                current_page += 1

            # Возвращаемся к основному содержимому
            try:
                driver.switch_to.default_content()
            except Exception:
                pass

            # Сохраняем результаты
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