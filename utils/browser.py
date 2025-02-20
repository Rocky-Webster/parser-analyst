from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

def setup_browser():
    options = Options()
    options.add_argument('--headless')  # Фоновый режим
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36')
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-logging"])

    # Новые опции для оптимизации
    options.add_argument('--blink-settings=imagesEnabled=false')  # Отключение изображений
    options.add_argument('--disable-extensions')  # Отключение расширений
    options.add_argument('--window-size=1920,1080')  # Установка размера окна для стабильности

    # Используем webdriver-manager для автоматической установки ChromeDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver

if __name__ == "__main__":
    # Тестовая проверка
    driver = setup_browser()
    driver.get("https://www.google.com")
    print(driver.title)
    driver.quit()