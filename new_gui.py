import sys
import os
import json
import time
import logging
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QProgressBar, QTextEdit, QFileDialog, QMessageBox
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QShortcut, QKeySequence
from parsers.site1_parser import Site1Parser
from parsers.site2_parser import Site2Parser
from parsers.site3_parser import Site3Parser
from visualization import plot_rating_histogram

# Абсолютный путь к папке data относительно gui.py
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CONFIG_FILE = os.path.join(PROJECT_ROOT, "parser_config.json")
WINDOW_SIZE = (600, 550)
SITE_NAMES = {
    "site1": "Отзовик",
    "site2": "iRecommend",
    "site3": "Озон"
}

class ParserThread(QThread):
    log_signal = pyqtSignal(str, bool)  # Сообщение, is_error
    progress_signal = pyqtSignal(int)   # Прогресс в процентах
    finished_signal = pyqtSignal()      # Сигнал завершения

    def __init__(self, parser, link, output_dir):
        super().__init__()
        self.parser = parser
        self.link = link
        self.output_dir = output_dir

    def run(self):
        self.log_signal.emit(f"Запуск парсера для {self.link}", False)
        try:
            self.parser.parse(self.link, output_dir=self.output_dir)
            self.progress_signal.emit(100)
            self.log_signal.emit("Парсер завершен", False)
        except Exception as e:
            self.log_signal.emit(f"Ошибка: {str(e)}", True)
        self.finished_signal.emit()

class ParserApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Парсер отзывов")
        self.resize(*WINDOW_SIZE)
        self.running = False

        self.output_dir = DATA_DIR
        self.site_links = {site: "" for site in SITE_NAMES}

        self.init_ui()
        self.setup_logging()
        self.setup_shortcuts()
        self.load_config()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Секция ввода ссылок
        links_frame = QWidget()
        links_layout = QVBoxLayout(links_frame)
        main_layout.addWidget(links_frame)

        self.link_inputs = {}
        for site in SITE_NAMES:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{SITE_NAMES[site]}:"))
            self.link_inputs[site] = QLineEdit()
            row.addWidget(self.link_inputs[site])
            add_btn = QPushButton("Добавить")
            add_btn.clicked.connect(lambda _, s=site: self.add_link(s))
            row.addWidget(add_btn)
            links_layout.addLayout(row)

        # Секция выбора папки
        output_frame = QWidget()
        output_layout = QHBoxLayout(output_frame)
        self.output_label = QLabel(f"Папка вывода: {self.output_dir}")
        output_layout.addWidget(self.output_label)
        browse_btn = QPushButton("Обзор...")
        browse_btn.clicked.connect(self.select_output_dir)
        output_layout.addWidget(browse_btn)
        main_layout.addWidget(output_frame)

        # Секция логов
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(150)
        main_layout.addWidget(self.log_text)

        # Секция управления
        control_frame = QWidget()
        control_layout = QHBoxLayout(control_frame)
        self.start_btn = QPushButton("Старт")
        self.start_btn.clicked.connect(self.toggle_parsing)
        control_layout.addWidget(self.start_btn)
        clear_btn = QPushButton("Очистить лог")
        clear_btn.clicked.connect(self.clear_log)
        control_layout.addWidget(clear_btn)
        open_btn = QPushButton("Открыть папку")
        open_btn.clicked.connect(self.open_output_dir)
        control_layout.addWidget(open_btn)
        self.progress = QProgressBar()
        self.progress.setMaximum(100)
        control_layout.addWidget(self.progress)
        main_layout.addWidget(control_frame)

    def setup_logging(self):
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(handler)
        self.logger = logging.getLogger()

    def log(self, message, is_error=False):
        color = "red" if is_error else "black"
        self.log_text.append(f"<span style='color:{color}'>[{time.strftime('%H:%M:%S')}] {'ERROR' if is_error else 'INFO'}: {message}</span>")
        self.log_text.ensureCursorVisible()

    def add_link(self, site):
        link = self.link_inputs[site].text()
        if link:
            self.site_links[site] = link
            self.log(f"Добавлена ссылка для {SITE_NAMES[site]}: {link}")
        else:
            self.log(f"Поле ссылки для {SITE_NAMES[site]} пустое", is_error=True)

    def select_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Выберите папку")
        if directory:
            self.output_dir = directory
            self.output_label.setText(f"Папка вывода: {self.output_dir}")

    def clear_log(self):
        self.log_text.clear()

    def open_output_dir(self):
        output_dir = os.path.abspath(self.output_dir)
        if os.path.exists(output_dir):
            subprocess.Popen(f'explorer "{output_dir}"' if os.name == 'nt' else ['xdg-open', output_dir])
        else:
            self.log(f"Папка {output_dir} не существует", is_error=True)

    def toggle_parsing(self):
        if self.running:
            self.running = False
            self.start_btn.setText("Старт")
            self.progress.setValue(0)
            self.log("Парсинг остановлен пользователем")
            # Ожидаем завершения всех потоков
            if hasattr(self, 'active_threads'):
                for thread in self.active_threads:
                    thread.wait()
        else:
            self.running = True
            self.start_btn.setText("Стоп")
            self.thread = threading.Thread(target=self.run_parsers)
            self.thread.start()

    def run_parsers(self):
        parsers = [
            (name, parser(), self.site_links[site])
            for site, name in SITE_NAMES.items()
            if self.site_links[site]
            for parser in [[Site1Parser, Site2Parser, Site3Parser][list(SITE_NAMES.keys()).index(site)]]
        ]
        total_steps = len(parsers)
        if not total_steps:
            self.log("Нет ссылок для парсинга", is_error=True)
            self.running = False
            self.start_btn.setText("Старт")
            return

        self.active_threads = []
        self.completed_tasks = 0

        def run_parser(name, parser, link):
            if not self.running:
                return
            thread = ParserThread(parser, link, self.output_dir)
            self.active_threads.append(thread)
            thread.log_signal.connect(self.log)
            thread.progress_signal.connect(lambda p: self.update_progress(p, name, total_steps))
            thread.finished_signal.connect(self.task_finished)
            thread.start()

        with ThreadPoolExecutor(max_workers=3) as executor:
            executor.map(lambda p: run_parser(*p), parsers)

    def update_progress(self, value, name, total_steps):
        if self.running:
            self.completed_tasks += 1  # Увеличиваем на 1 для каждого завершенного парсера
            progress = int((self.completed_tasks / total_steps) * 100)
            self.progress.setValue(progress)
            QApplication.processEvents()

    def task_finished(self):
        if self.running and self.completed_tasks >= len([s for s in self.site_links.values() if s]):
            self.running = False
            self.start_btn.setText("Старт")
            QApplication.processEvents()
            QMessageBox.information(self, "Информация", "Парсинг завершен!")

    def setup_shortcuts(self):
        for site in SITE_NAMES:
            input_field = self.link_inputs[site]
            QShortcut(QKeySequence("Ctrl+C"), input_field, lambda s=site: input_field.copy())
            QShortcut(QKeySequence("Ctrl+V"), input_field, lambda s=site: input_field.paste())
            QShortcut(QKeySequence("Ctrl+X"), input_field, lambda s=site: input_field.cut())
            QShortcut(QKeySequence("Ctrl+A"), input_field, lambda s=site: input_field.selectAll())

    def load_config(self):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                self.output_dir = config.get("output_dir", DATA_DIR)
                self.output_label.setText(f"Папка вывода: {self.output_dir}")
                for site, link in config.get("site_links", {}).items():
                    if site in self.site_links:
                        self.site_links[site] = link
                        self.link_inputs[site].setText(link)
        except FileNotFoundError:
            pass

    def save_config(self):
        config = {
            "output_dir": self.output_dir,
            "site_links": {site: self.link_inputs[site].text() for site in SITE_NAMES}
        }
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f)

    def closeEvent(self, event):
        if self.running:
            self.running = False
            if hasattr(self, 'active_threads'):
                for thread in self.active_threads:
                    thread.wait()
        self.save_config()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ParserApp()
    window.show()
    sys.exit(app.exec())