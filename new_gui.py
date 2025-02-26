import customtkinter as ctk
import logging
import os
import threading
import time
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from parsers.site1_parser import Site1Parser
from parsers.site2_parser import Site2Parser
from parsers.site3_parser import Site3Parser
from visualization import plot_rating_histogram
from tkinter import messagebox, filedialog
from tkinter import ttk  # Для Treeview
from review_analyzer import ReviewAnalyzer  # Импортируем наш анализатор

# Константы
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CONFIG_FILE = "parser_config.json"
WINDOW_SIZE = "900x700"
LOG_HEIGHT = 12
SITE_NAMES = {
    "site1": "Отзовик",
    "site2": "iRecommend",
    "site3": "Озон"
}

class InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.INFO

class ParserApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Парсер отзывов")
        self.root.geometry(WINDOW_SIZE)
        self.root.resizable(True, True)
        self.root.configure(bg="#1A1A1A")  # Явно указываем тёмный фон для корневого окна
        self.running = False

        self.output_dir = ctk.StringVar(value=DATA_DIR)
        self.site_links = {site: ctk.StringVar() for site in SITE_NAMES}

        self.create_widgets()
        self.setup_logger_redirect()
        self.load_config()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        ctk.set_appearance_mode("dark")  # Устанавливаем тёмную тему
        print(f"Текущая тема: {ctk.get_appearance_mode()}")  # Отладочный вывод
        self.bind_shortcuts()  # Привязываем горячие клавиши

    def create_widgets(self):
        main_frame = ctk.CTkFrame(self.root, fg_color="#1A1A1A")  # Тёмный фон
        main_frame.pack(padx=5, pady=5, fill="both", expand=True)

        # Создаём вкладки с помощью CTkTabview
        self.tabview = ctk.CTkTabview(main_frame, width=880, height=650, fg_color="#2C2C2C", bg_color="#1A1A1A", border_color="#1DA1F2")
        self.tabview.pack(padx=5, pady=5, fill="both", expand=True)
        self.tabview.add("Парсер отзывов")  # Текущая вкладка для парсинга
        self.tabview.add("Анализ слов")    # Новая вкладка для анализа

        # Вкладка "Парсер отзывов"
        parser_tab = self.tabview.tab("Парсер отзывов")
        
        # Рамка для ссылок
        links_frame = ctk.CTkFrame(parser_tab, fg_color="#2C2C2C")
        links_frame.pack(fill="x", pady=2, padx=5, expand=False)
        ctk.CTkLabel(links_frame, text="Ссылки на отзывы", font=("Arial", 14, "bold"), text_color="white").pack(anchor="w", padx=5, pady=2)
        
        for site in SITE_NAMES:
            site_frame = ctk.CTkFrame(links_frame, fg_color="#2C2C2C")
            site_frame.pack(fill="x", pady=2, padx=5)
            site_frame.grid_columnconfigure(1, weight=1)  # Делаем колонку с полем ввода расширяемой
            ctk.CTkLabel(site_frame, text=SITE_NAMES[site], font=("Arial", 12), text_color="white").grid(row=0, column=0, padx=5, pady=5, sticky="w")
            ctk.CTkEntry(site_frame, textvariable=self.site_links[site], width=600, height=30, font=("Arial", 12), fg_color="#3A3A3A", border_color="#1DA1F2", corner_radius=10).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
            ctk.CTkButton(site_frame, text="Добавить", command=lambda s=site: self.add_link(s), fg_color="#1DA1F2", hover_color="#166AB1", width=80, height=30, font=("Arial", 12), corner_radius=10).grid(row=0, column=2, padx=5, pady=5, sticky="e")

        # Рамка для папки сохранения
        output_frame = ctk.CTkFrame(parser_tab, fg_color="#2C2C2C")
        output_frame.pack(fill="x", pady=2, padx=5, expand=False)
        ctk.CTkLabel(output_frame, text="Папка сохранения", font=("Arial", 14, "bold"), text_color="white").pack(anchor="w", padx=5, pady=2)
        inner_frame = ctk.CTkFrame(output_frame, fg_color="#2C2C2C")
        inner_frame.pack(fill="x", padx=5, pady=2)
        ctk.CTkEntry(inner_frame, textvariable=self.output_dir, width=600, height=30, font=("Arial", 12), fg_color="#3A3A3A", border_color="#1DA1F2", corner_radius=10).pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkButton(inner_frame, text="Обзор...", command=self.select_output_dir, fg_color="#1DA1F2", hover_color="#166AB1", width=80, height=30, font=("Arial", 12), corner_radius=10).pack(side="right", padx=5)

        # Рамка для лога
        log_frame = ctk.CTkFrame(parser_tab, fg_color="#2C2C2C")
        log_frame.pack(fill="both", expand=True, pady=2, padx=5)
        ctk.CTkLabel(log_frame, text="Лог выполнения", font=("Arial", 14, "bold"), text_color="white").pack(anchor="w", padx=5, pady=2)
        self.log_text = ctk.CTkTextbox(log_frame, height=LOG_HEIGHT, width=850, state="disabled", font=("Arial", 12), text_color="white", fg_color="#3A3A3A", border_color="#1DA1F2")
        scrollbar = ctk.CTkScrollbar(log_frame, command=self.log_text.yview, fg_color="#2C2C2C", button_color="#1DA1F2")
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side="left", fill="both", expand=True, padx=5)
        scrollbar.pack(side="right", fill="y")

        # Рамка для управления
        control_frame = ctk.CTkFrame(parser_tab, fg_color="#1A1A1A")
        control_frame.pack(fill="x", pady=5, padx=5, expand=False)
        self.start_btn = ctk.CTkButton(control_frame, text="Старт", command=self.toggle_parsing, fg_color="#1DA1F2", hover_color="#166AB1", width=80, height=30, font=("Arial", 12), corner_radius=10)
        self.start_btn.pack(side="left", padx=5)
        ctk.CTkButton(control_frame, text="Очистить лог", command=self.clear_log, fg_color="#4A4A4A", hover_color="#5A5A5A", width=80, height=30, font=("Arial", 12), corner_radius=10).pack(side="left", padx=5)
        ctk.CTkButton(control_frame, text="Открыть папку", command=self.open_output_dir, fg_color="#4A4A4A", hover_color="#5A5A5A", width=80, height=30, font=("Arial", 12), corner_radius=10).pack(side="left", padx=5)
        self.progress = ctk.CTkProgressBar(control_frame, mode="determinate", width=250, height=20, progress_color="#1DA1F2", fg_color="#4A4A4A")
        self.progress.pack(side="right", padx=5)
        self.progress.set(0)

        # Вкладка "Анализ слов"
        analysis_tab = self.tabview.tab("Анализ слов")

        # Рамка для выбора файлов
        file_frame = ctk.CTkFrame(analysis_tab, fg_color="#2C2C2C")
        file_frame.pack(fill="x", pady=2, padx=5, expand=False)
        ctk.CTkLabel(file_frame, text="Выбор CSV-файлов для анализа", font=("Arial", 14, "bold"), text_color="white").pack(anchor="w", padx=5, pady=2)
        ctk.CTkButton(file_frame, text="Выбрать файлы для анализа по сайтам", command=self.select_files_for_analysis_by_site, fg_color="#1DA1F2", hover_color="#166AB1", width=250, height=30, font=("Arial", 12), corner_radius=10).pack(pady=5)
        ctk.CTkButton(file_frame, text="Агрегированный анализ выбранных файлов", command=self.select_files_for_aggregated_analysis, fg_color="#1DA1F2", hover_color="#166AB1", width=250, height=30, font=("Arial", 12), corner_radius=10).pack(pady=5)

        # Рамка для таблицы результатов
        result_frame = ctk.CTkFrame(analysis_tab, fg_color="#2C2C2C")
        result_frame.pack(fill="both", expand=True, pady=5, padx=5)
        ctk.CTkLabel(result_frame, text="Результаты анализа", font=("Arial", 14, "bold"), text_color="white").pack(anchor="w", padx=5, pady=2)

        # Создаём контейнер для Treeview с тёмным фоном
        tree_frame = ctk.CTkFrame(result_frame, fg_color="#333333")
        tree_frame.pack(fill="both", expand=True)

        # Настройка стиля для таблицы
        style = ttk.Style()
        style.theme_use("clam")  # Используем тему 'clam' для полной настройки стилей
        style.configure("Custom.Treeview",
                        background="#333333",
                        foreground="#FFFFFF",
                        fieldbackground="#333333",
                        rowheight=50)
        style.configure("Custom.Treeview.Heading",
                        background="#444444",
                        foreground="#FFFFFF")
        style.map("Custom.Treeview",
                  background=[("selected", "#1DA1F2")],
                  foreground=[("selected", "#FFFFFF")])

        # Создание таблицы
        self.result_tree = ttk.Treeview(tree_frame, columns=("Category", "Pros", "Cons", "PositiveKeywords", "NegativeKeywords", "CommonKeywords"), show="headings", height=10, style="Custom.Treeview")
        self.result_tree.heading("Category", text="Категория")
        self.result_tree.heading("Pros", text="Плюсы")
        self.result_tree.heading("Cons", text="Минусы")
        self.result_tree.heading("PositiveKeywords", text="Ключевые слова (положительные)")
        self.result_tree.heading("NegativeKeywords", text="Ключевые слова (отрицательные)")
        self.result_tree.heading("CommonKeywords", text="Общие ключевые слова")

        # Установка ширины столбцов
        self.result_tree.column("Category", width=150, anchor="w")
        self.result_tree.column("Pros", width=250, anchor="w")
        self.result_tree.column("Cons", width=250, anchor="w")
        self.result_tree.column("PositiveKeywords", width=300, anchor="w")
        self.result_tree.column("NegativeKeywords", width=300, anchor="w")
        self.result_tree.column("CommonKeywords", width=300, anchor="w")

        self.result_tree.pack(fill="both", expand=True)

        # Стиль скроллбара
        style.configure("Custom.Vertical.TScrollbar",
                        troughcolor="#1E1E1E",          # Тёмно-серый фон дорожки
                        background="#1DA1F2",           # Ярко-синий для ручки
                        arrowcolor="#FFFFFF")           # Белый цвет стрелок

        # Вертикальный скроллбар
        v_scrollbar = ttk.Scrollbar(tree_frame, orient='vertical', style="Custom.Vertical.TScrollbar", command=self.result_tree.yview)
        v_scrollbar.pack(side="right", fill="y")
        self.result_tree.configure(yscrollcommand=v_scrollbar.set)

        # Горизонтальный скроллбар (опционально, если строки слишком длинные)
        style.configure("Custom.Horizontal.TScrollbar",
                        troughcolor="#1E1E1E",
                        background="#1DA1F2",
                        arrowcolor="#FFFFFF")
        h_scrollbar = ttk.Scrollbar(tree_frame, orient='horizontal', style="Custom.Horizontal.TScrollbar", command=self.result_tree.xview)
        h_scrollbar.pack(side="bottom", fill="x")
        self.result_tree.configure(xscrollcommand=h_scrollbar.set)

        # ... (остальной код остаётся без изменений) ...

    def log(self, message, is_error=False):
        self.log_text.configure(state="normal")
        tag = "ERROR" if is_error else "INFO"
        color = "red" if is_error else "white"
        self.log_text.tag_config(tag, foreground=color)
        self.log_text.insert("end", f"[{time.strftime('%H:%M:%S')}] {tag}: {message}\n", tag)
        self.log_text.configure(state="disabled")
        self.log_text.see("end")

    def add_link(self, site):
        link = self.site_links[site].get()
        if link:
            self.log(f"Добавлена ссылка для {SITE_NAMES[site]}: {link}")
        else:
            self.log(f"Поле ссылки для {SITE_NAMES[site]} пустое", is_error=True)

    def setup_logger_redirect(self):
        handler = logging.StreamHandler(self)
        handler.setLevel(logging.INFO)
        handler.addFilter(InfoFilter())
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(handler)

    def write(self, message):
        if message.strip():
            self.log(message.strip())

    def flush(self):
        pass

    def select_output_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir.set(directory)

    def clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete(1.0, ctk.END)
        self.log_text.configure(state="disabled")

    def open_output_dir(self):
        output_dir = os.path.abspath(self.output_dir.get())
        if os.path.exists(output_dir):
            subprocess.Popen(f'explorer "{output_dir}"' if os.name == 'nt' else ['xdg-open', output_dir])
        else:
            self.log(f"Папка {output_dir} не существует", is_error=True)

    def toggle_parsing(self):
        if self.running:
            self.running = False
            self.start_btn.configure(text="Старт")
            self.progress.stop()
            self.log("Парсинг остановлен пользователем")
        else:
            self.running = True
            self.start_btn.configure(text="Стоп")
            self.progress.start()
            threading.Thread(target=self.run_parsers).start()

    def run_parsers(self):
        parsers = [(name, parser(), link) for site, (name, parser) in {
            "site1": ("Отзовик", Site1Parser),
            "site2": ("iRecommend", Site2Parser),
            "site3": ("Озон", Site3Parser)
        }.items() if (link := self.site_links[site].get())]

        total_steps = len(parsers)
        saved_files = []

        def run_parser(name, parser, link):
            if not self.running:
                return
            try:
                self.log(f"Запуск парсера: {name}")
                parser.parse(link, output_dir=self.output_dir.get())
                self.log(f"Парсер {name} завершен")
            except Exception as e:
                self.log(f"Ошибка при работе с {name}: {str(e)}", is_error=True)

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(run_parser, name, parser, link): name for name, parser, link in parsers}
            for i, future in enumerate(futures):
                future.result()
                self.progress.set((i + 1) / total_steps * 100)
                self.root.update_idletasks()

        self.running = False
        self.progress.set(0)
        self.start_btn.configure(text="Старт")
        messagebox.showinfo("Информация", "Парсинг завершен!")

    def select_files_for_analysis_by_site(self):
        """Выбор CSV-файлов для анализа отзывов по сайтам и отображение результатов в таблице."""
        csv_files = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
        if not csv_files:
            self.log("Не выбраны CSV-файлы для анализа!", is_error=True)
            return

        analyzer = ReviewAnalyzer()
        self.result_tree.delete(*self.result_tree.get_children())  # Очистка предыдущих данных
        for csv_file in csv_files:
            try:
                # Извлекаем имя сайта из имени файла, используя SITE_NAMES
                site_name = self.get_site_name_from_filename(csv_file)
                if not site_name:
                    self.log(f"Не удалось определить сайт для файла {csv_file}", is_error=True)
                    continue
                results = analyzer.analyze_reviews(csv_file)
                # Применяем перенос текста для каждого значения
                self.result_tree.insert("", "end", values=(
                    site_name,
                    self.wrap_text(results["Плюсы"]),
                    self.wrap_text(results["Минусы"]),
                    self.wrap_text(results["Ключевые слова (положительные)"]),
                    self.wrap_text(results["Ключевые слова (отрицательные)"]),
                    self.wrap_text(results["Общие ключевые слова"])
                ))
            except Exception as e:
                self.log(f"Ошибка при анализе файла {csv_file}: {str(e)}", is_error=True)
                continue

        self.log("Анализ отзывов по сайтам завершён!")
        self.tabview.set("Анализ слов")  # Переключаемся на вкладку анализа после анализа

    def select_files_for_aggregated_analysis(self):
        """Выбор CSV-файлов для агрегированного анализа отзывов и отображение результатов в таблице."""
        csv_files = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
        if not csv_files:
            self.log("Не выбраны CSV-файлы для анализа!", is_error=True)
            return

        analyzer = ReviewAnalyzer()
        self.result_tree.delete(*self.result_tree.get_children())  # Очистка предыдущих данных
        aggregated_results = analyzer.aggregate_reviews(csv_files)
        for key, value in aggregated_results.items():
            self.result_tree.insert("", "end", values=(key, self.wrap_text(value), "", "", "", ""))

        self.log("Агрегированный анализ отзывов завершён!")
        self.tabview.set("Анализ слов")  # Переключаемся на вкладку анализа после анализа

    def wrap_text(self, text):
        """Оборачивает текст для корректного отображения в таблице с переносами."""
        if not text:
            return ""
        return "\n".join([text[i:i+20] for i in range(0, len(text), 20)])  # Разбиваем текст на части по 20 символов

    def get_site_name_from_filename(self, filename):
        """Извлекает имя сайта из имени файла, используя SITE_NAMES."""
        base_name = os.path.basename(filename).lower()
        if "otzovik" in base_name:
            return SITE_NAMES["site1"]
        elif "irecommend" in base_name:
            return SITE_NAMES["site2"]
        elif "ozon" in base_name:
            return SITE_NAMES["site3"]
        return None

    def run_visualization(self, file_path):
        try:
            plot_rating_histogram(file_path)
        except Exception as e:
            self.log(f"Ошибка при визуализации: {str(e)}", is_error=True)

    def bind_shortcuts(self):
        def handle_ctrl_key(event):
            if event.state & 0x4:  # Проверка на Ctrl
                if event.keycode == 67:  # Ctrl + C (копировать)
                    event.widget.event_generate("<<Copy>>")
                    return "break"
                elif event.keycode == 86:  # Ctrl + V (вставить)
                    event.widget.event_generate("<<Paste>>")
                    return "break"
                elif event.keycode == 88:  # Ctrl + X (вырезать)
                    event.widget.event_generate("<<Cut>>")
                    return "break"
                elif event.keycode == 65:  # Ctrl + A (выделить всё)
                    self.select_all(event.widget)
                    return "break"
                elif event.keycode == 90:  # Ctrl + Z (отменить)
                    if isinstance(event.widget, ctk.CTkEntry):
                        event.widget.delete(0, ctk.END)  # Простое "отменить" — очистка поля
                    return "break"

        def bind_to_entry(widget):
            if isinstance(widget, ctk.CTkEntry):
                widget.bind("<Control-Key>", handle_ctrl_key)
            for child in widget.winfo_children():
                bind_to_entry(child)

        bind_to_entry(self.root)

    def select_all(self, widget):
        widget.select_range(0, 'end')
        widget.icursor('end')
    
    def load_config(self):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                self.output_dir.set(config.get("output_dir", DATA_DIR))
                for site, link in config.get("site_links", {}).items():
                    if site in self.site_links:
                        self.site_links[site].set(link)
        except FileNotFoundError:
            pass

    def save_config(self):
        config = {
            "output_dir": self.output_dir.get(),
            "site_links": {site: link.get() for site, link in self.site_links.items()}
        }
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f)

    def on_closing(self):
        if self.running:
            self.running = False
            time.sleep(1)  # Даем потокам завершить работу
        self.save_config()
        self.root.destroy()

if __name__ == "__main__":
    root = ctk.CTk()
    app = ParserApp(root)
    root.mainloop()