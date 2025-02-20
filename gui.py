import tkinter as tk
import logging
from tkinter import ttk, messagebox, filedialog
from parsers.site1_parser import Site1Parser
from parsers.site2_parser import Site2Parser
from parsers.site3_parser import Site3Parser
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
from visualization import plot_rating_histogram

# Абсолютный путь к папке data относительно gui.py
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CONFIG_FILE = "parser_config.json"
WINDOW_SIZE = "600x550"
LOG_HEIGHT = 10
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
        self.running = False

        self.output_dir = tk.StringVar(value=DATA_DIR)
        self.site_links = {site: tk.StringVar() for site in SITE_NAMES}

        self.create_widgets()
        self.setup_logger_redirect()
        self.bind_shortcuts()
        self.load_config()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(padx=10, pady=10, fill="both", expand=True)

        links_frame = ttk.LabelFrame(main_frame, text="Введите ссылки на страницы с отзывами")
        links_frame.pack(fill="x", pady=5)

        for site in SITE_NAMES:
            site_frame = ttk.Frame(links_frame)
            site_frame.pack(fill="x", pady=2)
            ttk.Label(site_frame, text=SITE_NAMES[site]).pack(side="left", padx=5)
            ttk.Entry(site_frame, textvariable=self.site_links[site], width=40).pack(side="left", padx=5, fill="x", expand=True)
            ttk.Button(site_frame, text="Добавить", command=lambda s=site: self.add_link(s)).pack(side="right", padx=5)

        output_frame = ttk.LabelFrame(main_frame, text="Папка для сохранения")
        output_frame.pack(fill="x", pady=5)
        ttk.Entry(output_frame, textvariable=self.output_dir).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(output_frame, text="Обзор...", command=self.select_output_dir).pack(side="right", padx=5)

        log_frame = ttk.LabelFrame(main_frame, text="Лог выполнения")
        log_frame.pack(fill="both", expand=True, pady=5)
        self.log_text = tk.Text(log_frame, height=LOG_HEIGHT, state="disabled")
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill="x", pady=5)
        self.start_btn = ttk.Button(control_frame, text="Старт", command=self.toggle_parsing)
        self.start_btn.pack(side="left", padx=5)
        ttk.Button(control_frame, text="Очистить лог", command=self.clear_log).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Открыть папку", command=self.open_output_dir).pack(side="left", padx=5)
        self.progress = ttk.Progressbar(control_frame, mode="determinate", maximum=100, length=200)
        self.progress.pack(side="right", padx=5)

    def log(self, message, is_error=False):
        self.log_text.configure(state="normal")
        tag = "ERROR" if is_error else "INFO"
        color = "red" if is_error else "black"
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
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state="disabled")

    def open_output_dir(self):
        output_dir = os.path.abspath(self.output_dir.get())  # Преобразуем в абсолютный путь
        if os.path.exists(output_dir):
            subprocess.Popen(f'explorer "{output_dir}"' if os.name == 'nt' else ['xdg-open', output_dir])
        else:
            self.log(f"Папка {output_dir} не существует", is_error=True)

    def toggle_parsing(self):
        if self.running:
            self.running = False
            self.start_btn.config(text="Старт")
            self.progress.stop()
            self.log("Парсинг остановлен пользователем")
        else:
            self.running = True
            self.start_btn.config(text="Стоп")
            self.progress.start(10)
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
                self.progress['value'] = ((i + 1) / total_steps) * 100
                self.root.update_idletasks()

        self.running = False
        self.progress.stop()
        self.start_btn.config(text="Старт")
        messagebox.showinfo("Информация", "Парсинг завершен!")
        if saved_files and messagebox.askyesno("Гистограмма", "Показать гистограмму последнего файла?"):
            self.run_visualization(saved_files[-1])

    def show_histogram(self):
        file_path = filedialog.askopenfilename(title="Выберите CSV файл", filetypes=[("CSV файлы", "*.csv")])
        if file_path:
            threading.Thread(target=self.run_visualization, args=(file_path,)).start()

    def run_visualization(self, file_path):
        try:
            plot_rating_histogram(file_path)
        except Exception as e:
            self.log(f"Ошибка при визуализации: {str(e)}", is_error=True)

    def bind_shortcuts(self):
        def handle_ctrl_key(event):
            if event.state & 0x4:
                if event.keycode == 67:  # Ctrl + C
                    event.widget.event_generate("<<Copy>>")
                    return "break"
                elif event.keycode == 86:  # Ctrl + V
                    event.widget.event_generate("<<Paste>>")
                    return "break"
                elif event.keycode == 88:  # Ctrl + X
                    event.widget.event_generate("<<Cut>>")
                    return "break"
                elif event.keycode == 65:  # Ctrl + A
                    self.select_all(event.widget)
                    return "break"

        def bind_to_entry(widget):
            if isinstance(widget, tk.Entry):
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
    root = tk.Tk()
    app = ParserApp(root)
    root.mainloop()