import tkinter as tk
from tkhtmlview import HTMLLabel
import plotly.graph_objects as go
import plotly.offline as pyo
import pandas as pd
import os

def plot_rating_histogram(csv_file_path):
    # Чтение данных из CSV
    df = pd.read_csv(csv_file_path)
    
    # Подсчет количества отзывов для каждой оценки
    rating_counts = df['Оценка'].value_counts().sort_index()

    # Построение гистограммы
    fig = go.Figure(data=[go.Bar(x=rating_counts.index, y=rating_counts.values)])
    
    # Добавляем подписи
    fig.update_layout(
        title="Количество отзывов по оценкам",
        xaxis_title="Оценка",
        yaxis_title="Количество отзывов",
        template="plotly_dark"
    )

    # Генерация HTML файла для отображения в Tkinter
    html_file_path = "rating_histogram.html"
    pyo.plot(fig, filename=html_file_path, auto_open=False)

    # Открытие GUI с отображением HTML
    open_html_in_gui(html_file_path)

def open_html_in_gui(html_file_path):
    # Создаем окно Tkinter для отображения HTML
    root = tk.Tk()
    root.title("Гистограмма оценок")
    
    # Используем HTMLLabel для отображения HTML-контента в Tkinter
    with open(html_file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    label = HTMLLabel(root, html=html_content)
    label.pack(fill="both", expand=True)

    # Показываем окно
    root.mainloop()

