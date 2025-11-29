"""Конвертер markdown отчётов в PDF через WeasyPrint."""

import logging
from io import BytesIO
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader
from weasyprint import CSS, HTML

logger = logging.getLogger(__name__)

# Путь к шаблонам
TEMPLATES_DIR = Path(__file__).parent / "templates"


class PDFConverter:
    """Конвертирует markdown отчёты в PDF с использованием WeasyPrint."""

    def __init__(self, templates_dir: Optional[Path] = None):
        """
        Инициализация конвертера.

        :param templates_dir: путь к директории с шаблонами
        """
        self.templates_dir = templates_dir or TEMPLATES_DIR
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=True,
        )
        logger.info(f"PDFConverter инициализирован, шаблоны: {self.templates_dir}")

    def markdown_to_html(self, markdown_text: str) -> str:
        """
        Преобразует markdown в HTML.

        Простой конвертер без внешних зависимостей.
        """
        lines = markdown_text.split("\n")
        html_lines = []
        in_table = False
        in_list = False

        for line in lines:
            stripped = line.strip()

            # Заголовки
            if stripped.startswith("# "):
                html_lines.append(f"<h1>{self._process_inline(stripped[2:])}</h1>")
            elif stripped.startswith("## "):
                html_lines.append(f"<h2>{self._process_inline(stripped[3:])}</h2>")
            elif stripped.startswith("### "):
                html_lines.append(f"<h3>{self._process_inline(stripped[4:])}</h3>")

            # Горизонтальная линия
            elif stripped == "---":
                html_lines.append("<hr>")

            # Таблицы
            elif stripped.startswith("|"):
                if not in_table:
                    html_lines.append("<table>")
                    in_table = True

                # Пропускаем строку-разделитель
                if set(stripped.replace("|", "").replace("-", "").strip()) == set():
                    continue

                cells = [c.strip() for c in stripped.split("|")[1:-1]]
                row = (
                    "<tr>"
                    + "".join(f"<td>{self._process_inline(c)}</td>" for c in cells)
                    + "</tr>"
                )
                html_lines.append(row)

            # Списки
            elif stripped.startswith("- "):
                if not in_list:
                    html_lines.append("<ul>")
                    in_list = True
                html_lines.append(f"<li>{self._process_inline(stripped[2:])}</li>")

            # Обычный текст или курсив
            elif stripped.startswith("*") and stripped.endswith("*"):
                html_lines.append(f"<p><em>{stripped[1:-1]}</em></p>")

            # Пустая строка
            elif not stripped:
                if in_table:
                    html_lines.append("</table>")
                    in_table = False
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append("")

            # Обычный параграф
            else:
                html_lines.append(f"<p>{self._process_inline(stripped)}</p>")

        # Закрываем открытые теги
        if in_table:
            html_lines.append("</table>")
        if in_list:
            html_lines.append("</ul>")

        return "\n".join(html_lines)

    def _process_inline(self, text: str) -> str:
        """Обрабатывает inline markdown разметку."""
        import re

        # Жирный текст **text**
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)

        # Курсив *text*
        text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)

        return text

    def convert_to_pdf(
        self,
        markdown_text: str,
        title: str = "Медицинский отчёт",
    ) -> bytes:
        """
        Конвертирует markdown отчёт в PDF.

        :param markdown_text: markdown текст отчёта
        :param title: заголовок документа
        :return: PDF как bytes
        """
        try:
            # Преобразуем markdown в HTML
            content_html = self.markdown_to_html(markdown_text)

            # Загружаем шаблон
            try:
                template = self.env.get_template("report.html")
                full_html = template.render(
                    title=title,
                    content=content_html,
                )
            except Exception:
                # Если шаблон не найден, используем встроенный
                full_html = self._get_default_template(title, content_html)

            # Создаём PDF
            html_doc = HTML(string=full_html)
            css = CSS(string=self._get_css())

            pdf_buffer = BytesIO()
            html_doc.write_pdf(pdf_buffer, stylesheets=[css])

            pdf_bytes = pdf_buffer.getvalue()
            logger.info(f"PDF сгенерирован: {len(pdf_bytes)} байт")

            return pdf_bytes

        except Exception as e:
            logger.exception("Ошибка генерации PDF")
            raise RuntimeError(f"Failed to generate PDF: {e}") from e

    def _get_default_template(self, title: str, content: str) -> str:
        """Возвращает HTML шаблон по умолчанию."""
        return f"""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
</head>
<body>
    <div class="container">
        {content}
    </div>
</body>
</html>
"""

    def _get_css(self) -> str:
        """Возвращает CSS стили для PDF."""
        return """
@page {
    size: A4;
    margin: 2cm;
    @top-center {
        content: "";
        border-bottom: 2px solid #1a5f7a;
        margin-bottom: 1cm;
    }
    @bottom-center {
        content: "Страница " counter(page) " из " counter(pages);
        font-size: 10px;
        color: #666;
        margin-top: 0.5cm;
    }
}

body {
    font-family: 'DejaVu Sans', 'Arial', sans-serif;
    font-size: 12px;
    line-height: 1.6;
    color: #333;
}

.container {
    max-width: 100%;
}

h1 {
    font-size: 28px;
    color: #1a5f7a;
    text-align: center;
    margin-bottom: 15px;
    font-weight: bold;
    letter-spacing: 1px;
}

h2 {
    font-size: 20px;
    color: #1a5f7a;
    margin-top: 30px;
    margin-bottom: 15px;
    border-bottom: 2px solid #2c7da0;
    padding-bottom: 8px;
    font-weight: bold;
}

h3 {
    font-size: 16px;
    color: #2c7da0;
    margin-top: 25px;
    margin-bottom: 12px;
    font-weight: bold;
}

p {
    margin: 10px 0;
    text-align: justify;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 15px 0;
    font-size: 11px;
}

th, td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: left;
}

th {
    background-color: #f8f9fa;
    font-weight: bold;
}

tr:nth-child(even) {
    background-color: #f8f9fa;
}

tr:hover {
    background-color: #e9ecef;
}

ul {
    margin: 10px 0;
    padding-left: 20px;
}

li {
    margin: 5px 0;
}

hr {
    border: none;
    border-top: 1px solid #ddd;
    margin: 20px 0;
}

strong {
    font-weight: bold;
}

em {
    font-style: italic;
    color: #666;
}

.footer {
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #ddd;
    font-size: 11px;
    color: #666;
    text-align: center;
}

.header-info {
    text-align: center;
    margin-bottom: 25px;
    color: #555;
    font-size: 13px;
}

.content-section {
    margin: 20px 0;
    padding: 15px;
    background-color: #f9f9f9;
    border-left: 4px solid #1a5f7a;
}
"""
