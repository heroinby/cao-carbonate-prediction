"""Собрать и выполнить Jupyter-ноутбук из .py-файла с маркерами ячеек.

Формат источника: ячейки разделяются строками ``# %%`` (код) и ``# %% [md]``
(markdown, каждая строка начинается с ``# ``).

Запуск: python tools/make_notebook.py notebooks_src/01_eda.py notebooks/01_eda.ipynb
"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def parse(text: str):
    """Разрезать текст .py на ячейки ноутбука."""
    cells, kind, buf = [], None, []  # тип текущей ячейки и накопленные строки

    def flush():
        # закрыть текущую ячейку
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if kind == "md":
            # в markdown-ячейках у строк снимаем "# "
            body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.split("\n"))
            cells.append(nbformat.v4.new_markdown_cell(body))
        elif body:
            cells.append(nbformat.v4.new_code_cell(body))

    for line in text.split("\n"):
        # "# %%" начинает новую ячейку, остальное копим
        if line.startswith("# %% [md]"):
            flush(); kind, buf = "md", []
        elif line.startswith("# %%"):
            flush(); kind, buf = "code", []
        else:
            buf.append(line)
    flush()
    return cells


def main(src: str, dst: str) -> None:
    """Собрать ноутбук из src, выполнить и сохранить в dst."""
    nb = nbformat.v4.new_notebook()
    nb.cells = parse(Path(src).read_text(encoding="utf-8"))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    # реально выполняем, чтобы в файле остались таблицы и графики
    NotebookClient(nb, timeout=3000, kernel_name="python3", resources={"metadata": {"path": "."}}).execute()
    nbformat.write(nb, dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
