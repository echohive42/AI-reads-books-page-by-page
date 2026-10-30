"""How the reader turns a file into pages.

A PDF keeps its own pages, read with PyMuPDF, so every note can cite the printed page.

EPUB, Word (.docx), Markdown and text files have no fixed pages: their text reflows to
fit the screen. They are read with Python's standard library alone, then cut into
"reader pages" of about PAGE_CHARS characters, always at a paragraph break (or, inside
a very long paragraph, at the end of a sentence). The same text always gives the same
pages, so a reading can stop and carry on safely.
"""

from __future__ import annotations

import posixpath
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote
from xml.etree import ElementTree

PAGE_CHARS = 2_400  # about one printed book page

BOOK_FORMATS = {
    ".pdf": "pdf",
    ".epub": "epub",
    ".docx": "docx",
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
}
_KINDLE = ("Kindle and MOBI files can't be read directly. Convert the book to EPUB first, for example "
           "with the free Calibre app (calibre-ebook.com), then read the .epub. Only DRM-free books can be converted.")
CONVERT_FIRST = {
    ".mobi": _KINDLE,
    ".azw": _KINDLE,
    ".azw3": _KINDLE,
    ".kfx": _KINDLE,
    ".doc": "Old .doc files can't be read directly. Open it in Word (or Pages) and save it as .docx.",
}


class UnsupportedBook(ValueError):
    """The file can't be read: an unsupported type, a damaged file, or a DRM-protected book."""


def book_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in BOOK_FORMATS:
        return BOOK_FORMATS[suffix]
    if suffix in CONVERT_FIRST:
        raise UnsupportedBook(CONVERT_FIRST[suffix])
    raise UnsupportedBook(
        f"Can't read {suffix or 'files without an extension'}: the reader takes PDF, EPUB, Word (.docx), "
        "Markdown (.md) and text (.txt) files."
    )


def is_pdf(path: Path) -> bool:
    return path.suffix.lower() == ".pdf"


class TextPage:
    def __init__(self, text: str):
        self.text = text

    def get_text(self) -> str:
        return self.text


class TextBook:
    """A book with no fixed pages, cut into reader pages; it reads like a PDF document."""

    def __init__(self, pages: list[str]):
        self.pages = [TextPage(page) for page in pages]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def __getitem__(self, index: int) -> TextPage:
        return self.pages[index]

    def close(self) -> None:
        pass


def open_book(path: Path) -> tuple[Any, dict[str, Any]]:
    """The document to read, and how it was cut into pages (saved with the progress)."""
    kind = book_format(path)
    if kind == "pdf":
        import pymupdf  # PyMuPDF (the old `fitz` name can clash with an unrelated package)

        return pymupdf.open(path), {"format": "pdf"}
    try:
        text = TEXT_READERS[kind](path)
    except (zipfile.BadZipFile, KeyError, StopIteration, ElementTree.ParseError) as exc:
        raise UnsupportedBook(f"{path.name} doesn't look like a valid {kind.upper()} file ({exc}).") from exc
    return TextBook(split_into_pages(text)), {"format": kind, "page_chars": PAGE_CHARS}


# --- cutting text into reader pages ------------------------------------------------------------


def paragraphs(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def _sentence_pieces(paragraph: str, size: int) -> list[str]:
    """A paragraph longer than a page, in pieces of at most `size`, cut at sentence ends."""
    pieces = []
    for sentence in re.split(r"(?<=[.!?…])\s+", paragraph):
        while len(sentence) > size:  # one enormous sentence: cut at the last space that fits
            cut = sentence.rfind(" ", 0, size + 1)
            cut = cut if cut > 0 else size
            pieces.append(sentence[:cut].rstrip())
            sentence = sentence[cut:].lstrip()
        if sentence:
            pieces.append(sentence)
    return pieces


def split_into_pages(text: str, size: int = PAGE_CHARS) -> list[str]:
    """Book-size pages, filled paragraph by paragraph. The same text always gives the same pages."""
    pages: list[str] = []
    page = ""

    def add(piece: str, joiner: str) -> None:
        nonlocal page
        if page and len(page) + len(joiner) + len(piece) > size:
            pages.append(page)
            page = ""
        page = page + joiner + piece if page else piece

    for paragraph in paragraphs(text):
        if len(paragraph) <= size:
            add(paragraph, "\n\n")
            continue
        for number, piece in enumerate(_sentence_pieces(paragraph, size)):
            add(piece, " " if number else "\n\n")
    if page:
        pages.append(page)
    return pages


# --- reading each format to plain text ---------------------------------------------------------


def plain_text(path: Path) -> str:
    """Markdown and text are read exactly as written, headings and links included."""
    data = path.read_bytes()
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:  # older Windows and Project Gutenberg files
        return data.decode("cp1252", errors="replace")


_BLOCK_TAGS = {
    "p", "div", "br", "hr", "li", "ul", "ol", "dd", "dt", "blockquote", "pre", "section", "article",
    "aside", "header", "footer", "figure", "figcaption", "table", "tr", "td", "th",
    "h1", "h2", "h3", "h4", "h5", "h6",
}
_HIDDEN_TAGS = {"head", "title", "script", "style"}


class _HTMLText(HTMLParser):
    """The visible text of one XHTML chapter, one paragraph per block."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in _HIDDEN_TAGS:
            self.hidden += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n\n")

    def handle_startendtag(self, tag: str, attrs: Any) -> None:
        if tag in _BLOCK_TAGS:
            self.parts.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _HIDDEN_TAGS:
            self.hidden = max(0, self.hidden - 1)
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(re.sub(r"\s+", " ", data))

    def text(self) -> str:
        return "\n\n".join(paragraphs("".join(self.parts)))


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def epub_text(path: Path) -> str:
    """The chapters of an EPUB in reading order (its spine), as plain text."""
    with zipfile.ZipFile(path) as book:
        names = set(book.namelist())
        encrypted = set()
        if "META-INF/encryption.xml" in names:
            encryption = ElementTree.fromstring(book.read("META-INF/encryption.xml"))
            encrypted = {unquote(element.get("URI", "")) for element in encryption.iter()
                         if _local(element.tag) == "CipherReference"}
        container = ElementTree.fromstring(book.read("META-INF/container.xml"))
        opf_path = next(element.get("full-path") for element in container.iter()
                        if _local(element.tag) == "rootfile" and element.get("full-path"))
        package = ElementTree.fromstring(book.read(opf_path))
        base = posixpath.dirname(opf_path)
        manifest = {
            element.get("id"): element
            for element in package.iter() if _local(element.tag) == "item"
        }
        chapters = []
        for itemref in (element for element in package.iter() if _local(element.tag) == "itemref"):
            item = manifest.get(itemref.get("idref"))
            if item is None or itemref.get("linear") == "no":
                continue
            if "html" not in (item.get("media-type") or ""):
                continue
            name = posixpath.normpath(posixpath.join(base, unquote(item.get("href", "").split("#")[0])))
            if name not in names:
                continue
            if name in encrypted:  # encrypted fonts are harmless; an encrypted chapter means DRM
                raise UnsupportedBook(f"{path.name} is DRM-protected, so it can't be read. "
                                      "Use a DRM-free copy.")
            parser = _HTMLText()
            parser.feed(book.read(name).decode("utf-8", errors="replace"))
            parser.close()
            chapters.append(parser.text())
    return "\n\n".join(chapter for chapter in chapters if chapter)


_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def docx_text(path: Path) -> str:
    """The paragraphs of a Word document's body, as plain text."""
    with zipfile.ZipFile(path) as document:
        body = ElementTree.fromstring(document.read("word/document.xml"))
    lines = []
    for paragraph in body.iter(f"{_W}p"):
        parts = []
        for element in paragraph.iter():
            if element.tag == f"{_W}t":
                parts.append(element.text or "")
            elif element.tag == f"{_W}tab":
                parts.append("\t")
            elif element.tag in (f"{_W}br", f"{_W}cr"):
                parts.append("\n")
        line = "".join(parts).strip()
        if line:
            lines.append(line)
    return "\n\n".join(lines)


TEXT_READERS: dict[str, Callable[[Path], str]] = {
    "epub": epub_text,
    "docx": docx_text,
    "markdown": plain_text,
    "text": plain_text,
}
