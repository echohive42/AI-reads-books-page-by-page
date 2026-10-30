#!/usr/bin/env python3
"""AI reads books page by page.

The AI reads a PDF the way a person studies a book: one page at a time. With each page
it sees only that page, the last lines of the page before, and its own notes so far.
It writes down what it learned (every note tagged with its page), and every few pages
it writes down what it understands so far. At the end it pulls the whole book together.

Run it on the OpenAI API, or on the Claude, ChatGPT (Codex) or Grok plan you already
have. See `python read_books.py --help`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from pydantic import BaseModel, ValidationError
from termcolor import colored

from backends import (
    BACKENDS,
    DEFAULT_TIMEOUT,
    Backend,
    BackendError,
    SetupError,
    TransientError,
    UsageLimitError,
    make_backend,
)

# source for the infinite descent book: https://infinitedescent.xyz/dl/infdesc.pdf

DEFAULT_PDF = Path("meditations.pdf")
DEFAULT_BASE_DIR = Path("book_analysis")
DEFAULT_INTERVAL = 20         # write "what I understand so far" every N pages (0 = never)
DEFAULT_CONTEXT_WINDOW = 8    # how many of its most recent notes the reader sees with each page
RECENT_PAGE_GISTS = 5         # one-line memories of the last few pages
PREVIOUS_PAGE_TAIL = 600      # characters from the end of the previous page, for sentences that run over
UNDERSTANDING_CHARS = 6_000   # how much of the running understanding goes along with each page
SUMMARY_CHUNK_SIZE = 300      # most notes in one summary request
SUMMARY_CHUNK_CHARS = 100_000  # most characters of notes in one summary request
SUMMARY_MAX_ITEM_CHARS = 8_000


class PageContent(BaseModel):
    has_content: bool
    page_gist: str = ""
    knowledge: list[str] = []


# The same shape as PageContent, written out so every backend gets an identical, strict schema.
PAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "has_content": {
            "type": "boolean",
            "description": "false for pages to skip (contents, index, blank, copyright, bibliography...)",
        },
        "page_gist": {
            "type": "string",
            "description": "one sentence: what this page adds to the book so far (empty when skipped)",
        },
        "knowledge": {
            "type": "array",
            "items": {"type": "string"},
            "description": "new, standalone, learnable points from this page (empty when skipped)",
        },
    },
    "required": ["has_content", "page_gist", "knowledge"],
    "additionalProperties": False,
}

PAGE_SYSTEM_PROMPT = """You are reading a book one page at a time, the way a careful person studies it. \
You see only the current page, the last lines of the page before it, and your own notes from the pages you \
have read so far. You never see the rest of the book.

SKIP the page if it contains:
- Table of contents
- Chapter listings
- Index pages
- Blank pages
- Copyright information
- Publishing details
- References or bibliography
- Acknowledgments

DO extract knowledge if the page contains:
- Preface content that explains important concepts
- Actual educational content
- Key definitions and concepts
- Important arguments or theories
- Examples and case studies
- Significant findings or conclusions
- Methodologies or frameworks
- Critical analyses or interpretations

For valid content:
- Set has_content to true
- Write page_gist: one sentence on what this page adds to the book so far
- Extract detailed, learnable knowledge points; each one must make sense on its own
- Include important quotes or key statements
- Capture examples with their context
- Preserve technical terms and definitions
- Use your notes to connect ideas: when this page continues or answers something from an earlier page, \
say so in the point
- Do not repeat a point that is already in your notes; add only what is new on this page

For pages to skip:
- Set has_content to false
- Return an empty page_gist and an empty knowledge list

Answer only with the JSON object."""

MARKDOWN_STYLE = """Use markdown formatting:
- ## for main sections
- ### for subsections
- Bullet points for lists
- `code blocks` for any code or formulas
- **bold** for emphasis
- *italic* for terminology
- > blockquotes for important notes

Return only the markdown, nothing else. Do not say 'here is the summary' or anything like that before or after."""

UNDERSTANDING_SYSTEM_PROMPT = f"""You are reading a book page by page and keeping track of what you understand so far.
You get what you understood before (if anything) and the notes you took on the pages since, each tagged with \
its page number. Write your updated understanding of the book so far:
- Keep what still matters from before, and revise anything the new pages changed or deepened
- Show how the ideas connect and develop across the pages, with page references like (p. 34)
- End with a short "### Threads to follow" list: questions or ideas you expect the book to develop next
- Write only from the notes; do not add outside knowledge about the book
- Stay concise but detailed: at most about 1,000 words

{MARKDOWN_STYLE}"""

PART_SYSTEM_PROMPT = f"""You have finished reading a book page by page. Here are your notes from one part of it, \
each tagged with its page number. Summarise this part in a concise but detailed way, keeping the key ideas, \
definitions, examples and quotes, with page references like (p. 34). Write only from the notes.

{MARKDOWN_STYLE}"""

FINAL_SYSTEM_PROMPT = f"""You have finished reading a book page by page. Create a comprehensive summary that pulls \
the whole book together, in a concise but detailed way:
- The main ideas, and how they build on each other from the first page you read to the last
- Key definitions, arguments, examples and important quotes, with page references like (p. 34)
- Write only from your notes and your running understanding; do not add outside knowledge about the book

{MARKDOWN_STYLE}"""


@dataclass
class PageReading:
    has_content: bool
    gist: str
    points: list[str]
    usage: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0


@dataclass
class APIResult:
    value: Any
    usage: dict[str, int]


def empty_state() -> dict[str, Any]:
    return {
        "version": 2,
        "document_id": None,
        "last_completed_page": 0,
        "knowledge": [],
        "reading_log": [],
        "understanding": None,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "reading_seconds": 0.0,
        "last_interval_summary_page": 0,
        "final_summary_page": 0,
    }


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_state()
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    defaults = empty_state()
    for key, value in defaults.items():
        state.setdefault(key, value)
    # Preserve knowledge from early versions of the script instead of deleting it.
    state["knowledge"] = [
        item if isinstance(item, dict) else {"page": None, "text": str(item)}
        for item in state["knowledge"]
    ]
    return state


def save_state(path: Path, state: dict[str, Any]) -> None:
    """Atomically persist progress only after a page has succeeded."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def setup_directories(
    source_pdf: Path | str,
    base_dir: Path = DEFAULT_BASE_DIR,
    reset: bool = False,
) -> Path:
    """Create output directories; preserve existing output unless reset is explicit."""
    source_pdf = Path(source_pdf)
    base_dir = Path(base_dir)
    pdf_dir = base_dir / "pdfs"
    knowledge_dir = base_dir / "knowledge_bases"
    summaries_dir = base_dir / "summaries"
    for directory in (pdf_dir, knowledge_dir, summaries_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if not source_pdf.exists():
        raise FileNotFoundError(f"PDF file not found: {source_pdf}")
    destination = pdf_dir / source_pdf.name
    if reset:
        selected_outputs = [knowledge_dir / f"{source_pdf.stem}_knowledge.json"]
        summary_name = re.compile(
            rf"^{re.escape(source_pdf.stem)}_(?:final|interval_\d{{5}})\.md$"
        )
        selected_outputs.extend(
            entry
            for entry in summaries_dir.iterdir()
            if entry.is_file()
            and summary_name.fullmatch(entry.name)
        )
        if source_pdf.resolve() != destination.resolve():
            selected_outputs.append(destination)
        for entry in selected_outputs:
            if entry.exists():
                entry.unlink()
    if source_pdf.resolve() != destination.resolve():
        if destination.exists() and file_digest(source_pdf) != file_digest(destination):
            raise RuntimeError(
                f"PDF changed for existing output set: {source_pdf.name}; rerun with --reset"
            )
        if not destination.exists():
            shutil.copy2(source_pdf, destination)
            print(colored(f"📄 Copied PDF to: {destination}", "green"))
    return destination


def file_digest(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def add_usage(total: dict[str, int], usage: dict[str, int]) -> None:
    for name, value in usage.items():
        total[name] = total.get(name, 0) + value


def call_with_retry(
    operation: Callable[[], Any],
    max_retries: int = 3,
    backoff: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Callable[[Exception, float], None] | None = None,
) -> Any:
    """Retry timeouts, network errors and malformed answers; never retry setup or usage-limit errors."""
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    for attempt in range(max_retries + 1):
        try:
            return operation()
        except (ConnectionError, TimeoutError, TransientError) as exc:
            if attempt == max_retries:
                raise
            delay = backoff * (2**attempt)
            if on_retry:
                on_retry(exc, delay)
            sleep(delay)
    raise AssertionError("unreachable")


def _say_retry(exc: Exception, delay: float) -> None:
    print(colored(f"   🔁 {exc} (trying again in {delay:.0f} s)", "yellow"))


def _note_line(item: dict[str, Any], limit: int | None = None) -> str:
    line = f"[page {item.get('page') or '?'}] {item['text']}"
    return line[:limit] if limit else line


def page_prompt(
    page_text: str,
    page_number: int,
    *,
    page_count: int | None = None,
    book_name: str | None = None,
    recent_knowledge: Sequence[dict[str, Any]] = (),
    recent_pages: Sequence[dict[str, Any]] = (),
    understanding: dict[str, Any] | None = None,
    previous_page_end: str = "",
) -> str:
    """Everything the reader sees for one page: the page itself and bounded notes from before."""
    of_total = f" of {page_count}" if page_count else ""
    parts = [f"Book: {book_name}, page {page_number}{of_total}" if book_name else f"Page {page_number}{of_total}"]
    if understanding and understanding.get("text"):
        text = understanding["text"]
        if len(text) > UNDERSTANDING_CHARS:
            text = text[:UNDERSTANDING_CHARS] + "\n[...]"
        parts.append(
            f"What you understood after page {understanding.get('page')}:\n<understanding>\n{text}\n</understanding>"
        )
    gists = [entry for entry in recent_pages if entry.get("gist")]
    if gists:
        parts.append(
            "What the last pages you read were about:\n"
            + "\n".join(f"[page {entry['page']}] {entry['gist']}" for entry in gists)
        )
    notes = "\n".join(_note_line(item) for item in recent_knowledge) or "(none yet)"
    parts.append(f"Your most recent notes:\n{notes}")
    if previous_page_end.strip():
        parts.append(
            f"The last lines of page {page_number - 1}:\n<previous_page_end>\n{previous_page_end.strip()}\n"
            "</previous_page_end>"
        )
    parts.append(f"Now read page {page_number}:\n<page>\n{page_text.strip() or '(this page has no text)'}\n</page>")
    return "\n\n".join(parts)


def extract_page_knowledge(
    backend: Backend,
    page_text: str,
    recent_knowledge: Sequence[dict[str, Any]],
    page_number: int,
    *,
    page_count: int | None = None,
    book_name: str | None = None,
    recent_pages: Sequence[dict[str, Any]] = (),
    understanding: dict[str, Any] | None = None,
    previous_page_end: str = "",
    max_retries: int = 3,
    backoff: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Callable[[Exception, float], None] | None = None,
) -> PageReading:
    """Read one page: one model call that sees only this page and the reader's bounded notes."""
    prompt = page_prompt(
        page_text,
        page_number,
        page_count=page_count,
        book_name=book_name,
        recent_knowledge=recent_knowledge,
        recent_pages=recent_pages,
        understanding=understanding,
        previous_page_end=previous_page_end,
    )

    def read() -> tuple[PageContent, Any]:
        reply = backend.ask(PAGE_SYSTEM_PROMPT, prompt, PAGE_SCHEMA)
        try:
            return PageContent.model_validate(reply.value), reply
        except ValidationError as exc:
            raise TransientError(f"The page notes did not match the schema: {exc.errors()[0]['msg']}") from exc

    parsed, reply = call_with_retry(read, max_retries=max_retries, backoff=backoff, sleep=sleep, on_retry=on_retry)
    points = [point.strip() for point in parsed.knowledge if point.strip()] if parsed.has_content else []
    gist = parsed.page_gist.strip() if parsed.has_content else ""
    return PageReading(bool(parsed.has_content), gist, points, reply.usage, reply.seconds)


def chunk_items(items: Sequence[str], max_items: int, max_chars: int) -> list[list[str]]:
    """Split notes into requests that respect both an item count and a character budget."""
    chunks: list[list[str]] = []
    current: list[str] = []
    size = 0
    for item in items:
        if current and (len(current) >= max_items or size + len(item) > max_chars):
            chunks.append(current)
            current, size = [], 0
        current.append(item)
        size += len(item)
    if current:
        chunks.append(current)
    return chunks


def _ask_text(backend: Backend, system: str, prompt: str, *, max_retries: int, backoff: float,
              sleep: Callable[[float], None], on_retry: Callable[[Exception, float], None] | None) -> Any:
    def ask() -> Any:
        reply = backend.ask(system, prompt)
        if not str(reply.value).strip():
            raise TransientError("The summary came back empty")
        return reply

    return call_with_retry(ask, max_retries=max_retries, backoff=backoff, sleep=sleep, on_retry=on_retry)


def update_understanding(
    backend: Backend,
    understanding: dict[str, Any] | None,
    new_knowledge: Sequence[dict[str, Any]],
    page_number: int,
    *,
    book_name: str | None = None,
    page_count: int | None = None,
    chunk_size: int = SUMMARY_CHUNK_SIZE,
    chunk_chars: int = SUMMARY_CHUNK_CHARS,
    max_item_chars: int = SUMMARY_MAX_ITEM_CHARS,
    max_retries: int = 3,
    backoff: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Callable[[Exception, float], None] | None = None,
) -> APIResult:
    """Fold the notes since the last update into "what I understand so far after page N".

    Each request holds the previous understanding plus a bounded batch of new notes, so it
    stays the same size however long the book is.
    """
    text = (understanding or {}).get("text") or ""
    since = (understanding or {}).get("page")
    usage: dict[str, int] = {}
    lines = [_note_line(item, max_item_chars) for item in new_knowledge]
    of_total = f" of {page_count}" if page_count else ""
    for chunk in chunk_items(lines, chunk_size, max(chunk_chars, 2 * max_item_chars)):
        before = (
            f"What you understood after page {since}:\n<understanding>\n{text}\n</understanding>"
            if text
            else "This is your first look back: you have not written down your understanding yet."
        )
        prompt = (
            f"Book: {book_name or 'this book'}. You have now read up to page {page_number}{of_total}.\n\n"
            f"{before}\n\nYour notes from the pages since:\n<notes>\n" + "\n".join(chunk) + "\n</notes>\n\n"
            f"Write what you understand so far, after page {page_number}."
        )
        reply = _ask_text(backend, UNDERSTANDING_SYSTEM_PROMPT, prompt, max_retries=max_retries,
                          backoff=backoff, sleep=sleep, on_retry=on_retry)
        text = str(reply.value).strip()
        since = page_number
        add_usage(usage, reply.usage)
    return APIResult(text, usage)


def final_summary(
    backend: Backend,
    knowledge: Sequence[dict[str, Any]],
    *,
    understanding: dict[str, Any] | None = None,
    book_name: str | None = None,
    pages: tuple[int, int] | None = None,
    chunk_size: int = SUMMARY_CHUNK_SIZE,
    chunk_chars: int = SUMMARY_CHUNK_CHARS,
    max_item_chars: int = SUMMARY_MAX_ITEM_CHARS,
    max_retries: int = 3,
    backoff: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Callable[[Exception, float], None] | None = None,
) -> APIResult:
    """Pull the whole book together from every page-tagged note.

    Most books fit in one request. A very long book is summarised part by part first
    (map/reduce), so no request ever grows past the chunk bounds.
    """
    if chunk_size < 2:
        raise ValueError("summary chunk size must be >= 2")
    if max_item_chars < 100:
        raise ValueError("max_item_chars must be >= 100")
    if not knowledge:
        return APIResult("", {})
    chunk_chars = max(chunk_chars, 2 * max_item_chars)  # every request holds at least two items
    asked = dict(max_retries=max_retries, backoff=backoff, sleep=sleep, on_retry=on_retry)
    usage: dict[str, int] = {}
    layer = [_note_line(item, max_item_chars) for item in knowledge]
    label = "notes"
    while True:
        chunks = chunk_items(layer, chunk_size, chunk_chars)
        if len(chunks) == 1:
            break
        print(colored(f"   🧩 Summarising {len(chunks)} parts of the book first...", "cyan"))
        next_layer = []
        for chunk in chunks:
            prompt = f"Book: {book_name or 'this book'}.\n\nYour {label}:\n<notes>\n" + "\n".join(chunk) + "\n</notes>"
            reply = _ask_text(backend, PART_SYSTEM_PROMPT, prompt, **asked)
            add_usage(usage, reply.usage)
            next_layer.append(f"<part>\n{str(reply.value).strip()[:max_item_chars]}\n</part>")
        layer, label = next_layer, "summaries of each part"
    span = f"pages {pages[0]}–{pages[1]}" if pages else "the pages you read"
    guide = ""
    if understanding and understanding.get("text"):
        guide = (
            f"Your running understanding (written after page {understanding.get('page')}):\n"
            f"<understanding>\n{understanding['text']}\n</understanding>\n\n"
        )
    prompt = (
        f"Book: {book_name or 'this book'}. You have finished reading {span}.\n\n{guide}"
        f"Your {label}:\n<notes>\n" + "\n".join(chunks[0]) + "\n</notes>\n\n"
        "Write the final summary that pulls the whole book together."
    )
    reply = _ask_text(backend, FINAL_SYSTEM_PROMPT, prompt, **asked)
    add_usage(usage, reply.usage)
    return APIResult(str(reply.value).strip(), usage)


def pymupdf_ocr(page: Any) -> str:
    try:
        text_page = page.get_textpage_ocr()
        return page.get_text(textpage=text_page)
    except Exception as exc:
        raise RuntimeError(
            "OCR dependency unavailable. Install Tesseract and its language data, "
            "or use --ocr off / inject a custom OCR engine."
        ) from exc


def extract_text(
    page: Any,
    *,
    ocr: str = "off",
    ocr_engine: Callable[[Any], str] | None = None,
) -> str:
    if ocr not in {"off", "auto", "always"}:
        raise ValueError("ocr must be off, auto, or always")
    text = page.get_text()
    if ocr == "off" or (ocr == "auto" and text.strip()):
        return text
    if ocr_engine is None:
        raise RuntimeError(
            "OCR fallback requested but no OCR dependency/engine is available; "
            "install Tesseract for PyMuPDF OCR or provide an OCR engine."
        )
    return ocr_engine(page)


def save_summary(
    summaries_dir: Path,
    book_name: str,
    summary: str,
    *,
    kind: str,
    page_number: int,
    page_count: int | None = None,
    first_page: int = 1,
    written_by: str | None = None,
) -> Path | None:
    if not summary:
        return None
    summaries_dir.mkdir(parents=True, exist_ok=True)
    path = summaries_dir / f"{Path(book_name).stem}_{kind}.md"
    of_total = f" of {page_count}" if page_count else ""
    if kind == "final":
        title = f"## The whole book, pulled together (pages {first_page}–{page_number}{of_total})"
    else:
        title = f"## What I understand so far, after page {page_number}{of_total}"
    by = f" by {written_by}" if written_by else ""
    content = (
        f"# Book Analysis: {book_name}\n{title}\n\n"
        f"Generated: {datetime.now().isoformat(timespec='seconds')}{by}  \n"
        f"Processed through page: {page_number}\n\n{summary}\n\n"
        "---\n*Read page by page with [AI reads books](https://github.com/echohive42/AI-reads-books-page-by-page)*\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


def _unwrap_result(result: Any) -> tuple[Any, dict[str, int]]:
    if isinstance(result, APIResult):
        return result.value, result.usage
    return result, {}


def _show_points(points: Sequence[str], shown: int = 3, width: int = 110) -> None:
    for point in points[:shown]:
        print(colored("      • " + textwrap.shorten(point, width=width, placeholder="…"), "white"))
    if len(points) > shown:
        print(colored(f"      … and {len(points) - shown} more", "white"))


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def run_analysis(
    document: Any,
    reader: Backend,
    state_path: Path,
    summaries_dir: Path,
    *,
    summary_backend: Backend | None = None,
    book_name: str | None = None,
    max_pages: int | None = None,
    start_page: int = 1,
    resume: bool = True,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    interval: int | None = DEFAULT_INTERVAL,
    summary_chunk_size: int = SUMMARY_CHUNK_SIZE,
    summary_chunk_chars: int = SUMMARY_CHUNK_CHARS,
    summary_max_item_chars: int = SUMMARY_MAX_ITEM_CHARS,
    ocr: str = "off",
    ocr_engine: Callable[[Any], str] | None = None,
    max_retries: int = 3,
    backoff: float = 1.0,
    document_id: str | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if context_window < 0:
        raise ValueError("context_window must be >= 0")
    if interval is not None and interval < 0:
        raise ValueError("interval must be >= 0")
    if start_page < 1:
        raise ValueError("start_page must be >= 1")
    summarizer = summary_backend or reader
    state = load_state(state_path) if resume else empty_state()
    if not resume:
        # Persist the fresh checkpoint before any API call. If page 1 fails,
        # a later default resume must not see stale progress from an older run.
        save_state(state_path, state)
    if resume and document_id and state.get("document_id") not in (None, document_id):
        raise RuntimeError("Cannot resume: state belongs to a different PDF; use --reset")
    if document_id:
        state["document_id"] = document_id
    book_name = book_name or state_path.name.removesuffix("_knowledge.json") + ".pdf"
    page_count = document.page_count
    limit = page_count if max_pages is None else min(page_count, start_page - 1 + max_pages)
    start = min(max(state["last_completed_page"], start_page - 1), limit)
    retry = dict(max_retries=max_retries, backoff=backoff, sleep=sleep, on_retry=_say_retry)
    bounds = dict(chunk_size=summary_chunk_size, chunk_chars=summary_chunk_chars, max_item_chars=summary_max_item_chars)

    def generate_interval_summary(page_number: int) -> None:
        recent = [
            item
            for item in state["knowledge"]
            if state["last_interval_summary_page"] < (item.get("page") or 0) <= page_number
        ]
        if recent:
            print(colored(f"\n📊 Progress: {page_number}/{page_count} pages read", "cyan"))
            print(colored(f"🤔 Updating what I understand so far ({len(recent)} new notes)...", "cyan"))
            result = update_understanding(
                summarizer,
                state["understanding"],
                recent,
                page_number,
                book_name=book_name,
                page_count=page_count,
                **bounds,
                **retry,
            )
            summary, summary_usage = _unwrap_result(result)
            path = save_summary(
                summaries_dir,
                book_name,
                summary,
                kind=f"interval_{page_number:05d}",
                page_number=page_number,
                page_count=page_count,
                written_by=summarizer.label(),
            )
            if summary:
                state["understanding"] = {"page": page_number, "text": summary}
                print(colored(f"✨ What I understand so far, saved to: {path}", "green"))
            add_usage(state["usage"], summary_usage)
        state["last_interval_summary_page"] = page_number
        save_state(state_path, state)

    # A page is saved before its interval summary. If that summary failed, retry it
    # before processing more pages so the interval cannot be skipped on resume.
    if interval:
        next_due = state["last_interval_summary_page"] + interval
        while next_due <= state["last_completed_page"] and next_due < limit:
            generate_interval_summary(next_due)
            next_due += interval

    if start < limit:
        span = f"page {limit}" if limit == start + 1 else f"pages {start + 1}–{limit}"
        print(colored(f"\n📚 Reading {span} of {page_count}...", "cyan"))
    elif state["last_completed_page"] >= limit:
        print(colored(f"\n📚 Already read through page {state['last_completed_page']}; nothing new to read.", "cyan"))
    for page_index in range(start, limit):
        page_number = page_index + 1
        print(colored(f"\n📖 Reading page {page_number}/{page_count}...", "yellow"))
        page_text = extract_text(document[page_index], ocr=ocr, ocr_engine=ocr_engine)
        if not page_text.strip():
            # Nothing to read on this page (blank, or a scan without OCR): no model call needed.
            reading = PageReading(False, "", [])
        else:
            previous_end = document[page_index - 1].get_text()[-PREVIOUS_PAGE_TAIL:] if page_index > 0 else ""
            reading = extract_page_knowledge(
                reader,
                page_text,
                state["knowledge"][-context_window:] if context_window else [],
                page_number,
                page_count=page_count,
                book_name=book_name,
                recent_pages=state["reading_log"][-RECENT_PAGE_GISTS:],
                understanding=state["understanding"],
                previous_page_end=previous_end,
                **retry,
            )
        # Update progress only after extraction returns successfully.
        existing_texts = {_normalise(item["text"]) for item in state["knowledge"]}
        new_points = []
        for point in reading.points:
            if _normalise(point) not in existing_texts:
                state["knowledge"].append({"page": page_number, "text": point})
                existing_texts.add(_normalise(point))
                new_points.append(point)
        state["reading_log"].append(
            {
                "page": page_number,
                "gist": reading.gist,
                "points": len(new_points),
                "seconds": round(reading.seconds, 1),
            }
        )
        add_usage(state["usage"], reading.usage)
        state["reading_seconds"] = round(state["reading_seconds"] + reading.seconds, 1)
        state["last_completed_page"] = page_number
        save_state(state_path, state)

        took = f" · {reading.seconds:.1f} s" if reading.seconds else ""
        if reading.has_content and new_points:
            if reading.gist:
                print(colored("   💭 " + textwrap.shorten(reading.gist, width=120, placeholder="…"), "magenta"))
            print(colored(
                f"   ✅ Found {len(new_points)} new knowledge points ({len(state['knowledge'])} so far){took}",
                "green",
            ))
            _show_points(new_points)
        elif reading.has_content:
            print(colored(f"   ✅ Nothing new on this page beyond my notes{took}", "green"))
        else:
            print(colored(f"   ⏭️  Skipping page (no relevant content){took}", "yellow"))
        print(colored(f"   💾 Saved knowledge base ({len(state['knowledge'])} items)", "blue"))

        if interval and page_number % interval == 0 and page_number < limit:
            generate_interval_summary(page_number)

    # `limit` is the effective final page, including when max_pages exceeds PDF length.
    if (
        state["knowledge"]
        and limit > 0
        and state["last_completed_page"] == limit
        and state["final_summary_page"] != limit
    ):
        print(colored(f"\n📊 Final page ({limit}/{page_count}) read", "cyan"))
        print(colored("🤔 Pulling the whole book together...", "cyan"))
        first_read = min((item["page"] for item in state["knowledge"] if item.get("page")), default=start_page)
        result = final_summary(
            summarizer,
            state["knowledge"],
            understanding=state["understanding"],
            book_name=book_name,
            pages=(first_read, limit),
            **bounds,
            **retry,
        )
        summary, summary_usage = _unwrap_result(result)
        path = save_summary(
            summaries_dir,
            book_name,
            summary,
            kind="final",
            page_number=limit,
            page_count=page_count,
            first_page=first_read,
            written_by=summarizer.label(),
        )
        add_usage(state["usage"], summary_usage)
        state["final_summary_page"] = limit
        save_state(state_path, state)
        if path:
            print(colored(f"✨ Final summary saved to: {path}", "green"))
    elif not state["knowledge"] and state["last_completed_page"] == limit and limit > 0:
        print(colored("\n⚠️  Skipping final summary: no knowledge points collected yet", "yellow"))
    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "An AI reads your PDF one page at a time, takes page-tagged notes, writes what it "
            "understands so far every few pages, and pulls the whole book together at the end."
        ),
        epilog=(
            "examples:\n"
            "  python read_books.py                                  # meditations.pdf on the OpenAI API\n"
            "  python read_books.py mybook.pdf --backend claude      # on your Claude plan\n"
            "  python read_books.py mybook.pdf --backend codex       # on your ChatGPT plan\n"
            "  python read_books.py mybook.pdf --backend grok        # on your Grok plan\n"
            "  python read_books.py mybook.pdf --backend anthropic   # with ANTHROPIC_API_KEY (also: gemini, xai, openrouter)\n"
            "  python read_books.py --start-page 31 --test-pages 3   # a quick 3-page taste"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pdf", type=Path, nargs="?", default=DEFAULT_PDF,
                        help=f"PDF file to read (default: {DEFAULT_PDF})")
    reading = parser.add_argument_group("who reads")
    reading.add_argument("--backend", choices=tuple(BACKENDS), default="openai",
                         help="openai, anthropic, gemini, xai, openrouter = an API key; claude, codex, grok = the CLI "
                              "and plan you already have (default: openai)")
    reading.add_argument("--model", default=None,
                         help="the model for pages and summaries (default: each backend's own; on openai "
                              f"{BACKENDS['openai'].page_model} reads and {BACKENDS['openai'].summary_model} summarises)")
    reading.add_argument("--summary-backend", choices=tuple(BACKENDS), default=None,
                         help="who writes the summaries (default: same as --backend)")
    reading.add_argument("--summary-model", "--analysis-model", dest="summary_model", default=None,
                         help="a different model for the summaries")
    reading.add_argument("--base-url", default=None,
                         help="send the openai backend's requests to another OpenAI-compatible endpoint")
    reading.add_argument("--effort", default=None,
                         help="reasoning effort passed to the model, e.g. low, medium, high (default: the model's own)")
    reading.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="seconds allowed per model call")
    pages = parser.add_argument_group("what to read")
    pages.add_argument("--start-page", type=int, default=1, help="first page to read, e.g. to skip front matter")
    pages.add_argument("--test-pages", type=int, default=None,
                       help="read at most N pages from the start page (default: the whole book)")
    pages.add_argument("--interval", type=int, default=DEFAULT_INTERVAL,
                       help=f"write 'what I understand so far' every N pages; 0 = never (default: {DEFAULT_INTERVAL})")
    pages.add_argument("--context-window", type=int, default=DEFAULT_CONTEXT_WINDOW,
                       help=f"recent notes the reader sees with each page (default: {DEFAULT_CONTEXT_WINDOW})")
    pages.add_argument("--ocr", choices=("off", "auto", "always"), default="off",
                       help="OCR scanned pages with Tesseract (default: off)")
    output = parser.add_argument_group("output and recovery")
    output.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    output.add_argument("--summary-chunk-size", type=int, default=SUMMARY_CHUNK_SIZE,
                        help="most notes in one summary request")
    output.add_argument("--summary-max-item-chars", type=int, default=SUMMARY_MAX_ITEM_CHARS)
    output.add_argument("--max-retries", type=int, default=3)
    output.add_argument("--backoff", type=float, default=1.0)
    output.add_argument("--reset", action="store_true", help="delete this book's saved notes and summaries first")
    resume_group = output.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true", default=True,
                              help="carry on from the last page read (default)")
    resume_group.add_argument("--no-resume", dest="resume", action="store_false")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.test_pages is not None and args.test_pages <= 0:
        parser.error("--test-pages must be > 0")
    if args.start_page < 1:
        parser.error("--start-page must be >= 1")
    if args.interval < 0:
        parser.error("--interval must be >= 0")
    if args.context_window < 0:
        parser.error("--context-window must be >= 0")
    if args.summary_chunk_size < 2:
        parser.error("--summary-chunk-size must be >= 2")
    if args.summary_max_item_chars < 100:
        parser.error("--summary-max-item-chars must be >= 100")
    if args.max_retries < 0:
        parser.error("--max-retries must be >= 0")
    if args.timeout <= 0:
        parser.error("--timeout must be > 0")
    if args.base_url and "openai" not in (args.backend, args.summary_backend):
        parser.error("--base-url only works with --backend openai (or --summary-backend openai)")


def make_backends(args: argparse.Namespace) -> tuple[Backend, Backend]:
    reader = make_backend(args.backend, args.model, args.effort, args.timeout, args.base_url)
    summary_name = args.summary_backend or args.backend
    summary_model = args.summary_model
    if summary_model is None:
        # --model covers the summaries too when the same backend writes them.
        summary_model = args.model if (summary_name == args.backend and args.model) else BACKENDS[summary_name].summary_model
    summarizer = make_backend(summary_name, summary_model, args.effort, args.timeout, args.base_url)
    return reader, summarizer


def print_banner(book: Path, page_count: int, reader: Backend, summarizer: Backend, args: argparse.Namespace) -> None:
    last = page_count if args.test_pages is None else min(page_count, args.start_page - 1 + args.test_pages)
    every = f"every {args.interval} pages" if args.interval else "off"
    span = f"page {last}" if last == args.start_page else f"pages {args.start_page}–{last}"
    print(colored(f"""
📚 AI reads books, page by page 📚
---------------------------------
   Book:          {book.name} ({page_count} pages)
   Reading:       {span}, one page per model call
   Reader:        {reader.label()}
   Summaries:     {summarizer.label()}
   Understanding: {every}
   Output:        {args.base_dir}/
""", "cyan"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    try:
        reader, summarizer = make_backends(args)
        reader.check()
        summarizer.check()
    except SetupError as exc:
        print(colored(f"🔑 {exc}", "red"))
        return 2
    try:
        pdf_path = setup_directories(args.pdf, args.base_dir, reset=args.reset)
    except (FileNotFoundError, RuntimeError) as exc:
        print(colored(f"❌ {exc}", "red"))
        return 2
    import pymupdf  # PyMuPDF (the old `fitz` name can clash with an unrelated package)

    stem = args.pdf.stem
    state_path = args.base_dir / "knowledge_bases" / f"{stem}_knowledge.json"
    summaries_dir = args.base_dir / "summaries"
    document = pymupdf.open(pdf_path)
    print_banner(args.pdf, document.page_count, reader, summarizer, args)
    started = time.monotonic()
    try:
        state = run_analysis(
            document,
            reader,
            state_path,
            summaries_dir,
            summary_backend=summarizer,
            book_name=args.pdf.name,
            max_pages=args.test_pages,
            start_page=args.start_page,
            resume=args.resume and not args.reset,
            context_window=args.context_window,
            interval=args.interval or None,
            summary_chunk_size=args.summary_chunk_size,
            summary_max_item_chars=args.summary_max_item_chars,
            ocr=args.ocr,
            ocr_engine=pymupdf_ocr if args.ocr != "off" else None,
            max_retries=args.max_retries,
            backoff=args.backoff,
            document_id=file_digest(pdf_path),
        )
    except KeyboardInterrupt:
        print(colored(f"\n⏸️  Paused. {_progress_note(state_path)} Run the same command again to carry on.", "yellow"))
        return 130
    except UsageLimitError as exc:
        print(colored(f"\n⏸️  {exc}\n   {_progress_note(state_path)} Run the same command again later to carry on.",
                      "yellow"))
        return 75
    except SetupError as exc:
        print(colored(f"\n🔑 {exc}\n   {_progress_note(state_path)}", "red"))
        return 2
    except BackendError as exc:
        print(colored(f"\n❌ {exc}\n   {_progress_note(state_path)} Run the same command again to retry.", "red"))
        return 1
    finally:
        document.close()
    elapsed = time.monotonic() - started
    timed = [entry["seconds"] for entry in state["reading_log"] if entry.get("seconds")]
    per_page = sum(timed) / len(timed) if timed else 0
    usage = {name: value for name, value in state.get("usage", {}).items() if value}
    print(colored(
        f"\n✨ Processing complete! ✨ Read through page {state['last_completed_page']}, "
        f"{len(state['knowledge'])} knowledge points.",
        "green", attrs=["bold"],
    ))
    if per_page:
        print(colored(f"   ⏱️  {per_page:.1f} s per page read on average; this run took {elapsed:.0f} s", "green"))
    if usage:
        print(colored(f"   🧮 Tokens so far: {json.dumps(usage)}", "green"))
    return 0


def _progress_note(state_path: Path) -> str:
    try:
        page = load_state(state_path)["last_completed_page"]
    except (OSError, ValueError):
        return "Nothing was lost."
    return f"Progress is saved through page {page}." if page else "No pages were finished yet."


if __name__ == "__main__":
    sys.exit(main())
