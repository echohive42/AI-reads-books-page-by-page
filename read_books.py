#!/usr/bin/env python3
"""Page-by-page PDF knowledge extraction with resumable, bounded summarization."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import fitz  # PyMuPDF
from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
from pydantic import BaseModel
from termcolor import colored

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_ANALYSIS_MODEL = "gpt-4o-mini"
DEFAULT_BASE_DIR = Path("book_analysis")


class PageContent(BaseModel):
    has_content: bool
    knowledge: list[str]


@dataclass
class APIResult:
    value: Any
    usage: dict[str, int]


def empty_state() -> dict[str, Any]:
    return {
        "version": 1,
        "document_id": None,
        "last_completed_page": 0,
        "knowledge": [],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
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


def usage_dict(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    result: dict[str, int] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if isinstance(usage, dict):
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)
        if isinstance(value, int):
            result[name] = value
    return result


def add_usage(total: dict[str, int], usage: dict[str, int]) -> None:
    for name, value in usage.items():
        total[name] = total.get(name, 0) + value


def call_with_retry(
    operation: Callable[[], Any],
    max_retries: int = 3,
    backoff: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    for attempt in range(max_retries + 1):
        try:
            return operation()
        except (
            ConnectionError,
            TimeoutError,
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
            InternalServerError,
        ):
            if attempt == max_retries:
                raise
            sleep(backoff * (2**attempt))
    raise AssertionError("unreachable")


def extract_page_knowledge(
    client: Any,
    page_text: str,
    recent_knowledge: Sequence[dict[str, Any]],
    page_number: int,
    *,
    model: str = DEFAULT_MODEL,
    max_retries: int = 3,
    backoff: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> APIResult:
    context = "\n".join(
        f"[page {item.get('page', '?')}] {item['text']}" for item in recent_knowledge
    ) or "(none)"
    user_prompt = (
        f"Recent bounded knowledge context:\n{context}\n\n"
        f"Current page {page_number}:\n{page_text}"
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Study this single book page. Skip blank, contents, index, copyright, "
                "publishing, bibliography and acknowledgments pages. Extract detailed, "
                "standalone learnable facts from educational content. The recent context "
                "is only for continuity: do not repeat facts already present."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    response = call_with_retry(
        lambda: client.beta.chat.completions.parse(
            model=model, messages=messages, response_format=PageContent
        ),
        max_retries=max_retries,
        backoff=backoff,
        sleep=sleep,
    )
    parsed = response.choices[0].message.parsed
    points = parsed.knowledge if parsed and parsed.has_content else []
    return APIResult(points, usage_dict(response))


def _tagged(items: Sequence[str]) -> str:
    return "\n".join(f"<item>{item}</item>" for item in items)


def _summary_call(
    client: Any,
    items: Sequence[str],
    *,
    model: str,
    max_retries: int,
    backoff: float,
    sleep: Callable[[float], None],
) -> APIResult:
    response = call_with_retry(
        lambda: client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Merge these bounded items into a concise, detailed Markdown summary. "
                        "Preserve source page references. Return only Markdown."
                    ),
                },
                {"role": "user", "content": _tagged(items)},
            ],
        ),
        max_retries=max_retries,
        backoff=backoff,
        sleep=sleep,
    )
    return APIResult(response.choices[0].message.content or "", usage_dict(response))


def hierarchical_summary(
    client: Any,
    knowledge: Sequence[dict[str, Any]],
    *,
    model: str = DEFAULT_ANALYSIS_MODEL,
    chunk_size: int = 20,
    max_item_chars: int = 8_000,
    max_retries: int = 3,
    backoff: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> APIResult:
    """Map/reduce summaries so no model request exceeds chunk_size items."""
    if chunk_size < 2:
        raise ValueError("summary chunk size must be >= 2")
    if max_item_chars < 100:
        raise ValueError("max_item_chars must be >= 100")
    if not knowledge:
        return APIResult("", {})
    layer = [
        f"[page {item.get('page', '?')}] {item['text']}"[:max_item_chars]
        for item in knowledge
    ]
    total_usage: dict[str, int] = {}
    while True:
        next_layer: list[str] = []
        for offset in range(0, len(layer), chunk_size):
            result = _summary_call(
                client,
                layer[offset : offset + chunk_size],
                model=model,
                max_retries=max_retries,
                backoff=backoff,
                sleep=sleep,
            )
            next_layer.append(result.value[:max_item_chars])
            add_usage(total_usage, result.usage)
        if len(next_layer) == 1:
            return APIResult(next_layer[0], total_usage)
        layer = next_layer


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
) -> Path | None:
    if not summary:
        return None
    summaries_dir.mkdir(parents=True, exist_ok=True)
    path = summaries_dir / f"{Path(book_name).stem}_{kind}.md"
    content = (
        f"# Book Analysis: {book_name}\n\n"
        f"Generated: {datetime.now().isoformat(timespec='seconds')}  \n"
        f"Processed through page: {page_number}\n\n{summary}\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


def _unwrap_result(result: Any) -> tuple[Any, dict[str, int]]:
    if isinstance(result, APIResult):
        return result.value, result.usage
    return result, {}


def run_analysis(
    document: Any,
    client: Any,
    state_path: Path,
    summaries_dir: Path,
    extractor: Callable[..., Any] = extract_page_knowledge,
    summarizer: Callable[..., Any] = hierarchical_summary,
    max_pages: int | None = None,
    resume: bool = True,
    context_window: int = 8,
    interval: int | None = None,
    model: str = DEFAULT_MODEL,
    analysis_model: str = DEFAULT_ANALYSIS_MODEL,
    summary_chunk_size: int = 20,
    summary_max_item_chars: int = 8_000,
    ocr: str = "off",
    ocr_engine: Callable[[Any], str] | None = None,
    max_retries: int = 3,
    backoff: float = 1.0,
    document_id: str | None = None,
) -> dict[str, Any]:
    if context_window < 0:
        raise ValueError("context_window must be >= 0")
    if interval is not None and interval <= 0:
        raise ValueError("interval must be > 0")
    state = load_state(state_path) if resume else empty_state()
    if not resume:
        # Persist the fresh checkpoint before any API call. If page 1 fails,
        # a later default resume must not see stale progress from an older run.
        save_state(state_path, state)
    if resume and document_id and state.get("document_id") not in (None, document_id):
        raise RuntimeError("Cannot resume: state belongs to a different PDF; use --reset")
    if document_id:
        state["document_id"] = document_id
    limit = min(document.page_count, max_pages if max_pages is not None else document.page_count)
    start = min(state["last_completed_page"], limit)
    book_name = state_path.name.removesuffix("_knowledge.json") + ".pdf"

    def generate_interval_summary(page_number: int) -> None:
        recent = [
            item
            for item in state["knowledge"]
            if state["last_interval_summary_page"] < (item.get("page") or 0) <= page_number
        ]
        summary_result = summarizer(
            client,
            recent,
            model=analysis_model,
            chunk_size=summary_chunk_size,
            max_item_chars=summary_max_item_chars,
            max_retries=max_retries,
            backoff=backoff,
        )
        summary, summary_usage = _unwrap_result(summary_result)
        save_summary(
            summaries_dir,
            book_name,
            summary,
            kind=f"interval_{page_number:05d}",
            page_number=page_number,
        )
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

    for page_index in range(start, limit):
        page_number = page_index + 1
        page_text = extract_text(document[page_index], ocr=ocr, ocr_engine=ocr_engine)
        context = state["knowledge"][-context_window:] if context_window else []
        raw_result = extractor(
            client,
            page_text,
            context,
            page_number,
            model=model,
            max_retries=max_retries,
            backoff=backoff,
        )
        points, request_usage = _unwrap_result(raw_result)
        # Update progress only after extraction returns successfully.
        existing_texts = {item["text"] for item in state["knowledge"]}
        for point in points:
            if point not in existing_texts:
                state["knowledge"].append({"page": page_number, "text": point})
                existing_texts.add(point)
        add_usage(state["usage"], request_usage)
        state["last_completed_page"] = page_number
        save_state(state_path, state)

        if interval and page_number % interval == 0 and page_number < limit:
            generate_interval_summary(page_number)

    # `limit` is the effective final page, including when max_pages exceeds PDF length.
    if (
        state["knowledge"]
        and limit > 0
        and state["last_completed_page"] == limit
        and state["final_summary_page"] != limit
    ):
        summary_result = summarizer(
            client,
            state["knowledge"],
            model=analysis_model,
            chunk_size=summary_chunk_size,
            max_item_chars=summary_max_item_chars,
            max_retries=max_retries,
            backoff=backoff,
        )
        summary, summary_usage = _unwrap_result(summary_result)
        save_summary(
            summaries_dir,
            book_name,
            summary,
            kind="final",
            page_number=limit,
        )
        add_usage(state["usage"], summary_usage)
        state["final_summary_page"] = limit
        save_state(state_path, state)
    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract page-sourced knowledge from a PDF and build bounded summaries."
    )
    parser.add_argument("pdf", type=Path, help="PDF file to process")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="page extraction model")
    parser.add_argument("--analysis-model", default=DEFAULT_ANALYSIS_MODEL)
    parser.add_argument("--interval", type=int, default=None, help="interval summary pages")
    parser.add_argument("--test-pages", type=int, default=None, help="maximum pages to process")
    parser.add_argument("--context-window", type=int, default=8, help="recent knowledge items")
    parser.add_argument("--summary-chunk-size", type=int, default=20)
    parser.add_argument("--summary-max-item-chars", type=int, default=8_000)
    parser.add_argument("--ocr", choices=("off", "auto", "always"), default="off")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--backoff", type=float, default=1.0)
    parser.add_argument("--reset", action="store_true", help="explicitly delete this output set")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true", default=True)
    resume_group.add_argument("--no-resume", dest="resume", action="store_false")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.test_pages is not None and args.test_pages <= 0:
        parser.error("--test-pages must be > 0")
    if args.interval is not None and args.interval <= 0:
        parser.error("--interval must be > 0")
    if args.context_window < 0:
        parser.error("--context-window must be >= 0")
    if args.summary_chunk_size < 2:
        parser.error("--summary-chunk-size must be >= 2")
    if args.summary_max_item_chars < 100:
        parser.error("--summary-max-item-chars must be >= 100")
    if args.max_retries < 0:
        parser.error("--max-retries must be >= 0")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    pdf_path = setup_directories(args.pdf, args.base_dir, reset=args.reset)
    stem = args.pdf.stem
    state_path = args.base_dir / "knowledge_bases" / f"{stem}_knowledge.json"
    summaries_dir = args.base_dir / "summaries"
    client = OpenAI()
    document = fitz.open(pdf_path)
    try:
        state = run_analysis(
            document,
            client,
            state_path,
            summaries_dir,
            max_pages=args.test_pages,
            resume=args.resume and not args.reset,
            context_window=args.context_window,
            interval=args.interval,
            model=args.model,
            analysis_model=args.analysis_model,
            summary_chunk_size=args.summary_chunk_size,
            summary_max_item_chars=args.summary_max_item_chars,
            ocr=args.ocr,
            ocr_engine=pymupdf_ocr if args.ocr != "off" else None,
            max_retries=args.max_retries,
            backoff=args.backoff,
            document_id=file_digest(pdf_path),
        )
    finally:
        document.close()
    usage = state.get("usage", {})
    print(
        colored(
            f"✨ Complete through page {state['last_completed_page']}; "
            f"knowledge={len(state['knowledge'])}; usage={json.dumps(usage)}",
            "green",
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
