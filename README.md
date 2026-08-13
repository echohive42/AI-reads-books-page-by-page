# 📚 AI Reads Books Page by Page

A resumable PDF knowledge extractor and hierarchical summarizer built with OpenAI and PyMuPDF.

The script processes a PDF one page at a time, stores page-sourced knowledge in JSON, creates bounded interval/final summaries, and can safely resume after interruption without reprocessing completed pages.

## Highlights

- **Resumable by default** — progress is persisted atomically after each successful page.
- **Page-level provenance** — every knowledge point retains its source page.
- **Bounded context** — only recent knowledge is supplied during page extraction.
- **Bounded hierarchical summaries** — map/reduce requests limit both item count and item size.
- **Separate models** — use one model for extraction and another for analysis.
- **Optional OCR** — PyMuPDF/Tesseract fallback for scanned pages.
- **Retries and usage tracking** — transient OpenAI failures are retried; token usage is accumulated when available.
- **Safe reset** — existing output is preserved unless `--reset` is explicitly supplied.
- **Tested without paid API calls** — the pytest suite uses fake clients and documents.

## Requirements

- Python 3.10+
- An OpenAI API key in `OPENAI_API_KEY`
- Optional OCR: Tesseract plus the language packs needed by your books

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="..."
```

For development:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Quick start

```bash
python read_books.py path/to/book.pdf
```

A more complete example:

```bash
python read_books.py path/to/book.pdf \
  --base-dir book_analysis \
  --model gpt-4o-mini \
  --analysis-model gpt-4o-mini \
  --interval 20 \
  --context-window 8 \
  --summary-chunk-size 20 \
  --summary-max-item-chars 8000 \
  --ocr auto
```

The script is non-interactive; run `python read_books.py --help` for every option.

## Important CLI options

| Option | Meaning |
|---|---|
| `pdf` | Source PDF path (required) |
| `--base-dir` | Output root; default `book_analysis` |
| `--model` | Page knowledge extraction model |
| `--analysis-model` | Interval/final summary model |
| `--interval N` | Create an interval summary every N pages |
| `--test-pages N` | Process at most N pages; values above the PDF length are safe |
| `--context-window N` | Recent knowledge items passed to the page extractor |
| `--summary-chunk-size N` | Maximum items in each hierarchical summary request |
| `--summary-max-item-chars N` | Maximum characters retained for each summary item |
| `--ocr off\|auto\|always` | OCR mode; default `off` |
| `--max-retries N` | Retries for transient OpenAI/network failures |
| `--backoff SECONDS` | Initial exponential retry delay |
| `--resume` | Resume the matching PDF state; default |
| `--no-resume` | Start processing at page 1 and replace state as work succeeds |
| `--reset` | Delete only this book's copied PDF/state/summaries and start fresh |

## Output

For `book.pdf`, the default layout is:

```text
book_analysis/
├── pdfs/book.pdf
├── knowledge_bases/book_knowledge.json
└── summaries/
    ├── book_interval_00020.md
    └── book_final.md
```

The JSON state includes:

- a SHA-256 document identity, preventing accidental resume against a changed PDF;
- `last_completed_page`;
- `knowledge` entries shaped as `{ "page": 12, "text": "..." }`;
- accumulated prompt/completion/total token usage when returned by the API;
- completed interval/final summary checkpoints.

State is written through a temporary file and atomically replaced. A failed page is never marked complete. If an interval summary fails after its page progress was saved, the missing interval is retried on the next run before later pages are processed.

## Resume and reset safety

Normal reruns resume automatically. If a PDF with the same filename has changed, the script stops instead of mixing old knowledge with new content:

```bash
python read_books.py book.pdf --reset
```

`--reset` is scoped to the selected book; it does not delete other books stored under the same `--base-dir`.

## OCR

`--ocr auto` uses extracted text when present and invokes OCR only for empty pages. `--ocr always` OCRs every page. PyMuPDF OCR requires a working Tesseract installation and language data; if unavailable, the program raises a clear dependency error instead of silently skipping scanned content.

Example on macOS:

```bash
brew install tesseract
```

Install any additional language packs required by your PDFs according to your Tesseract distribution.

## How bounded summarization works

Page extraction sees only `--context-window` recent knowledge items. Summaries use map/reduce layers: at most `--summary-chunk-size` items enter a request, and each item is truncated to `--summary-max-item-chars`. Intermediate summaries are subject to the same bound. This avoids repeatedly submitting one unbounded, ever-growing knowledge list.

## Limitations

- PyMuPDF text extraction may not preserve complex reading order in multi-column or heavily designed PDFs.
- OCR quality depends on Tesseract, scan quality, and installed languages.
- Exact-string knowledge de-duplication does not merge semantic paraphrases.
- Character bounds control request growth but are not an exact tokenizer-based budget.
- AI summaries can omit or distort details; retain the page-sourced JSON for verification.

## Verification

```bash
python -m pytest -q
python -m py_compile read_books.py
python read_books.py --help
```

Tests never call the real OpenAI API or process the bundled books.

## Support the author

This repository is one small doorway into the larger AI-building practice shared by the original author. You can [support the project on Patreon](https://www.patreon.com/c/echohive42/membership) and explore the author's [Get Amplified collection](https://www.patreon.com/collection/1845761).

## License

MIT. See [LICENCE](LICENCE).
