# 📚 AI reads books, page by page

Give it a PDF, and an AI reads it the way you would study a book: one page at a time.

On every page it writes down what it learned, each note tagged with its page number. Every 20 pages it stops and writes down **what it understands so far**. When it reaches the last page, it pulls the whole book together into one final summary. You watch it happen live in your terminal, page by page.

## ✨ Run it on the ChatGPT, Claude or Grok plan you already have, no API key needed

The reader can now use the coding CLI that comes with your plan instead of a paid API key:

| You have | Run with | Sign in once with |
|---|---|---|
| A ChatGPT plan | `--backend codex` (the [Codex CLI](https://developers.openai.com/codex)) | `codex login` |
| A Claude plan | `--backend claude` ([Claude Code](https://code.claude.com/docs)) | run `claude` and sign in |
| A Grok plan | `--backend grok` ([Grok Build](https://docs.x.ai/build/overview)) | `grok login` |
| An OpenAI API key | `--backend api` (the default) | `export OPENAI_API_KEY=...` |

Whichever you pick, it still reads one page per call. The CLI only ever sees the page in front of it and the reader's own notes. It never gets the PDF file, a shell, or any tools (see [What the AI can see](#-what-the-ai-can-see)).

## 🚀 Quick start

```bash
git clone https://github.com/echohive42/AI-reads-books-page-by-page
cd AI-reads-books-page-by-page
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then pick the way you want it to read. Each example reads 3 pages of the bundled `meditations.pdf`, starting at page 31 where Book One begins. That is a quick taste before you give it a whole book.

**On your ChatGPT plan (Codex):**

```bash
codex login
python read_books.py --backend codex --start-page 31 --test-pages 3
```

**On your Claude plan (Claude Code):**

```bash
claude   # sign in once, then exit
python read_books.py --backend claude --start-page 31 --test-pages 3
```

**On your Grok plan (Grok Build):**

```bash
grok login
python read_books.py --backend grok --start-page 31 --test-pages 3
```

**With an OpenAI API key:**

```bash
export OPENAI_API_KEY="sk-..."
python read_books.py --start-page 31 --test-pages 3
```

When you are ready for a whole book:

```bash
python read_books.py path/to/your_book.pdf --backend codex
```

Stop it any time with Ctrl+C. Every finished page is saved, and running the same command again carries on from where it stopped.

## 🧠 How it works

The idea is simple: read like a person, not like a search engine. Nobody understands a book by seeing all of it at once. You read a page, connect it to what you already know, and your understanding grows as you go.

**Every page is its own model call.** For page N, the reader sends:

1. 📄 the text of page N (pulled out with PyMuPDF),
2. ✂️ the last lines of page N-1, so a sentence that runs over the page break still makes sense,
3. 📝 its most recent notes, each tagged with the page it came from,
4. 💭 one-line memories of the last few pages,
5. 🧭 what it understood so far, from its last look back.

It answers with a small JSON object: is this page worth reading (it skips contents, index, copyright and bibliography pages), a one-line gist of what the page adds, and the new knowledge points. Points it already has are dropped, so the notes do not fill up with repeats. Then the page is saved.

**Every 20 pages it looks back.** It takes what it understood last time, adds the notes from the pages since, and writes an updated "What I understand so far, after page N". It keeps page references like (p. 34) and ends with the threads it expects the book to develop. That understanding then travels with every page that follows, so page 140 is read in the light of pages 1 to 139.

**At the end it pulls the whole book together** from every page-tagged note and its latest understanding. A very long book is summarised part by part first, so no single request grows without limit.

Here is page 33 of `meditations.pdf` from a real run on Grok. Look at the first point: a sentence that started at the bottom of page 32 gets finished here.

```text
📖 Reading page 33/248...
   💭 The catalogue finishes Rusticus with careful reading, slow assent, and Epictetus, then credits Apollonius with…
   ✅ Found 11 new knowledge points (23 so far) · 26.2 s
      • The lesson from Rusticus that broke off at the end of the previous page is completed here: read carefully and…
      • Marcus owes to Rusticus his acquaintance with the discourses of Epictetus, which Rusticus communicated to him…
      • From Apollonius (Book I, section 8) Marcus learned freedom of will and undeviating steadiness of purpose, and…
      … and 8 more
   💾 Saved knowledge base (23 items)
```

And when it stops to look back (that test run used `--interval 2`; the default is every 20 pages):

```text
📊 Progress: 32/248 pages read
🤔 Updating what I understand so far (12 new notes)...
✨ What I understand so far, saved to: book_analysis/summaries/meditations_interval_00032.md
```

## ❤️ Go deeper: Get Amplified + weekly 1000x LAB meetings

This repo is one small doorway into the larger AI-building practice I share with patrons.

- ❤️ [Support me on Patreon](https://www.patreon.com/c/echohive42/membership) to get the full project collection, source code, explanations, and ongoing AI-building material.
- 🎥 [Get Amplified](https://www.patreon.com/collection/1845761) is my series for thinking fast, building faster, and speed-running your creativity: 90+ lessons, and some are free to watch. Learn to use Codex, Claude Code, Cursor, and other AI tools effectively and creatively. Get amplified by becoming amplifiable.
- 🧠 [1000x LAB for Architect+ tiers](https://www.patreon.com/collection/759209) meets live every Sunday at 12 PM Pacific, and every past meeting stays in the archive. These sessions go behind the scenes on real builds, agent workflows, creative tooling, and the decisions that turn experiments into finished, usable systems.
- 🤝 Higher memberships also include 1-on-1 meetings for more direct guidance on your AI builds, workflows, and creative direction.

## 📂 What you get

For `your_book.pdf`:

```text
book_analysis/
├── pdfs/your_book.pdf                      # a copy of the book
├── knowledge_bases/your_book_knowledge.json  # every note with its page, gists, progress
└── summaries/
    ├── your_book_interval_00020.md         # what it understood after page 20
    ├── your_book_interval_00040.md         # ...after page 40, and so on
    └── your_book_final.md                  # the whole book, pulled together
```

The knowledge base is saved safely after every page, together with the PDF's fingerprint. If you change the PDF but keep its name, the reader stops instead of mixing two books; use `--reset` to start that book over. `--reset` only touches that one book's files.

## ⚙️ Options

Run `python read_books.py --help` for all of them. The ones you will use most:

| Option | What it does |
|---|---|
| `pdf` | The book to read (default: `meditations.pdf`) |
| `--backend api\|claude\|codex\|grok` | Who reads it (default: `api`) |
| `--model NAME` | The model for reading pages (default: `gpt-6-luna` on the API, the CLI's own default otherwise) |
| `--summary-backend`, `--summary-model` | Let a different backend or model write the summaries (default on the API: `gpt-6.1-sol`) |
| `--effort low\|medium\|high` | Reasoning effort, for models that have it |
| `--start-page N` | Start at page N, for example to skip the front matter |
| `--test-pages N` | Read only N pages, to try things out |
| `--interval N` | Look back every N pages (default: 20; 0 turns it off) |
| `--context-window N` | How many recent notes travel with each page (default: 8) |
| `--ocr auto` | Read scanned pages with Tesseract (`brew install tesseract` on a Mac) |
| `--reset` | Forget this book's notes and summaries and start over |
| `--timeout SECONDS` | How long one model call may take (default: 600) |

For example, read with Grok but let Claude write the summaries:

```bash
python read_books.py your_book.pdf --backend grok --summary-backend claude
```

## 🔒 What the AI can see

Only the text the reader sends it. Each CLI call is a one-shot run with no shell, in an empty temporary folder, and with its tools switched off. The page text goes in through standard input (or a temporary prompt file for Grok), never on the command line. No call ever uses an always-approve or skip-permissions flag.

- **Claude Code:** `claude -p --tools "" --safe-mode --strict-mcp-config --no-session-persistence --json-schema ...`
- **Codex:** `codex exec -s read-only --ephemeral --ignore-user-config` with its shell, exec, image and plugin features turned off and web search disabled, plus `--output-schema`.
- **Grok Build:** `--sandbox strict --no-subagents --disable-web-search`, every built-in tool removed and denied, a turn limit, and `--json-schema`.

You can see every flag in [`backends.py`](backends.py).

Codex and Claude runs are not saved anywhere. Grok Build has no such option yet, so it keeps each page's call in its own local session history (`~/.grok/sessions`), like any other Grok session.

## ⚖️ Honest trade-offs

- **A CLI is slower than the API.** Every page starts a fresh CLI run, and that startup time comes on top of the model's own thinking. In our 3 page test with each CLI's default model, a page took about 13 seconds on Codex and about 19 seconds on Grok. So a 250 page book is roughly an hour or more of reading, plus the summaries. You can leave it running, and Ctrl+C never loses a finished page.
- **Plan limits can stop a long book.** When your plan hits its usage limit, the reader stops, keeps every page so far, and tells you. Run the same command later and it carries on.
- **Your book's text goes to the provider you choose** (OpenAI, Anthropic or xAI), under your account's terms. Only send books you are happy to share with them.
- **Summaries are AI writing.** They can miss or bend things. Every note keeps its page number, so you can check it against the book.
- **PDF text is not always clean.** Multi-column or heavily designed pages can come out in the wrong order, and scanned pages need `--ocr auto`.

## 🧪 Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The tests use fake backends and a stand-in CLI, so they never call a real model or use any of your plan.

## 🙏 Thanks

Thanks to [xdlkc](https://github.com/xdlkc) for the pull request that made reading resumable: safe saving after every page, page-tagged notes, bounded summaries, OCR and the first test suite. This version builds on it.

## 📜 License

MIT, see [LICENCE](LICENCE).
