import re

import pytest

import read_books
from backends import Backend, Reply, SetupError, TransientError, UsageLimitError


class FakePage:
    def __init__(self, text):
        self.text = text

    def get_text(self):
        return self.text


class FakeDocument:
    def __init__(self, texts):
        self.pages = [FakePage(text) for text in texts]
        self.page_count = len(self.pages)

    def __getitem__(self, index):
        return self.pages[index]

    def close(self):
        pass


def page_of(prompt):
    return int(re.search(r"Now read page (\d+)", prompt).group(1))


def section(prompt, title):
    """The lines of one titled part of a page prompt."""
    match = re.search(rf"{title}:\n(.*?)(?:\n\n|\Z)", prompt, re.DOTALL)
    return match.group(1) if match else ""


class FakeBackend(Backend):
    """Answers like a model would: page notes for schema calls, Markdown otherwise."""

    name = "fake"

    def __init__(self, page=None, summary=None):
        super().__init__("fake-model")
        self.page = page or (lambda prompt, number: {
            "has_content": True, "page_gist": f"gist {number}", "knowledge": [f"fact {number}"],
        })
        self.summary = summary or (lambda system, prompt: "SUMMARY")
        self.calls = []

    def ask(self, system, prompt, schema=None):
        self.calls.append({"system": system, "prompt": prompt, "schema": schema})
        if schema:
            return Reply(self.page(prompt, page_of(prompt)), {"total_tokens": 3}, 0.5)
        return Reply(self.summary(system, prompt), {"total_tokens": 5}, 1.0)

    def page_calls(self):
        return [call for call in self.calls if call["schema"]]

    def summary_calls(self):
        return [call for call in self.calls if not call["schema"]]


def run(document, reader, tmp_path, **options):
    options.setdefault("interval", None)
    options.setdefault("sleep", lambda _delay: None)
    return read_books.run_analysis(
        document,
        reader,
        options.pop("state_path", tmp_path / "book_knowledge.json"),
        options.pop("summaries_dir", tmp_path / "summaries"),
        book_name="book.pdf",
        **options,
    )


# --- output folders --------------------------------------------------------------------------


def test_setup_preserves_outputs_by_default_and_reset_clears_them(tmp_path):
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"pdf")
    base = tmp_path / "analysis"
    knowledge = base / "knowledge_bases" / "book_knowledge.json"
    summary = base / "summaries" / "book_final.md"
    knowledge.parent.mkdir(parents=True)
    summary.parent.mkdir(parents=True)
    knowledge.write_text("keep", encoding="utf-8")
    summary.write_text("keep", encoding="utf-8")

    copied = read_books.setup_directories(pdf, base, reset=False)

    assert knowledge.read_text(encoding="utf-8") == "keep"
    assert summary.read_text(encoding="utf-8") == "keep"
    assert copied == base / "pdfs" / "book.pdf"

    read_books.setup_directories(pdf, base, reset=True)

    assert not knowledge.exists()
    assert not summary.exists()


def test_setup_rejects_changed_pdf_with_same_name_without_reset(tmp_path):
    source = tmp_path / "incoming" / "book.pdf"
    source.parent.mkdir()
    source.write_bytes(b"version one")
    base = tmp_path / "analysis"
    read_books.setup_directories(source, base)
    source.write_bytes(b"version two")

    with pytest.raises(RuntimeError, match="changed.*--reset"):
        read_books.setup_directories(source, base)

    copied = read_books.setup_directories(source, base, reset=True)
    assert copied.read_bytes() == b"version two"


def test_reset_only_deletes_selected_books_outputs(tmp_path):
    source = tmp_path / "book.pdf"
    source.write_bytes(b"book")
    base = tmp_path / "analysis"
    other_state = base / "knowledge_bases" / "other_knowledge.json"
    other_summary = base / "summaries" / "other_final.md"
    similar_summary = base / "summaries" / "book_notes_final.md"
    other_state.parent.mkdir(parents=True)
    other_summary.parent.mkdir(parents=True)
    other_state.write_text("keep", encoding="utf-8")
    other_summary.write_text("keep", encoding="utf-8")
    similar_summary.write_text("keep", encoding="utf-8")

    read_books.setup_directories(source, base, reset=True)

    assert other_state.read_text(encoding="utf-8") == "keep"
    assert other_summary.read_text(encoding="utf-8") == "keep"
    assert similar_summary.read_text(encoding="utf-8") == "keep"


def test_reset_treats_glob_characters_in_book_name_literally(tmp_path):
    source = tmp_path / "book[1].pdf"
    source.write_bytes(b"book")
    base = tmp_path / "analysis"
    selected = base / "summaries" / "book[1]_final.md"
    unrelated = base / "summaries" / "book1_final.md"
    selected.parent.mkdir(parents=True)
    selected.write_text("delete", encoding="utf-8")
    unrelated.write_text("keep", encoding="utf-8")

    read_books.setup_directories(source, base, reset=True)

    assert not selected.exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"


# --- reading page by page ---------------------------------------------------------------------


def test_every_page_is_its_own_call_and_sees_only_that_page(tmp_path):
    reader = FakeBackend()
    run(FakeDocument(["alpha text", "beta text", "gamma text"]), reader, tmp_path)

    pages = reader.page_calls()
    assert [page_of(call["prompt"]) for call in pages] == [1, 2, 3]
    for call, own, later in zip(pages, ["alpha", "beta", "gamma"], [["beta", "gamma"], ["gamma"], []]):
        body = re.search(r"<page>\n(.*?)\n</page>", call["prompt"], re.DOTALL).group(1)
        assert body == f"{own} text"
        assert all(word not in call["prompt"] for word in later)  # never a page it has not reached
        assert call["system"] == read_books.PAGE_SYSTEM_PROMPT
        assert call["schema"] == read_books.PAGE_SCHEMA


def test_page_prompt_carries_notes_gists_understanding_and_the_previous_pages_end(tmp_path):
    reader = FakeBackend()
    long_page = "start of page one. " + "x" * 2000 + " and the sentence runs over"
    run(FakeDocument([long_page, "page two"]), reader, tmp_path)

    first, second = (call["prompt"] for call in reader.page_calls())
    assert "(none yet)" in section(first, "Your most recent notes")
    assert "<previous_page_end>" not in first
    assert "[page 1] fact 1" in section(second, "Your most recent notes")
    assert "[page 1] gist 1" in second
    tail = re.search(r"<previous_page_end>\n(.*?)\n</previous_page_end>", second, re.DOTALL).group(1)
    assert tail.endswith("the sentence runs over")
    assert len(tail) <= read_books.PREVIOUS_PAGE_TAIL
    assert "start of page one" not in second  # only the tail, never the whole previous page

    prompt = read_books.page_prompt(
        "body", 9, page_count=20, book_name="b.pdf",
        understanding={"page": 8, "text": "u" * (read_books.UNDERSTANDING_CHARS + 500)},
    )
    assert "Book: b.pdf, page 9 of 20" in prompt
    assert "What you understood after page 8" in prompt
    assert prompt.count("u") < read_books.UNDERSTANDING_CHARS + 100


def test_run_analysis_passes_only_recent_notes(tmp_path):
    reader = FakeBackend()
    run(FakeDocument(["one", "two", "three", "four"]), reader, tmp_path, context_window=2)

    notes = [section(call["prompt"], "Your most recent notes") for call in reader.page_calls()]
    assert notes == [
        "(none yet)",
        "[page 1] fact 1",
        "[page 1] fact 1\n[page 2] fact 2",
        "[page 2] fact 2\n[page 3] fact 3",
    ]


def test_resume_starts_after_persisted_page_without_duplicate_knowledge(tmp_path):
    reader = FakeBackend()
    document = FakeDocument(["one", "two", "three"])
    run(document, reader, tmp_path, max_pages=2)
    run(document, reader, tmp_path, resume=True)

    state = read_books.load_state(tmp_path / "book_knowledge.json")
    assert [page_of(call["prompt"]) for call in reader.page_calls()] == [1, 2, 3]
    assert state["last_completed_page"] == 3
    assert [item["text"] for item in state["knowledge"]] == ["fact 1", "fact 2", "fact 3"]
    assert [item["page"] for item in state["knowledge"]] == [1, 2, 3]
    assert [entry["gist"] for entry in state["reading_log"]] == ["gist 1", "gist 2", "gist 3"]


def test_failed_page_is_not_marked_complete(tmp_path):
    def failing(_prompt, _number):
        raise RuntimeError("API failed")

    with pytest.raises(RuntimeError, match="API failed"):
        run(FakeDocument(["one"]), FakeBackend(page=failing), tmp_path)

    state = read_books.load_state(tmp_path / "book_knowledge.json")
    assert state["last_completed_page"] == 0
    assert state["knowledge"] == []


def test_no_resume_clears_old_progress_before_a_new_failed_run(tmp_path):
    state_path = tmp_path / "book_knowledge.json"
    old_state = read_books.empty_state()
    old_state["last_completed_page"] = 5
    old_state["knowledge"] = [{"page": 5, "text": "stale"}]
    read_books.save_state(state_path, old_state)

    def failing(_prompt, _number):
        raise RuntimeError("new run failed")

    with pytest.raises(RuntimeError, match="new run failed"):
        run(FakeDocument(["one"]), FakeBackend(page=failing), tmp_path, resume=False)

    state = read_books.load_state(state_path)
    assert state["last_completed_page"] == 0
    assert state["knowledge"] == []


def test_page_reading_retries_bad_answers_and_records_usage():
    answers = iter([
        TransientError("timed out"),
        {"has_content": True, "knowledge": "not a list", "page_gist": ""},  # schema mismatch
        {"has_content": True, "page_gist": " the gist ", "knowledge": [" new fact ", " "]},
    ])

    def page(_prompt, _number):
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    reader = FakeBackend(page=page)
    retries = []
    reading = read_books.extract_page_knowledge(
        reader, "page body", [{"page": 1, "text": "old 1"}], 3,
        max_retries=2, backoff=0, sleep=lambda _delay: None,
        on_retry=lambda exc, _delay: retries.append(type(exc).__name__),
    )

    assert (reading.has_content, reading.gist, reading.points) == (True, "the gist", ["new fact"])
    assert reading.usage == {"total_tokens": 3} and reading.seconds == 0.5
    assert retries == ["TransientError", "TransientError"]
    assert "old 1" in reader.calls[-1]["prompt"] and "page body" in reader.calls[-1]["prompt"]


def test_skipped_pages_keep_no_notes(tmp_path):
    reader = FakeBackend(page=lambda _p, number: {
        "has_content": number != 1, "page_gist": f"gist {number}", "knowledge": [f"fact {number}"],
    })
    state = run(FakeDocument(["contents", "chapter"]), reader, tmp_path)

    assert [item["page"] for item in state["knowledge"]] == [2]
    assert state["reading_log"][0] == {"page": 1, "gist": "", "points": 0, "seconds": 0.5}


def test_blank_pages_need_no_model_call(tmp_path):
    reader = FakeBackend()
    state = run(FakeDocument(["one", "   \n", "three"]), reader, tmp_path)

    assert [page_of(call["prompt"]) for call in reader.page_calls()] == [1, 3]
    assert state["last_completed_page"] == 3


def test_repeated_points_are_dropped_even_with_different_spacing_and_case(tmp_path):
    reader = FakeBackend(page=lambda _p, number: {
        "has_content": True, "page_gist": "", "knowledge": ["The  Same point" if number == 1 else "the same POINT",
                                                            f"fact {number}"],
    })
    state = run(FakeDocument(["one", "two"]), reader, tmp_path)

    assert [item["text"] for item in state["knowledge"]] == ["The  Same point", "fact 1", "fact 2"]
    assert [entry["points"] for entry in state["reading_log"]] == [2, 1]


def test_start_page_skips_front_matter_and_test_pages_counts_from_there(tmp_path):
    reader = FakeBackend(summary=lambda _s, _p: "FINAL")
    state = run(FakeDocument(["cover", "two", "three", "four"]), reader, tmp_path, start_page=2, max_pages=2)

    assert [page_of(call["prompt"]) for call in reader.page_calls()] == [2, 3]
    assert state["last_completed_page"] == 3
    final = (tmp_path / "summaries" / "book_final.md").read_text(encoding="utf-8")
    assert "pages 2–3 of 4" in final


def test_max_pages_above_pdf_length_still_writes_final_summary(tmp_path):
    reader = FakeBackend(summary=lambda _s, _p: "FINAL")
    run(FakeDocument(["one", "two"]), reader, tmp_path, max_pages=99)

    final_files = list((tmp_path / "summaries").glob("*_final.md"))
    assert len(final_files) == 1
    assert "FINAL" in final_files[0].read_text(encoding="utf-8")


def test_smaller_max_pages_never_regresses_or_relabels_completed_state(tmp_path):
    reader = FakeBackend()
    document = FakeDocument(["one", "two", "three"])
    run(document, reader, tmp_path, max_pages=3)
    original_final = (tmp_path / "summaries" / "book_final.md").read_text(encoding="utf-8")

    run(document, reader, tmp_path, max_pages=2)

    state = read_books.load_state(tmp_path / "book_knowledge.json")
    assert state["last_completed_page"] == 3
    assert state["final_summary_page"] == 3
    assert len(reader.summary_calls()) == 1
    assert (tmp_path / "summaries" / "book_final.md").read_text(encoding="utf-8") == original_final


def test_resume_rejects_a_different_document(tmp_path):
    run(FakeDocument(["one"]), FakeBackend(), tmp_path, document_id="first")

    with pytest.raises(RuntimeError, match="different PDF"):
        run(FakeDocument(["replacement"]), FakeBackend(), tmp_path, document_id="second")


# --- understanding builds as it reads -------------------------------------------------------------


def test_understanding_rolls_forward_and_feeds_the_next_pages(tmp_path):
    def summary(system, prompt):
        if system == read_books.UNDERSTANDING_SYSTEM_PROMPT:
            return f"UNDERSTANDING AFTER {re.search(r'after page (\d+)\.$', prompt).group(1)}"
        return "FINAL"

    reader = FakeBackend(summary=summary)
    state = run(FakeDocument(["p1", "p2", "p3", "p4", "p5"]), reader, tmp_path, interval=2)

    first, second, final = (call["prompt"] for call in reader.summary_calls())
    assert "first look back" in first
    assert "[page 1] fact 1" in first and "[page 2] fact 2" in first
    # The second update builds on the first and only adds the notes since.
    assert "UNDERSTANDING AFTER 2" in second
    assert "[page 3] fact 3" in second and "[page 4] fact 4" in second and "fact 1" not in second
    # Pages after an update read with it.
    page_prompts = {page_of(call["prompt"]): call["prompt"] for call in reader.page_calls()}
    assert "UNDERSTANDING AFTER 2" in page_prompts[3]
    assert "UNDERSTANDING AFTER 4" in page_prompts[5]
    assert "<understanding>" not in page_prompts[2]
    # The final summary pulls every note and the latest understanding together.
    assert all(f"fact {number}" in final for number in range(1, 6))
    assert "UNDERSTANDING AFTER 4" in final
    assert state["understanding"] == {"page": 4, "text": "UNDERSTANDING AFTER 4"}

    summaries = tmp_path / "summaries"
    assert sorted(path.name for path in summaries.iterdir()) == [
        "book_final.md", "book_interval_00002.md", "book_interval_00004.md",
    ]
    interval = (summaries / "book_interval_00004.md").read_text(encoding="utf-8")
    assert "What I understand so far, after page 4 of 5" in interval
    assert "by fake (fake-model)" in interval


def test_interval_summary_is_not_written_on_the_last_page(tmp_path):
    reader = FakeBackend()
    run(FakeDocument(["p1", "p2", "p3", "p4"]), reader, tmp_path, interval=2)

    summaries = sorted(path.name for path in (tmp_path / "summaries").iterdir())
    assert summaries == ["book_final.md", "book_interval_00002.md"]


def test_resume_retries_an_interval_summary_that_failed_after_page_save(tmp_path):
    calls = []

    def flaky_summary(_system, _prompt):
        calls.append("summary")
        if len(calls) == 1:
            raise ConnectionError("summary failed")
        return "RECOVERED"

    document = FakeDocument(["one", "two", "three"])
    options = dict(interval=2, max_pages=3, max_retries=0)

    with pytest.raises(ConnectionError, match="summary failed"):
        run(document, FakeBackend(summary=flaky_summary), tmp_path, **options)

    state = read_books.load_state(tmp_path / "book_knowledge.json")
    assert state["last_completed_page"] == 2
    assert state["last_interval_summary_page"] == 0

    run(document, FakeBackend(summary=flaky_summary), tmp_path, **options)
    state = read_books.load_state(tmp_path / "book_knowledge.json")
    assert state["last_interval_summary_page"] == 2
    assert "RECOVERED" in (tmp_path / "summaries" / "book_interval_00002.md").read_text(encoding="utf-8")


def test_a_separate_summary_backend_writes_the_summaries(tmp_path):
    reader, writer = FakeBackend(), FakeBackend()
    run(FakeDocument(["one", "two", "three"]), reader, tmp_path, summary_backend=writer, interval=1)

    assert reader.summary_calls() == []
    assert len(writer.summary_calls()) == 3 and writer.page_calls() == []


# --- the final summary ----------------------------------------------------------------------------


def test_final_summary_is_one_call_when_the_notes_fit_and_is_never_truncated():
    long_answer = "word " * 10_000
    writer = FakeBackend(summary=lambda _s, _p: long_answer)
    knowledge = [{"page": number, "text": f"fact-{number}"} for number in range(1, 11)]

    result = read_books.final_summary(writer, knowledge, max_retries=0, max_item_chars=100)

    assert len(writer.calls) == 1
    assert result.value == long_answer.strip()


def test_final_summary_bounds_every_request_for_a_long_book():
    writer = FakeBackend(summary=lambda _s, prompt: f"part-{prompt.count('[page')}")
    knowledge = [{"page": number, "text": f"fact-{number}"} for number in range(1, 11)]

    result = read_books.final_summary(writer, knowledge, chunk_size=3, max_retries=0)

    assert result.value.startswith("part-")
    assert len(writer.calls) > 1
    assert result.usage["total_tokens"] == 5 * len(writer.calls)
    assert all(call["prompt"].count("[page") <= 3 for call in writer.calls)
    assert all(call["prompt"].count("<part>") <= 3 for call in writer.calls)
    assert writer.calls[-1]["system"] == read_books.FINAL_SYSTEM_PROMPT


def test_final_summary_bounds_the_size_of_each_item():
    writer = FakeBackend(summary=lambda _s, _p: "short")
    knowledge = [{"page": 1, "text": "x" * 50_000}]

    read_books.final_summary(writer, knowledge, chunk_size=2, max_item_chars=1_000, max_retries=0)

    assert writer.calls[0]["prompt"].count("x") <= 1_000


def test_an_empty_summary_is_retried():
    answers = iter(["", "FINAL"])
    writer = FakeBackend(summary=lambda _s, _p: next(answers))

    result = read_books.final_summary(writer, [{"page": 1, "text": "fact"}], backoff=0, sleep=lambda _d: None)

    assert result.value == "FINAL" and len(writer.calls) == 2


# --- small parts --------------------------------------------------------------------------------


def test_ocr_fallback_is_optional_and_injectable():
    page = FakePage("")
    assert read_books.extract_text(page, ocr="off") == ""
    assert read_books.extract_text(page, ocr="auto", ocr_engine=lambda _page: "OCR text") == "OCR text"


def test_ocr_missing_dependency_has_clear_error():
    page = FakePage("")
    with pytest.raises(RuntimeError, match="OCR.*dependency"):
        read_books.extract_text(page, ocr="auto", ocr_engine=None)


def test_retry_does_not_repeat_programming_setup_or_usage_limit_errors():
    for error in (ValueError("bad request"), SetupError("log in"), UsageLimitError("limit")):
        attempts = 0

        def operation():
            nonlocal attempts
            attempts += 1
            raise error

        with pytest.raises(type(error)):
            read_books.call_with_retry(operation, max_retries=3, backoff=0)
        assert attempts == 1


def test_old_state_files_still_load(tmp_path):
    path = tmp_path / "old_knowledge.json"
    path.write_text('{"knowledge": ["an old point"]}', encoding="utf-8")

    state = read_books.load_state(path)

    assert state["knowledge"] == [{"page": None, "text": "an old point"}]
    assert state["last_completed_page"] == 0 and state["understanding"] is None


def test_cli_defaults_keep_the_bare_command_working():
    args = read_books.build_parser().parse_args([])

    assert str(args.pdf) == "meditations.pdf"
    assert (args.backend, args.model, args.summary_backend, args.summary_model) == ("api", None, None, None)
    assert (args.interval, args.start_page, args.test_pages, args.resume, args.reset) == (20, 1, None, True, False)


def test_cli_parses_backends_models_resume_reset_and_limits():
    parser = read_books.build_parser()
    args = parser.parse_args([
        "book.pdf", "--backend", "codex", "--model", "m1", "--summary-backend", "claude",
        "--analysis-model", "m2", "--effort", "high", "--interval", "7", "--start-page", "3",
        "--test-pages", "9", "--context-window", "3", "--summary-chunk-size", "5",
        "--summary-max-item-chars", "2000", "--ocr", "auto", "--reset", "--timeout", "30",
    ])

    assert (args.backend, args.model, args.summary_backend, args.summary_model) == ("codex", "m1", "claude", "m2")
    assert (args.effort, args.interval, args.start_page, args.test_pages) == ("high", 7, 3, 9)
    assert (args.context_window, args.summary_chunk_size, args.ocr, args.timeout) == (3, 5, "auto", 30)
    assert args.summary_max_item_chars == 2000 and args.reset is True
    with pytest.raises(SystemExit):
        parser.parse_args(["--backend", "gemini"])


def test_summary_model_defaults_follow_the_backend():
    parser = read_books.build_parser()

    def models(*argv):
        reader, writer = read_books.make_backends(parser.parse_args(list(argv)))
        return reader.name, reader.model, writer.name, writer.model

    assert models() == ("api", read_books.API_PAGE_MODEL, "api", read_books.API_SUMMARY_MODEL)
    assert models("--backend", "grok") == ("grok", None, "grok", None)
    assert models("--backend", "codex", "--model", "m") == ("codex", "m", "codex", "m")
    assert models("--backend", "codex", "--model", "m", "--summary-backend", "claude") == (
        "codex", "m", "claude", None,
    )
    assert models("--backend", "claude", "--summary-backend", "api") == (
        "claude", None, "api", read_books.API_SUMMARY_MODEL,
    )


# --- the whole program, on a real PDF -----------------------------------------------------------


def make_pdf(path, pages):
    import pymupdf

    document = pymupdf.open()
    for text in pages:
        document.new_page().insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_main_reads_a_real_pdf_end_to_end(tmp_path, monkeypatch, capsys):
    pdf = tmp_path / "tiny.pdf"
    make_pdf(pdf, ["The first idea.", "The second idea.", "The third idea."])
    reader = FakeBackend()
    monkeypatch.setattr(read_books, "make_backends", lambda _args: (reader, reader))

    code = read_books.main([str(pdf), "--base-dir", str(tmp_path / "out"), "--interval", "2"])

    assert code == 0
    assert "The second idea." in reader.page_calls()[1]["prompt"]
    out = capsys.readouterr().out
    assert "📖 Reading page 2/3" in out and "✨ Processing complete!" in out
    assert (tmp_path / "out" / "summaries" / "tiny_final.md").exists()
    assert (tmp_path / "out" / "summaries" / "tiny_interval_00002.md").exists()
    state = read_books.load_state(tmp_path / "out" / "knowledge_bases" / "tiny_knowledge.json")
    assert state["last_completed_page"] == 3 and len(state["document_id"]) == 64


def test_main_pauses_on_a_usage_limit_and_resumes_later(tmp_path, monkeypatch, capsys):
    pdf = tmp_path / "tiny.pdf"
    make_pdf(pdf, ["one", "two", "three"])

    def page(_prompt, number):
        if number == 2:
            raise UsageLimitError("claude: your plan's usage limit was reached.")
        return {"has_content": True, "page_gist": "", "knowledge": [f"fact {number}"]}

    limited = FakeBackend(page=page)
    monkeypatch.setattr(read_books, "make_backends", lambda _args: (limited, limited))
    options = [str(pdf), "--base-dir", str(tmp_path / "out")]

    assert read_books.main(options) == 75
    assert "Progress is saved through page 1" in capsys.readouterr().out

    later = FakeBackend()
    monkeypatch.setattr(read_books, "make_backends", lambda _args: (later, later))
    assert read_books.main(options) == 0
    assert [page_of(call["prompt"]) for call in later.page_calls()] == [2, 3]


def test_main_stops_before_reading_when_the_api_key_is_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    pdf = tmp_path / "tiny.pdf"
    pdf.write_bytes(b"not read")

    assert read_books.main([str(pdf), "--base-dir", str(tmp_path / "out")]) == 2
    assert "--backend claude" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()
