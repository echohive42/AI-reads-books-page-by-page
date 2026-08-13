from pathlib import Path

import pytest

import read_books


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


def test_resume_starts_after_persisted_page_without_duplicate_knowledge(tmp_path):
    state_path = tmp_path / "book_knowledge.json"
    calls = []

    def extractor(_client, text, context, page_number, **_kwargs):
        calls.append((text, page_number))
        return [f"fact {page_number}"]

    read_books.run_analysis(
        FakeDocument(["one", "two", "three"]),
        client=object(),
        state_path=state_path,
        summaries_dir=tmp_path / "summaries",
        extractor=extractor,
        summarizer=lambda *_args, **_kwargs: "summary",
        max_pages=2,
    )
    read_books.run_analysis(
        FakeDocument(["one", "two", "three"]),
        client=object(),
        state_path=state_path,
        summaries_dir=tmp_path / "summaries",
        extractor=extractor,
        summarizer=lambda *_args, **_kwargs: "summary",
        resume=True,
    )

    state = read_books.load_state(state_path)
    assert calls == [("one", 1), ("two", 2), ("three", 3)]
    assert state["last_completed_page"] == 3
    assert [item["text"] for item in state["knowledge"]] == ["fact 1", "fact 2", "fact 3"]
    assert [item["page"] for item in state["knowledge"]] == [1, 2, 3]


def test_failed_page_is_not_marked_complete(tmp_path):
    state_path = tmp_path / "book_knowledge.json"

    def failing(*_args, **_kwargs):
        raise RuntimeError("API failed")

    with pytest.raises(RuntimeError, match="API failed"):
        read_books.run_analysis(
            FakeDocument(["one"]),
            client=object(),
            state_path=state_path,
            summaries_dir=tmp_path / "summaries",
            extractor=failing,
            summarizer=lambda *_args, **_kwargs: "summary",
        )

    state = read_books.load_state(state_path)
    assert state["last_completed_page"] == 0
    assert state["knowledge"] == []


def test_no_resume_clears_old_progress_before_a_new_failed_run(tmp_path):
    state_path = tmp_path / "book_knowledge.json"
    old_state = read_books.empty_state()
    old_state["last_completed_page"] = 5
    old_state["knowledge"] = [{"page": 5, "text": "stale"}]
    read_books.save_state(state_path, old_state)

    with pytest.raises(RuntimeError, match="new run failed"):
        read_books.run_analysis(
            FakeDocument(["one"]),
            client=object(),
            state_path=state_path,
            summaries_dir=tmp_path / "summaries",
            extractor=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("new run failed")
            ),
            summarizer=lambda *_args, **_kwargs: "summary",
            resume=False,
        )

    state = read_books.load_state(state_path)
    assert state["last_completed_page"] == 0
    assert state["knowledge"] == []


def test_model_page_extraction_retries_records_usage_and_includes_bounded_context():
    client = FakeOpenAI(parsed=read_books.PageContent(has_content=True, knowledge=["new fact"]), failures=1)
    context = [{"page": 1, "text": "old 1"}, {"page": 2, "text": "old 2"}]

    result = read_books.extract_page_knowledge(
        client,
        "page body",
        context,
        3,
        model="page-model",
        max_retries=2,
        backoff=0,
        sleep=lambda _delay: None,
    )

    assert result.value == ["new fact"]
    assert result.usage == {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}
    assert client.parse_calls == 2
    request = client.last_parse_request
    assert request["model"] == "page-model"
    prompt = request["messages"][-1]["content"]
    assert "old 1" in prompt and "old 2" in prompt and "page body" in prompt


def test_run_analysis_passes_only_recent_context(tmp_path):
    contexts = []

    def extractor(_client, _text, context, page_number, **_kwargs):
        contexts.append([item["text"] for item in context])
        return [f"fact {page_number}"]

    read_books.run_analysis(
        FakeDocument(["one", "two", "three", "four"]),
        object(),
        tmp_path / "state.json",
        tmp_path / "summaries",
        extractor,
        lambda *_args, **_kwargs: "summary",
        context_window=2,
    )

    assert contexts == [[], ["fact 1"], ["fact 1", "fact 2"], ["fact 2", "fact 3"]]


def test_hierarchical_summary_uses_analysis_model_and_bounds_every_request():
    client = FakeOpenAI(summary_prefix="summary")
    knowledge = [{"page": number, "text": f"fact-{number}"} for number in range(1, 11)]

    result = read_books.hierarchical_summary(
        client, knowledge, model="analysis-model", chunk_size=3, max_retries=0
    )

    assert result.value.startswith("summary")
    assert result.usage["total_tokens"] == 84
    assert len(client.create_requests) > 1
    assert {request["model"] for request in client.create_requests} == {"analysis-model"}
    assert all(request["messages"][-1]["content"].count("<item>") <= 3 for request in client.create_requests)


def test_hierarchical_summary_bounds_the_size_of_each_item():
    client = FakeOpenAI(summary_prefix="short")
    knowledge = [{"page": 1, "text": "x" * 50_000}]

    read_books.hierarchical_summary(
        client,
        knowledge,
        model="analysis-model",
        chunk_size=2,
        max_item_chars=1_000,
        max_retries=0,
    )

    prompt = client.create_requests[0]["messages"][-1]["content"]
    assert len(prompt) < 1_100


def test_ocr_fallback_is_optional_and_injectable():
    page = FakePage("")
    assert read_books.extract_text(page, ocr="off") == ""
    assert read_books.extract_text(page, ocr="auto", ocr_engine=lambda _page: "OCR text") == "OCR text"


def test_ocr_missing_dependency_has_clear_error():
    page = FakePage("")
    with pytest.raises(RuntimeError, match="OCR.*dependency"):
        read_books.extract_text(page, ocr="auto", ocr_engine=None)


def test_max_pages_above_pdf_length_still_writes_final_summary(tmp_path):
    summaries = tmp_path / "summaries"
    read_books.run_analysis(
        FakeDocument(["one", "two"]),
        object(),
        tmp_path / "state.json",
        summaries,
        lambda _client, _text, _context, page_number, **_kwargs: [f"fact {page_number}"],
        lambda *_args, **_kwargs: "FINAL",
        max_pages=99,
    )

    final_files = list(summaries.glob("*_final.md"))
    assert len(final_files) == 1
    assert "FINAL" in final_files[0].read_text(encoding="utf-8")


def test_cli_parses_models_resume_reset_and_limits():
    parser = read_books.build_parser()
    defaults = parser.parse_args(["book.pdf"])
    args = parser.parse_args([
        "book.pdf", "--model", "m1", "--analysis-model", "m2", "--interval", "7",
        "--test-pages", "9", "--context-window", "3", "--summary-chunk-size", "5",
        "--summary-max-item-chars", "2000", "--ocr", "auto", "--reset",
    ])

    assert defaults.resume is True and defaults.reset is False
    assert (args.model, args.analysis_model, args.interval, args.test_pages) == ("m1", "m2", 7, 9)
    assert (args.context_window, args.summary_chunk_size, args.ocr) == (3, 5, "auto")
    assert args.summary_max_item_chars == 2000
    assert args.reset is True


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


def test_resume_rejects_a_different_document(tmp_path):
    state_path = tmp_path / "state.json"
    common = dict(
        client=object(),
        state_path=state_path,
        summaries_dir=tmp_path / "summaries",
        extractor=lambda _c, _t, _ctx, page, **_kw: [f"fact {page}"],
        summarizer=lambda *_args, **_kwargs: "summary",
    )
    read_books.run_analysis(FakeDocument(["one"]), document_id="first", **common)

    with pytest.raises(RuntimeError, match="different PDF"):
        read_books.run_analysis(FakeDocument(["replacement"]), document_id="second", **common)


def test_resume_retries_an_interval_summary_that_failed_after_page_save(tmp_path):
    state_path = tmp_path / "state.json"
    calls = []

    def flaky_summary(*_args, **_kwargs):
        calls.append("summary")
        if len(calls) == 1:
            raise ConnectionError("summary failed")
        return "RECOVERED"

    options = dict(
        client=object(),
        state_path=state_path,
        summaries_dir=tmp_path / "summaries",
        extractor=lambda _c, _t, _ctx, page, **_kw: [f"fact {page}"],
        summarizer=flaky_summary,
        interval=2,
        max_pages=3,
    )

    with pytest.raises(ConnectionError, match="summary failed"):
        read_books.run_analysis(FakeDocument(["one", "two", "three"]), **options)

    state = read_books.load_state(state_path)
    assert state["last_completed_page"] == 2
    assert state["last_interval_summary_page"] == 0

    read_books.run_analysis(FakeDocument(["one", "two", "three"]), **options)
    state = read_books.load_state(state_path)
    assert state["last_interval_summary_page"] == 2
    assert "RECOVERED" in (tmp_path / "summaries" / "state.json_interval_00002.md").read_text(
        encoding="utf-8"
    )


def test_smaller_max_pages_never_regresses_or_relabels_completed_state(tmp_path):
    state_path = tmp_path / "book_knowledge.json"
    summaries = tmp_path / "summaries"
    summary_inputs = []

    def summarizer(_client, knowledge, **_kwargs):
        summary_inputs.append([item["page"] for item in knowledge])
        return "SUMMARY"

    common = dict(
        client=object(),
        state_path=state_path,
        summaries_dir=summaries,
        extractor=lambda _c, _t, _ctx, page, **_kw: [f"fact {page}"],
        summarizer=summarizer,
    )
    document = FakeDocument(["one", "two", "three"])
    read_books.run_analysis(document, max_pages=3, **common)
    original_final = (summaries / "book_final.md").read_text(encoding="utf-8")

    read_books.run_analysis(document, max_pages=2, **common)

    state = read_books.load_state(state_path)
    assert state["last_completed_page"] == 3
    assert state["final_summary_page"] == 3
    assert summary_inputs == [[1, 2, 3]]
    assert (summaries / "book_final.md").read_text(encoding="utf-8") == original_final


def test_retry_does_not_repeat_programming_errors():
    attempts = 0

    def bad_operation():
        nonlocal attempts
        attempts += 1
        raise ValueError("bad request")

    with pytest.raises(ValueError, match="bad request"):
        read_books.call_with_retry(bad_operation, max_retries=3, backoff=0)
    assert attempts == 1


class FakeOpenAI:
    def __init__(self, parsed=None, failures=0, summary_prefix="summary"):
        self.parsed = parsed
        self.failures = failures
        self.summary_prefix = summary_prefix
        self.parse_calls = 0
        self.last_parse_request = None
        self.create_requests = []
        self.beta = _Obj(chat=_Obj(completions=_Obj(parse=self.parse)))
        self.chat = _Obj(completions=_Obj(create=self.create))

    def parse(self, **request):
        self.parse_calls += 1
        self.last_parse_request = request
        if self.parse_calls <= self.failures:
            raise ConnectionError("temporary")
        message = _Obj(parsed=self.parsed)
        usage = _Obj(prompt_tokens=11, completion_tokens=4, total_tokens=15)
        return _Obj(choices=[_Obj(message=message)], usage=usage)

    def create(self, **request):
        self.create_requests.append(request)
        message = _Obj(content=f"{self.summary_prefix}-{len(self.create_requests)}")
        usage = _Obj(prompt_tokens=10, completion_tokens=2, total_tokens=12)
        return _Obj(choices=[_Obj(message=message)], usage=usage)


class _Obj:
    def __init__(self, **values):
        self.__dict__.update(values)
