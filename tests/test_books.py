import zipfile
from pathlib import Path

import pytest

import books
import read_books
from books import UnsupportedBook, book_format, open_book, split_into_pages
from test_read_books import FakeBackend, FakeDocument, page_of, run


def make_epub(path, chapters, spine, encryption=None):
    """A tiny EPUB: `chapters` maps file name to XHTML body, `spine` gives the reading order."""
    with zipfile.ZipFile(path, "w") as book:
        book.writestr("mimetype", "application/epub+zip")
        book.writestr("META-INF/container.xml", """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>""")
        if encryption:
            book.writestr("META-INF/encryption.xml", encryption)
        items = "".join(
            f'<item id="{Path(name).stem}" href="{name}" media-type="application/xhtml+xml"/>'
            for name in chapters
        )
        itemrefs = "".join(f'<itemref idref="{Path(name).stem}"/>' for name in spine)
        book.writestr("OEBPS/content.opf", f"""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <manifest>{items}<item id="css" href="style.css" media-type="text/css"/></manifest>
  <spine>{itemrefs}</spine>
</package>""")
        book.writestr("OEBPS/style.css", "p { color: red; }")
        for name, body in chapters.items():
            book.writestr(f"OEBPS/{name}", f"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Running Head</title>
<style>p {{ margin: 0 }}</style></head><body>{body}</body></html>""")


def make_docx(path, paragraphs):
    """A tiny Word file; each paragraph is a list of runs: text, "\\t" for a tab, "\\n" for a break."""
    def run_xml(part):
        if part == "\t":
            return "<w:r><w:tab/></w:r>"
        if part == "\n":
            return "<w:r><w:br/></w:r>"
        return f"<w:r><w:t xml:space=\"preserve\">{part}</w:t></w:r>"

    body = "".join(f"<w:p>{''.join(run_xml(part) for part in runs)}</w:p>" for runs in paragraphs)
    with zipfile.ZipFile(path, "w") as document:
        document.writestr("word/document.xml", f"""<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>{body}</w:body></w:document>""")


# --- reader pages --------------------------------------------------------------------------------


def test_pages_are_cut_at_paragraph_breaks_and_are_the_same_every_time():
    text = "\n\n".join(f"Paragraph {number}. " + "word " * 40 for number in range(30))

    pages = split_into_pages(text, size=500)

    assert pages == split_into_pages(text, size=500)
    assert len(pages) > 1 and all(len(page) <= 500 for page in pages)
    assert all(page.startswith("Paragraph ") for page in pages)  # no paragraph is cut in two
    assert "\n\n".join(pages) == "\n\n".join(books.paragraphs(text))


def test_a_long_paragraph_is_split_at_sentence_ends():
    paragraph = " ".join(f"Sentence number {number} ends here." for number in range(60))

    pages = split_into_pages(paragraph, size=300)

    assert len(pages) > 1 and all(len(page) <= 300 for page in pages)
    assert all(page.endswith("ends here.") for page in pages)
    assert " ".join(pages) == paragraph


def test_one_enormous_sentence_is_cut_at_a_space_as_a_last_resort():
    sentence = "word " * 200

    pages = split_into_pages(sentence, size=100)

    assert all(len(page) <= 100 for page in pages)
    assert " ".join(pages) == sentence.strip()


# --- formats -------------------------------------------------------------------------------------


def test_epub_follows_the_spine_and_skips_the_head(tmp_path):
    epub = tmp_path / "book.epub"
    make_epub(epub, {
        "a.xhtml": "<h1>Chapter One</h1><p>First   chapter\ntext.</p><script>hidden()</script>",
        "b.xhtml": "<h1>Chapter Two</h1><p>Second <em>chapter</em> text.</p>",
        "notes.xhtml": "<p>Notes.</p>",
    }, spine=["b.xhtml", "a.xhtml"])

    document, layout = open_book(epub)
    text = document[0].get_text()

    assert layout == {"format": "epub", "page_chars": books.PAGE_CHARS}
    assert text == "Chapter Two\n\nSecond chapter text.\n\nChapter One\n\nFirst chapter text."
    assert "Running Head" not in text and "hidden" not in text and "color" not in text
    assert "Notes." not in text  # not in the spine


def test_a_drm_protected_epub_says_so(tmp_path):
    epub = tmp_path / "locked.epub"
    make_epub(epub, {"a.xhtml": "<p>Secret.</p>"}, spine=["a.xhtml"], encryption="""<?xml version="1.0"?>
<encryption xmlns="urn:oasis:names:tc:opendocument:xmlns:container" xmlns:enc="http://www.w3.org/2001/04/xmlenc#">
  <enc:EncryptedData><enc:CipherData><enc:CipherReference URI="OEBPS/a.xhtml"/></enc:CipherData></enc:EncryptedData>
</encryption>""")

    with pytest.raises(UnsupportedBook, match="DRM-protected"):
        open_book(epub)


def test_an_epub_with_only_its_fonts_obfuscated_still_reads(tmp_path):
    epub = tmp_path / "fonts.epub"
    make_epub(epub, {"a.xhtml": "<p>Open text.</p>"}, spine=["a.xhtml"], encryption="""<?xml version="1.0"?>
<encryption xmlns="urn:oasis:names:tc:opendocument:xmlns:container" xmlns:enc="http://www.w3.org/2001/04/xmlenc#">
  <enc:EncryptedData><enc:CipherData><enc:CipherReference URI="OEBPS/fonts/serif.otf"/></enc:CipherData></enc:EncryptedData>
</encryption>""")

    document, _layout = open_book(epub)

    assert document[0].get_text() == "Open text."


def test_a_damaged_epub_gets_a_clear_message(tmp_path):
    epub = tmp_path / "broken.epub"
    epub.write_bytes(b"not a zip file")

    with pytest.raises(UnsupportedBook, match="doesn't look like a valid EPUB"):
        open_book(epub)


def test_docx_paragraphs_tabs_and_breaks(tmp_path):
    docx = tmp_path / "notes.docx"
    make_docx(docx, [["Title"], [], ["Name", "\t", "Value"], ["Line one", "\n", "line two &amp; more"]])

    document, layout = open_book(docx)

    assert layout["format"] == "docx"
    assert document[0].get_text() == "Title\n\nName\tValue\n\nLine one\nline two & more"


def test_markdown_and_text_are_read_as_written(tmp_path):
    markdown = tmp_path / "notes.md"
    markdown.write_text("# Heading\n\nSome *text* and a [link](https://example.com).\n", encoding="utf-8")

    document, layout = open_book(markdown)

    assert layout["format"] == "markdown" and document.page_count == 1
    assert document[0].get_text() == "# Heading\n\nSome *text* and a [link](https://example.com)."
    assert book_format(Path("story.txt")) == "text" and book_format(Path("Book.PDF")) == "pdf"


def test_an_old_windows_text_file_still_reads(tmp_path):
    text = tmp_path / "old.txt"
    text.write_bytes("Caf\u00e9 \u201cquotes\u201d \u2013 done.".encode("cp1252"))

    document, _layout = open_book(text)

    assert document[0].get_text() == "Caf\u00e9 \u201cquotes\u201d \u2013 done."


@pytest.mark.parametrize("name, message", [
    ("book.mobi", "Convert the book to EPUB"),
    ("book.azw3", "Calibre"),
    ("report.doc", "save it as .docx"),
    ("photo.jpg", "the reader takes PDF, EPUB"),
])
def test_unsupported_files_say_what_to_do(name, message):
    with pytest.raises(UnsupportedBook, match=message):
        book_format(Path(name))


# --- reading them ---------------------------------------------------------------------------------


def test_resume_catches_a_change_in_how_the_book_was_cut_into_pages(tmp_path):
    layout = {"format": "epub", "page_chars": 2400}
    run(FakeDocument(["one", "two"]), FakeBackend(), tmp_path, max_pages=1, document_id="same", page_layout=layout)

    with pytest.raises(read_books.CannotResume, match="cut into pages differently"):
        run(FakeDocument(["one", "two"]), FakeBackend(), tmp_path, document_id="same",
            page_layout={"format": "epub", "page_chars": 3000})

    later = FakeBackend()
    run(FakeDocument(["one", "two"]), later, tmp_path, document_id="same", page_layout=layout)
    assert [page_of(call["prompt"]) for call in later.page_calls()] == [2]


def test_main_reads_an_epub_end_to_end(tmp_path, monkeypatch, capsys):
    epub = tmp_path / "tiny.epub"
    long_chapter = "".join(f"<p>Idea {number}. {'More words here. ' * 30}</p>" for number in range(12))
    make_epub(epub, {"one.xhtml": long_chapter}, spine=["one.xhtml"])
    reader = FakeBackend()
    monkeypatch.setattr(read_books, "make_backends", lambda _args: (reader, reader))

    code = read_books.main([str(epub), "--base-dir", str(tmp_path / "out")])

    assert code == 0
    out = capsys.readouterr().out
    assert "reader pages of about 2,400 characters" in out and "📖 Reading page 3/3" in out
    assert "Idea 0." in reader.page_calls()[0]["prompt"]
    assert (tmp_path / "out" / "books" / "tiny.epub").exists()
    final = (tmp_path / "out" / "summaries" / "tiny_final.md").read_text(encoding="utf-8")
    assert "not the printed edition's" in final
    state = read_books.load_state(tmp_path / "out" / "knowledge_bases" / "tiny_knowledge.json")
    assert state["page_layout"] == {"format": "epub", "page_chars": 2400}


def test_main_turns_away_a_mobi_before_making_any_folders(tmp_path, capsys):
    mobi = tmp_path / "book.mobi"
    mobi.write_bytes(b"not read")

    assert read_books.main([str(mobi), "--base-dir", str(tmp_path / "out")]) == 2
    assert "Calibre" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def test_a_book_with_no_text_stops_with_a_hint(tmp_path, monkeypatch, capsys):
    epub = tmp_path / "comic.epub"
    make_epub(epub, {"a.xhtml": '<p><img src="page1.jpg" alt=""/></p>'}, spine=["a.xhtml"])
    reader = FakeBackend()
    monkeypatch.setattr(read_books, "make_backends", lambda _args: (reader, reader))

    assert read_books.main([str(epub), "--base-dir", str(tmp_path / "out")]) == 2
    out = capsys.readouterr().out
    assert "No text found in comic.epub" in out and "--ocr auto" in out
    assert reader.calls == []


def test_ocr_is_only_for_pdfs(tmp_path):
    with pytest.raises(SystemExit):
        read_books.main([str(tmp_path / "book.epub"), "--ocr", "auto"])
