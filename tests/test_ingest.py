from pathlib import Path

import pymupdf

from paper2slides.ingest import (
    build_article,
    find_article_title,
    parse_caption,
    strip_references,
)


def make_paper(path: Path) -> None:
    """A two-page synthetic paper: title, a vector figure with caption, references."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 80), "Learning Widgets With Gadgets", fontsize=20)
    page.insert_text((72, 110), "Alice Author, Bob Author", fontsize=11)
    page.insert_text(
        (72, 140),
        "We study widgets. As shown in Figure 1, accuracy improves with more gadgets.",
        fontsize=10,
    )
    page.draw_rect(pymupdf.Rect(72, 170, 320, 330), color=(0, 0, 1), fill=(0.8, 0.8, 1), width=1)
    page.draw_line((90, 310), (300, 200), color=(1, 0, 0), width=2)
    page.insert_text((72, 350), "Figure 1 | Accuracy of widgets versus number of gadgets.", fontsize=9)
    page.insert_text((72, 400), "Figure 1 compares the two settings in detail.", fontsize=10)

    page2 = doc.new_page(width=595, height=842)
    page2.insert_text((72, 80), "References", fontsize=12)
    page2.insert_text((72, 100), "[1] Someone. Something. 2020.", fontsize=9)
    doc.save(path)
    doc.close()


def test_parse_caption_accepts_real_captions_and_rejects_body_text():
    assert parse_caption("Figure 1 | Accuracy of widgets.") == 1
    assert parse_caption("Fig. 12: Overview of the system") == 12
    assert parse_caption("Figure 3. Results.") == 3
    assert parse_caption("Figure 4 compares " + "x" * 300) is None
    assert parse_caption("In Figure 2 we show") is None
    assert parse_caption("Table 1 | Results") is None


def test_strip_references_keeps_body_and_drops_bibliography():
    body = "Intro\n" + "word " * 2000 + "\nReferences\n[1] A. Author. Title. 2020.\n"
    stripped = strip_references(body)
    assert "[1] A. Author" not in stripped
    assert stripped.startswith("Intro")


def test_strip_references_keeps_appendix():
    body = "Intro\n" + "word " * 2000 + "\nReferences\n[1] A. Author.\n\nAppendix A Extra results\nmore\n"
    stripped = strip_references(body)
    assert "[1] A. Author" not in stripped
    assert "Appendix A Extra results" in stripped


def test_build_article_extracts_title_and_vector_figure(tmp_path: Path):
    pdf = tmp_path / "widgets_paper.pdf"
    make_paper(pdf)

    article = build_article(pdf, tmp_path / "figures")

    assert article.title == "Learning Widgets With Gadgets"
    assert article.page_count == 2
    assert len(article.figures) == 1

    figure = article.figures[0]
    assert figure.figure_number == 1
    assert figure.caption.startswith("Figure 1 | Accuracy")
    assert figure.path.exists() and figure.path.suffix == ".png"
    assert figure.width > 100 and figure.height > 50
    # Body sentences mentioning the figure become its context; the caption itself does not.
    assert any("accuracy improves" in sentence for sentence in figure.context)
    assert all(not sentence.startswith("Figure 1 |") for sentence in figure.context)


def test_title_falls_back_to_file_name(tmp_path: Path):
    doc = pymupdf.open()
    doc.new_page()
    pdf = tmp_path / "my_cool_paper.pdf"
    doc.save(pdf)
    with pymupdf.open(pdf) as opened:
        assert find_article_title(opened, pdf) == "my cool paper"
