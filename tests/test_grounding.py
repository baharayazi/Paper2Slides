from pathlib import Path

from paper2slides.grounding import (
    FigureIndex,
    ground_figures,
    mentioned_figure_numbers,
    stem,
    tokenize,
)
from paper2slides.llm import normalize_figure_id, suggested_outline
from paper2slides.models import Figure


def fig(figure_id: str, number: int, caption: str, context: list[str] | None = None) -> Figure:
    return Figure(
        figure_id=figure_id,
        page_number=number,
        path=Path(f"data/figures/{figure_id}.png"),
        width=800,
        height=600,
        caption=caption,
        figure_number=number,
        context=context or [],
    )


FIGURES = [
    fig("F1", 1, "Figure 1 | Overview of the proposed architecture with retriever and planner agents."),
    fig("F2", 2, "Figure 2 | Accuracy of our method versus baselines on three benchmarks.", ["Figure 2 shows that accuracy improves by 4.1%."]),
    fig("F3", 3, "Figure 3 | Annotation interface used for human evaluation."),
]


def test_tokenize_and_stem_merge_inflections():
    assert stem("baselines") == "baseline"
    assert stem("planning") == "plann"  # crude but consistent
    assert "architecture" in tokenize("The Architecture of the system")
    assert "the" not in tokenize("the system")


def test_mentioned_figure_numbers_handles_english_and_persian():
    assert mentioned_figure_numbers("see Figure 2 and Fig. 3") == {2, 3}
    assert mentioned_figure_numbers("در شکل ۲ و شکل 4") == {2, 4}  # Persian digits too


def test_lexical_ranking_prefers_matching_caption():
    index = FigureIndex(FIGURES)
    ranked = index.rank("Our method beats the baselines in accuracy on every benchmark")
    assert ranked[0][0].figure_id == "F2"

    ranked = index.rank("System architecture: retriever, planner and visualizer agents")
    assert ranked[0][0].figure_id == "F1"


def test_explicit_mention_boosts_figure():
    index = FigureIndex(FIGURES)
    ranked = index.rank("Unrelated words, but see Figure 3")
    assert ranked[0][0].figure_id == "F3"


def test_normalize_figure_id():
    valid = {"F1", "F2"}
    assert normalize_figure_id("F1", valid) == "F1"
    assert normalize_figure_id(" f2 ", valid) == "F2"
    assert normalize_figure_id("null", valid) is None
    assert normalize_figure_id("figures/figure9.png", valid) is None
    assert normalize_figure_id(None, valid) is None


def test_suggested_outline_has_requested_length_and_arc():
    for count in range(4, 11):
        outline = suggested_outline(count)
        assert len(outline) == count
        assert outline[0].startswith("Problem")
        assert outline[-1].startswith("Conclusion")


def fake_judge(slide, candidates):
    """Pick the candidate whose caption shares the most words with the slide title."""
    title_words = set(slide["title"].lower().split())
    best, best_overlap = None, 0
    for candidate in candidates:
        overlap = len(title_words & set(candidate.caption.lower().split()))
        if overlap > best_overlap:
            best, best_overlap = candidate, overlap
    if best is None:
        return {"figure_id": None, "relevance": 1, "explanation": ""}
    return {"figure_id": best.figure_id, "relevance": 5, "explanation": f"shows {best.figure_id}"}


def test_ground_figures_assigns_each_figure_once_and_fills_fields():
    plan = {
        "presentation_title": "T",
        "slides": [
            {"title": "Motivation", "key_message": "Why", "bullets": ["x"], "visual_need": "", "figure_id": None},
            {"title": "Accuracy versus baselines", "key_message": "better", "bullets": ["4.1% gain"], "visual_need": "bar chart of accuracy", "figure_id": "F2"},
            {"title": "Accuracy on benchmarks again", "key_message": "still better", "bullets": ["same"], "visual_need": "accuracy chart", "figure_id": "F2"},
            {"title": "Architecture overview", "key_message": "agents", "bullets": ["retriever", "planner"], "visual_need": "architecture diagram", "figure_id": None},
        ],
    }

    grounded = ground_figures(plan, FIGURES, llm=None, language="en", judge=fake_judge)
    slides = grounded["slides"]

    used = [slide["image_figure_id"] for slide in slides if slide["image_figure_id"]]
    assert len(used) == len(set(used)), "a figure must not appear on two slides"
    assert slides[0]["image_path"] is None
    assert slides[1]["image_figure_id"] == "F2"
    assert slides[1]["image_path"].endswith("F2.png")
    assert slides[1]["image_page"] == 2
    assert slides[1]["image_explanation"] == "shows F2"
    assert slides[2]["image_figure_id"] is None  # lost the conflict, nothing else fits
    assert slides[3]["image_figure_id"] == "F1"
    assert "figure_id" not in slides[1] and slides[1]["planner_figure_id"] == "F2"
    # the original plan is untouched
    assert plan["slides"][1]["figure_id"] == "F2"


def test_clean_visual_need_drops_placeholder_phrases():
    from paper2slides.llm import clean_visual_need

    assert clean_visual_need("empty string") == ""
    assert clean_visual_need("None.") == ""
    assert clean_visual_need(None) == ""
    assert clean_visual_need(" bar chart of accuracy ") == "bar chart of accuracy"


def test_slide_without_visual_need_or_planner_pick_stays_text_only():
    plan = {
        "presentation_title": "T",
        "slides": [
            {"title": "Accuracy versus baselines", "key_message": "", "bullets": ["x"], "visual_need": "", "figure_id": None},
        ],
    }
    calls = []

    def spy(slide, candidates):
        calls.append(slide["title"])
        return {"figure_id": "F2", "relevance": 5, "explanation": "x"}

    grounded = ground_figures(plan, FIGURES, llm=None, language="en", judge=spy)
    assert calls == []
    assert grounded["slides"][0]["image_figure_id"] is None


def test_planner_pick_without_visual_need_needs_perfect_relevance():
    plan = {
        "presentation_title": "T",
        "slides": [
            {"title": "Conclusion", "key_message": "", "bullets": ["x"], "visual_need": "", "figure_id": "F1"},
        ],
    }
    grounded = ground_figures(
        plan, FIGURES, llm=None, language="en",
        judge=lambda slide, candidates: {"figure_id": "F1", "relevance": 4, "explanation": "ok"},
    )
    assert grounded["slides"][0]["image_figure_id"] is None

    grounded = ground_figures(
        plan, FIGURES, llm=None, language="en",
        judge=lambda slide, candidates: {"figure_id": "F1", "relevance": 5, "explanation": "ok"},
    )
    assert grounded["slides"][0]["image_figure_id"] == "F1"


def test_wrong_script_detects_english_left_in_persian_output():
    from paper2slides.llm import _wrong_script

    assert _wrong_script("fa", "Quantitative Performance Gains")
    assert not _wrong_script("fa", "دقت نسبت به مدل پایه ۴.۱٪ بهبود می‌یابد")
    assert not _wrong_script("fa", "PaperBanana")  # a lone proper noun is fine
    assert _wrong_script("fa", "اتوماسیون ایлюстраشن‌های آکادمیک")  # Cyrillic leaked in
    assert not _wrong_script("en", "anything goes")
