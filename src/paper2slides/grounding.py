"""
Visual grounding: choose the most relevant figure for each slide.

Two stages:

1. A lexical ranker (TF-IDF over caption + in-text mentions) shortlists the
   figures that share vocabulary with the slide text and its English
   ``visual_need`` description. Explicit "Figure N" mentions get a boost.
2. An LLM judge looks at the shortlist (plus the planner's own pick), chooses
   at most one figure, scores its relevance 1-5 and writes the one-sentence
   explanation that is shown under the figure.

Each figure is used on at most one slide; conflicts are resolved in favor of
the slide where the judge was more confident.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .llm import LLM, LLMError, judge_figure_for_slide, normalize_figure_id
from .models import Figure

logger = logging.getLogger(__name__)

MAX_CANDIDATES = 3
MIN_LEXICAL_SCORE = 0.02
MIN_RELEVANCE = 3
MIN_RELEVANCE_WITHOUT_VISUAL_NEED = 5
EXPLICIT_MENTION_BOOST = 1.0

STOPWORDS = frozenset(
    ["a", "an", "the", "of", "and", "or", "to", "in", "on", "for", "with", "by", "from", "as", "is", "are", "was", "were", "be", "been", "being", "this", "that", "these", "those", "it", "its", "we", "our", "their", "they", "them", "at", "into", "over", "under", "between", "across", "which", "who", "whose", "than", "then", "thus", "vs", "versus", "via", "using", "used", "use", "based", "show", "shows", "shown", "showing", "figure", "fig", "image", "images", "result", "results", "method", "methods", "approach", "paper", "proposed", "propose", "model", "models", "data", "set", "sets", "table", "also", "each", "both", "more", "most", "less", "can", "may", "might", "such", "e", "g", "i", "et", "al", "one", "two", "three", "first", "second", "new", "different", "same", "other", "all", "any", "some", "when", "where", "while", "how", "what", "why", "not", "no", "yes", "per", "within", "without", "up", "down", "out", "about", "after", "before", "during", "through", "against"]
)

SUFFIXES = (
    "ization", "isation", "ations", "ation", "ingly", "ings", "ing",
    "edly", "ness", "ed", "ies", "s",
)

FIGURE_MENTION = re.compile(r"(?:figure|fig\.?|شکل|نمودار|تصویر)\s*(\d+)", re.IGNORECASE)
TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\-]+|\d+(?:\.\d+)?%?|[؀-ۿ]{2,}")

Judge = Callable[[dict[str, Any], list[Figure]], dict[str, Any]]


def stem(token: str) -> str:
    """A deliberately tiny stemmer: enough to merge plurals and -ing forms."""
    for suffix in SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            root = token[: -len(suffix)]
            if suffix == "ies":
                return root + "y"
            if suffix == "s" and root.endswith(("ss", "x", "z", "ch", "sh")) and root.endswith("e"):
                return root[:-1]  # boxes -> box
            return root
    return token


def tokenize(text: str) -> list[str]:
    tokens = []
    for raw in TOKEN.findall(text.lower()):
        token = raw.strip("-")
        if not token or token in STOPWORDS:
            continue
        if len(token) < 3 and not token[0].isdigit():
            continue
        tokens.append(stem(token))
    return tokens


def mentioned_figure_numbers(text: str) -> set[int]:
    return {int(match.group(1)) for match in FIGURE_MENTION.finditer(text)}


def slide_query(slide: dict[str, Any]) -> str:
    parts = [
        slide.get("title") or "",
        slide.get("key_message") or "",
        *(slide.get("bullets") or []),
        slide.get("visual_need") or "",
        slide.get("speaker_notes") or "",
    ]
    return " ".join(str(part) for part in parts)


class FigureIndex:
    """TF-IDF index over figure captions and their in-text mentions."""

    def __init__(self, figures: list[Figure]) -> None:
        self.figures = list(figures)
        documents = [
            Counter(tokenize(figure.caption) * 2 + tokenize(" ".join(figure.context)))
            for figure in self.figures
        ]
        document_frequency: Counter[str] = Counter()
        for document in documents:
            document_frequency.update(document.keys())

        size = len(documents)
        self._default_idf = math.log((size + 1) / 1) + 1
        self.idf = {
            token: math.log((size + 1) / (count + 1)) + 1
            for token, count in document_frequency.items()
        }
        self.vectors = [self._vector(document) for document in documents]

    def _vector(self, counts: Counter[str]) -> dict[str, float]:
        vector = {
            token: (1 + math.log(count)) * self.idf.get(token, self._default_idf)
            for token, count in counts.items()
        }
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        return {token: value / norm for token, value in vector.items()}

    def rank(self, query: str) -> list[tuple[Figure, float]]:
        query_vector = self._vector(Counter(tokenize(query)))
        mentioned = mentioned_figure_numbers(query)

        ranked = []
        for figure, vector in zip(self.figures, self.vectors):
            score = sum(weight * vector.get(token, 0.0) for token, weight in query_vector.items())
            if figure.figure_number is not None and figure.figure_number in mentioned:
                score += EXPLICIT_MENTION_BOOST
            ranked.append((figure, score))

        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked


@dataclass
class SlideGrounding:
    figure: Figure | None = None
    relevance: int = 0
    explanation: str = ""
    lexical_score: float = 0.0
    candidates: list[str] = field(default_factory=list)


def _shortlist(
    slide: dict[str, Any],
    index: FigureIndex,
    by_id: dict[str, Figure],
    excluded: set[str],
    max_candidates: int,
) -> tuple[list[Figure], dict[str, float]]:
    ranked = index.rank(slide_query(slide))
    scores = {figure.figure_id: score for figure, score in ranked}

    candidates: list[Figure] = []
    planner_pick = normalize_figure_id(slide.get("planner_figure_id"), set(by_id))
    if planner_pick and planner_pick not in excluded:
        candidates.append(by_id[planner_pick])

    for figure, score in ranked:
        if len(candidates) >= max_candidates + (1 if planner_pick else 0):
            break
        if score < MIN_LEXICAL_SCORE or figure.figure_id in excluded:
            continue
        if all(figure.figure_id != existing.figure_id for existing in candidates):
            candidates.append(figure)

    return candidates, scores


def _judge_slide(
    slide: dict[str, Any],
    candidates: list[Figure],
    scores: dict[str, float],
    judge: Judge,
    min_relevance: int,
) -> SlideGrounding:
    grounding = SlideGrounding(candidates=[figure.figure_id for figure in candidates])
    if not candidates:
        return grounding

    try:
        verdict = judge(slide, candidates)
    except LLMError as err:
        logger.warning("Figure judge failed for slide '%s': %s", slide.get("title"), err)
        return grounding

    chosen_id = verdict.get("figure_id")
    chosen = next((figure for figure in candidates if figure.figure_id == chosen_id), None)
    relevance = int(verdict.get("relevance") or 0)
    if chosen is None or relevance < min_relevance:
        return grounding

    grounding.figure = chosen
    grounding.relevance = relevance
    grounding.explanation = verdict.get("explanation") or chosen.caption
    grounding.lexical_score = scores.get(chosen.figure_id, 0.0)
    return grounding


def ground_figures(
    plan: dict[str, Any],
    figures: list[Figure],
    llm: LLM | None,
    language: str,
    *,
    judge: Judge | None = None,
    max_candidates: int = MAX_CANDIDATES,
    min_relevance: int = MIN_RELEVANCE,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Return a copy of the plan with verified image fields on every slide."""
    if judge is None:
        if llm is None:
            raise ValueError("ground_figures needs either an LLM or a judge callable")
        judge = lambda slide, candidates: judge_figure_for_slide(llm, slide, candidates, language)

    plan = deepcopy(plan)
    slides = plan["slides"]
    by_id = {figure.figure_id: figure for figure in figures}
    index = FigureIndex(figures)

    for slide in slides:
        slide["planner_figure_id"] = normalize_figure_id(slide.pop("figure_id", None), set(by_id))

    def threshold(slide: dict[str, Any]) -> int:
        if slide.get("visual_need"):
            return min_relevance
        return max(min_relevance, MIN_RELEVANCE_WITHOUT_VISUAL_NEED)

    groundings: list[SlideGrounding] = []
    for number, slide in enumerate(slides, start=1):
        if not slide.get("visual_need") and not slide.get("planner_figure_id"):
            groundings.append(SlideGrounding())  # planner wants a text-only slide
            if progress:
                progress(f"  slide {number}: text-only (no visual requested)")
            continue
        candidates, scores = _shortlist(slide, index, by_id, set(), max_candidates)
        grounding = _judge_slide(slide, candidates, scores, judge, threshold(slide))
        groundings.append(grounding)
        if progress:
            chosen = grounding.figure.figure_id if grounding.figure else "none"
            progress(
                f"  slide {number}: candidates {grounding.candidates or '[]'} -> {chosen}"
                + (f" (relevance {grounding.relevance}/5)" if grounding.figure else "")
            )

    # One figure per presentation: keep the most confident slide, re-judge the rest.
    taken: set[str] = set()
    order = sorted(
        range(len(slides)),
        key=lambda i: (groundings[i].relevance, groundings[i].lexical_score),
        reverse=True,
    )
    for position in order:
        grounding = groundings[position]
        if grounding.figure is None:
            continue
        if grounding.figure.figure_id not in taken:
            taken.add(grounding.figure.figure_id)
            continue

        slide = slides[position]
        if not slide.get("visual_need"):
            groundings[position] = SlideGrounding(candidates=grounding.candidates)
            if progress:
                progress(f"  slide {position + 1}: figure already used, slide stays text-only")
            continue
        candidates, scores = _shortlist(slide, index, by_id, taken, max_candidates)
        retry = _judge_slide(slide, candidates, scores, judge, threshold(slide))
        if retry.figure is not None and retry.figure.figure_id not in taken:
            taken.add(retry.figure.figure_id)
            groundings[position] = retry
        else:
            groundings[position] = SlideGrounding(candidates=retry.candidates)
        if progress:
            chosen = groundings[position].figure.figure_id if groundings[position].figure else "none"
            progress(f"  slide {position + 1}: figure already used, re-judged -> {chosen}")

    for slide, grounding in zip(slides, groundings):
        figure = grounding.figure
        slide["candidate_figure_ids"] = grounding.candidates
        slide["image_figure_id"] = figure.figure_id if figure else None
        slide["image_path"] = figure.path.as_posix() if figure else None
        slide["image_caption"] = figure.caption if figure else None
        slide["image_page"] = figure.page_number if figure else None
        slide["image_relevance"] = grounding.relevance if figure else None
        slide["image_explanation"] = grounding.explanation if figure else None

    return plan
