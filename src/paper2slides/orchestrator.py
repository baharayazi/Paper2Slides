"""End-to-end pipeline: PDF -> article -> summary -> plan -> grounded plan -> PPTX."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .grounding import ground_figures
from .ingest import build_article, save_json, strip_references
from .llm import LLM, plan_presentation, summarize_article, translate_plan
from .render import build_render_plan, check_renderer, render_pptx

MIN_SLIDES = 4
MAX_SLIDES = 10

REASONING_LANGUAGE = "en"

Progress = Callable[[str], None]


@dataclass
class PipelineResult:
    article_json: Path
    summary_json: Path
    plan_json: Path
    final_plan_json: Path
    figures_dir: Path
    pptx: Path | None
    slide_count: int
    figures_found: int
    figures_used: int
    presentation_title: str


def _noop(_: str) -> None:
    return None


def run_pipeline(
    pdf_path: Path,
    *,
    slide_count: int = 5,
    language: str = "en",
    output_path: Path | None = None,
    work_dir: Path = Path("data"),
    llm: LLM | None = None,
    render: bool = True,
    progress: Progress = _noop,
) -> PipelineResult:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file '{pdf_path}' does not exist")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"'{pdf_path}' is not a PDF file")
    if not MIN_SLIDES <= slide_count <= MAX_SLIDES:
        raise ValueError(f"slide_count must be between {MIN_SLIDES} and {MAX_SLIDES}")

    if render:
        check_renderer()  # fail early, before spending minutes on the model

    work_dir = Path(work_dir)
    figures_dir = work_dir / "figures"
    output_path = Path(output_path) if output_path else Path("outputs") / f"{pdf_path.stem}.pptx"
    llm = llm or LLM()
    started = time.perf_counter()
    translate = language != REASONING_LANGUAGE
    total_steps = 6 if translate else 5
    step = 0

    def banner(message: str) -> None:
        nonlocal step
        step += 1
        progress(f"[{step}/{total_steps}] {message}")

    banner(f"Extracting text and figures from {pdf_path.name} ...")
    article = build_article(pdf_path, figures_dir)
    article_json = work_dir / "article.json"
    save_json(article.to_dict(), article_json)
    progress(
        f"      title: {article.title}\n"
        f"      pages: {article.page_count}, figures: {len(article.figures)}"
    )

    banner(f"Summarizing the paper with {llm.model} ...")
    body_text = strip_references(article.text)
    summary = summarize_article(llm, article.title, body_text, REASONING_LANGUAGE)
    summary_json = work_dir / "article_summary.json"
    save_json(summary, summary_json)
    progress(f"      {summary.get('one_sentence_summary', '')}")

    banner(f"Planning {slide_count} slides ...")
    plan = plan_presentation(
        llm, summary, article.figures, slide_count, REASONING_LANGUAGE, article.title
    )
    plan_json = work_dir / "presentation_plan.json"
    save_json(plan, plan_json)
    for number, slide in enumerate(plan["slides"], start=1):
        progress(f"      {number}. {slide['title']}")

    banner("Matching figures to slides ...")
    grounded = ground_figures(
        plan, article.figures, llm, REASONING_LANGUAGE, progress=progress
    )

    if translate:
        banner(f"Translating slide text to '{language}' ...")
        grounded = translate_plan(llm, grounded, language, progress=progress)

    render_plan = build_render_plan(grounded, article, language)
    final_plan_json = work_dir / "final_presentation_plan.json"
    save_json(render_plan, final_plan_json)
    figures_used = sum(1 for slide in grounded["slides"] if slide.get("image_path"))

    pptx: Path | None = None
    if render:
        banner(f"Rendering PowerPoint to {output_path} ...")
        pptx = render_pptx(final_plan_json, output_path)
    else:
        banner("Rendering skipped (--no-render)")

    elapsed = time.perf_counter() - started
    progress(f"Done in {elapsed:.0f}s: {slide_count} slides, {figures_used} with figures.")

    return PipelineResult(
        article_json=article_json,
        summary_json=summary_json,
        plan_json=plan_json,
        final_plan_json=final_plan_json,
        figures_dir=figures_dir,
        pptx=pptx,
        slide_count=slide_count,
        figures_found=len(article.figures),
        figures_used=figures_used,
        presentation_title=plan.get("presentation_title", article.title),
    )


def result_to_dict(result: PipelineResult) -> dict[str, Any]:
    return {
        "presentation_title": result.presentation_title,
        "slide_count": result.slide_count,
        "figures_found": result.figures_found,
        "figures_used": result.figures_used,
        "pptx": result.pptx.as_posix() if result.pptx else None,
        "article_json": result.article_json.as_posix(),
        "summary_json": result.summary_json.as_posix(),
        "plan_json": result.plan_json.as_posix(),
        "final_plan_json": result.final_plan_json.as_posix(),
    }
