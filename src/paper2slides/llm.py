"""
Local LLM agents backed by Ollama.

Every agent call goes through :meth:`LLM.chat_json`, which enforces a JSON
schema, uses a large enough context window, retries with feedback when the
model returns something invalid, and disables Qwen-style "thinking" so the
answer comes back quickly.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from dotenv import load_dotenv
from ollama import Client, ResponseError

from .models import Figure

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen3:4b"
DEFAULT_NUM_CTX = 16_384
CHARACTERS_PER_TOKEN = 3.2  # conservative for English prose
RESERVED_TOKENS = 3_000  # system prompt + JSON answer

Validator = Callable[[dict[str, Any]], str | None]


class LLMError(RuntimeError):
    """Raised when the local model cannot produce a usable answer."""


def _string_array(min_items: int, max_items: int) -> dict[str, Any]:
    return {
        "type": "array",
        "minItems": min_items,
        "maxItems": max_items,
        "items": {"type": "string"},
    }


RESEARCH_SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "one_sentence_summary": {"type": "string"},
        "background": {"type": "string"},
        "research_question": {"type": "string"},
        "method": {"type": "string"},
        "main_findings": _string_array(2, 6),
        "key_results": _string_array(0, 6),
        "limitations": _string_array(0, 4),
        "conclusion": {"type": "string"},
        "keywords": _string_array(3, 8),
    },
    "required": [
        "one_sentence_summary",
        "background",
        "research_question",
        "method",
        "main_findings",
        "key_results",
        "limitations",
        "conclusion",
        "keywords",
    ],
    "additionalProperties": False,
}


def slide_schema(figure_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "key_message": {"type": "string"},
            "bullets": _string_array(1, 4),
            "speaker_notes": {"type": "string"},
            "visual_need": {"type": "string"},
            "figure_id": {"type": ["string", "null"], "enum": [*figure_ids, None]},
        },
        "required": [
            "title",
            "key_message",
            "bullets",
            "speaker_notes",
            "visual_need",
            "figure_id",
        ],
        "additionalProperties": False,
    }


def presentation_plan_schema(slide_count: int, figure_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "presentation_title": {"type": "string"},
            "slides": {
                "type": "array",
                "minItems": slide_count,
                "maxItems": slide_count,
                "items": slide_schema(figure_ids),
            },
        },
        "required": ["presentation_title", "slides"],
        "additionalProperties": False,
    }


SCRIPT_CHECKS = {
    "fa": re.compile(r"[\u0600-\u06FF]"),
    "ar": re.compile(r"[\u0600-\u06FF]"),
    "ur": re.compile(r"[\u0600-\u06FF]"),
    "he": re.compile(r"[\u0590-\u05FF]"),
    "ru": re.compile(r"[\u0400-\u04FF]"),
    "zh": re.compile(r"[\u4e00-\u9fff]"),
    "ja": re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]"),
    "ko": re.compile(r"[\uac00-\ud7af]"),
}


def figure_judgement_schema(figure_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "figure_id": {"type": ["string", "null"], "enum": [*figure_ids, None]},
            "relevance": {"type": "integer", "minimum": 1, "maximum": 5},
            "explanation": {"type": "string"},
        },
        "required": ["figure_id", "relevance", "explanation"],
        "additionalProperties": False,
    }

def get_language_instruction(language: str) -> str:
    """Return a clear natural-language instruction for the output language."""
    instructions = {
        "fa": (
            "Persian (Farsi), written entirely in Persian script. "
            "Do not write English except for unavoidable proper nouns, "
            "acronyms and metric names."
        ),
        "en": "English",
        "de": "German",
        "fr": "French",
        "es": "Spanish",
        "ar": "Arabic, written entirely in Arabic script",
        "tr": "Turkish",
    }
    return instructions.get(language, language)


def normalize_figure_id(value: Any, valid_ids: set[str]) -> str | None:
    """Map whatever the model returned to a known figure id, or None."""
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "null", "none", "no", "n/a", "-"}:
        return None
    if text in valid_ids:
        return text
    upper = text.upper().replace(" ", "")
    for candidate in valid_ids:
        if candidate.upper() == upper:
            return candidate
    return None


NO_VISUAL_PHRASES = {
    "", "empty string", "empty", "none", "null", "n/a", "no figure", "no figure needed",
    "no image", "no image needed", "not needed", "-",
}


def clean_visual_need(value: Any) -> str:
    """Normalise the planner's visual_need; small models write 'empty string' literally."""
    text = (str(value) if value is not None else "").strip().strip("\"'")
    if text.lower().rstrip(".") in NO_VISUAL_PHRASES:
        return ""
    return text


def suggested_outline(slide_count: int) -> list[str]:
    """A default narrative arc the planner can adapt to the paper."""
    outline = [
        "Problem and motivation (why this paper matters)",
        "Approach / method",
        "Key results",
        "Conclusion and takeaways",
    ]
    extras = [
        (1, "Background and research question"),
        (-1, "Limitations and future work"),
        (3, "Experimental setup"),
        (-2, "Additional results and analysis"),
        (3, "Method details"),
        (-2, "Discussion"),
    ]
    for position, topic in extras:
        if len(outline) >= slide_count:
            break
        if position < 0:
            outline.insert(len(outline) + position + 1 - 1, topic)
        else:
            outline.insert(min(position, len(outline) - 1), topic)
    while len(outline) < slide_count:
        outline.insert(len(outline) - 1, "Additional analysis")
    return outline[:slide_count]


def figure_catalog(figures: list[Figure], max_caption: int = 400) -> list[dict[str, Any]]:
    """Compact figure descriptions handed to the planner."""
    catalog = []
    for figure in figures:
        caption = figure.caption.strip()
        if len(caption) > max_caption:
            caption = caption[: max_caption - 1].rsplit(" ", 1)[0] + "…"
        catalog.append(
            {
                "figure_id": figure.figure_id,
                "page": figure.page_number,
                "caption": caption or "(no caption found)",
            }
        )
    return catalog


class LLM:
    """Thin wrapper over the Ollama chat API with schema-constrained output."""

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        num_ctx: int | None = None,
    ) -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.num_ctx = int(num_ctx or os.getenv("OLLAMA_NUM_CTX", DEFAULT_NUM_CTX))
        self.client = Client(host=self.host)
        self._think_supported = True

    @property
    def max_article_characters(self) -> int:
        return max(12_000, int((self.num_ctx - RESERVED_TOKENS) * CHARACTERS_PER_TOKEN))

    def chat_json(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        *,
        validate: Validator | None = None,
        max_attempts: int = 3,
        num_predict: int = 4_096,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            *(history or []),
            {"role": "user", "content": user},
        ]
        last_error = "unknown error"

        for attempt in range(1, max_attempts + 1):
            content = self._chat(messages, schema, num_predict)
            try:
                data = json.loads(content)
            except json.JSONDecodeError as err:
                last_error = f"the answer was not valid JSON ({err.msg})"
            else:
                problem = validate(data) if validate else None
                if problem is None:
                    return data
                last_error = problem

            logger.warning("LLM attempt %d rejected: %s", attempt, last_error)
            messages = messages + [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        f"Your previous answer was rejected: {last_error}. "
                        "Return a corrected JSON object that follows the "
                        "schema and the instructions exactly."
                    ),
                },
            ]

        raise LLMError(
            f"The model '{self.model}' did not return a usable answer after "
            f"{max_attempts} attempts: {last_error}"
        )

    def _chat(self, messages: list[dict[str, str]], schema: dict[str, Any], num_predict: int) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "format": schema,
            "options": {
                "temperature": 0,
                "num_ctx": self.num_ctx,
                "num_predict": num_predict,
            },
            "keep_alive": "10m",
        }
        if self._think_supported:
            kwargs["think"] = False

        try:
            response = self.client.chat(**kwargs)
        except ResponseError as err:
            message = str(err).lower()
            if self._think_supported and "think" in message:
                self._think_supported = False
                return self._chat(messages, schema, num_predict)
            if "not found" in message:
                raise LLMError(
                    f"Model '{self.model}' is not available in Ollama. "
                    f"Run: ollama pull {self.model}"
                ) from err
            raise LLMError(f"Ollama request failed: {err}") from err
        except ConnectionError as err:
            raise LLMError(
                f"Cannot reach Ollama at {self.host}. Start it with `ollama serve`."
            ) from err

        return response.message.content or ""


def summarize_article(llm: LLM, title: str, text: str, language: str) -> dict[str, Any]:
    """Research-reader agent: structured summary of the paper."""
    language_instruction = get_language_instruction(language)
    article_text = text[: llm.max_article_characters]
    truncated = len(text) > len(article_text)

    system = (
        "You are a careful research reader. Read the academic paper and "
        "produce a structured summary that a presenter could build slides "
        "from. Rules: use only information present in the paper; never invent "
        "facts, numbers or citations; quote numbers exactly as reported; "
        "`key_results` lists the most important quantitative results with "
        "their metric and the comparison baseline; `keywords` are 3-8 short "
        "English technical terms (always English, used for figure matching). "
        f"Write every other JSON value in {language_instruction}."
    )
    user = (
        f"Paper title: {title}\n\n"
        + ("(The paper was truncated to fit the context window.)\n\n" if truncated else "")
        + f"Paper text:\n{article_text}"
    )

    def validate(data: dict[str, Any]) -> str | None:
        if not data.get("main_findings"):
            return "main_findings must not be empty"
        return None

    return llm.chat_json(system, user, RESEARCH_SUMMARY_SCHEMA, validate=validate)


def plan_presentation(
    llm: LLM,
    summary: dict[str, Any],
    figures: list[Figure],
    slide_count: int,
    language: str,
    paper_title: str,
) -> dict[str, Any]:
    """Planner agent: narrative plan with exactly `slide_count` slides."""
    language_instruction = get_language_instruction(language)
    figure_ids = [figure.figure_id for figure in figures]
    valid_ids = set(figure_ids)
    outline = suggested_outline(slide_count)

    planning_input = json.dumps(
        {
            "paper_title": paper_title,
            "research_summary": summary,
            "available_figures": figure_catalog(figures),
            "requested_slide_count": slide_count,
            "suggested_outline": outline,
        },
        ensure_ascii=False,
        indent=2,
    )

    system = (
        "You are an expert at turning research papers into conference talks. "
        "Create a presentation plan from the research summary.\n"
        f"- Produce exactly {slide_count} slides that tell one coherent story: "
        "start with the problem and why it matters, end with the conclusion "
        "and takeaways. Adapt the suggested outline to this paper.\n"
        "- Each slide has a short title (max 8 words), one key_message "
        "(one sentence, the single point of the slide), 1-4 bullets "
        "(max 15 words each, concrete, no filler), and speaker_notes "
        "(2-4 sentences the presenter can say aloud).\n"
        "- Use only facts from the summary. Never invent numbers.\n"
        "- figure_id: pick one of the available figure ids only when that "
        "figure's caption directly supports the slide's key_message; "
        "otherwise null. Never use the same figure on two slides. The first "
        "slide usually has no figure unless an overview figure exists.\n"
        "- visual_need: one short ENGLISH sentence describing what kind of "
        "figure from the paper would best support this slide (e.g. "
        "'architecture overview diagram', 'bar chart comparing accuracy to "
        "baselines'). Use an empty string if the slide needs no figure.\n"
        f"- Write title, key_message, bullets, speaker_notes and "
        f"presentation_title in {language_instruction}."
    )

    def validate(data: dict[str, Any]) -> str | None:
        slides = data.get("slides") or []
        if len(slides) != slide_count:
            return f"expected exactly {slide_count} slides but got {len(slides)}"
        for index, slide in enumerate(slides, start=1):
            if not slide.get("title") or not slide.get("bullets"):
                return f"slide {index} is missing a title or bullets"
        return None

    plan = llm.chat_json(
        system,
        planning_input,
        presentation_plan_schema(slide_count, figure_ids),
        validate=validate,
        num_predict=6_144,
    )

    for slide in plan["slides"]:
        slide["figure_id"] = normalize_figure_id(slide.get("figure_id"), valid_ids)
        slide["visual_need"] = clean_visual_need(slide.get("visual_need"))

    return plan


def judge_figure_for_slide(
    llm: LLM,
    slide: dict[str, Any],
    candidates: list[Figure],
    language: str,
) -> dict[str, Any]:
    """Visual-grounding agent: choose the best figure for one slide, or none."""
    language_instruction = get_language_instruction(language)
    candidate_ids = [figure.figure_id for figure in candidates]
    valid_ids = set(candidate_ids)

    review_input = json.dumps(
        {
            "slide": {
                "title": slide.get("title"),
                "key_message": slide.get("key_message"),
                "bullets": slide.get("bullets"),
                "visual_need": slide.get("visual_need"),
            },
            "candidate_figures": [
                {
                    "figure_id": figure.figure_id,
                    "page": figure.page_number,
                    "caption": figure.caption,
                    "mentioned_in_text_as": figure.context,
                }
                for figure in candidates
            ],
        },
        ensure_ascii=False,
        indent=2,
    )

    system = (
        "You match figures from a research paper to one presentation slide. "
        "Judge each candidate by its caption and by how the paper's text "
        "refers to it. Choose the single figure that best supports the "
        "slide's key_message and bullets, or null if none genuinely does.\n"
        "relevance scale: 5 = the figure shows exactly what the slide claims; "
        "4 = strongly related; 3 = somewhat related and still helpful; "
        "2 = weakly related; 1 = unrelated. Use null when relevance would be "
        "below 3.\n"
        "explanation: one short sentence (at most 25 words) shown under the "
        "figure on the slide, stating what the figure shows and how it "
        "supports the slide text. Never invent details that are not in the "
        "caption or context.\n"
        f"Write the explanation in {language_instruction}."
    )

    def validate(data: dict[str, Any]) -> str | None:
        chosen = normalize_figure_id(data.get("figure_id"), valid_ids)
        if data.get("figure_id") not in (None, "null", "") and chosen is None:
            return f"figure_id must be one of {candidate_ids} or null"
        return None

    result = llm.chat_json(
        system,
        review_input,
        figure_judgement_schema(candidate_ids),
        validate=validate,
        num_predict=512,
    )
    result["figure_id"] = normalize_figure_id(result.get("figure_id"), valid_ids)
    result["relevance"] = int(result.get("relevance") or 0)
    result["explanation"] = (result.get("explanation") or "").strip()
    return result


__all__ = [
    "LLM",
    "LLMError",
    "Translator",
    "clean_visual_need",
    "deepcopy",
    "get_language_instruction",
    "judge_figure_for_slide",
    "normalize_figure_id",
    "plan_presentation",
    "suggested_outline",
    "summarize_article",
    "translate_plan",
]


TRANSLATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"translation": {"type": "string"}},
    "required": ["translation"],
    "additionalProperties": False,
}

TRANSLATION_EXAMPLES: dict[str, tuple[str, str]] = {
    "fa": ("Accuracy improves by 4.1% over the baseline", "دقت نسبت به مدل پایه ۴.۱٪ بهبود می‌یابد"),
    "ar": ("Accuracy improves by 4.1% over the baseline", "تتحسن الدقة بنسبة 4.1% مقارنة بالنموذج الأساسي"),
    "de": ("Accuracy improves by 4.1% over the baseline", "Die Genauigkeit verbessert sich um 4,1 % gegenüber der Baseline"),
    "fr": ("Accuracy improves by 4.1% over the baseline", "La précision s'améliore de 4,1 % par rapport à la référence"),
    "es": ("Accuracy improves by 4.1% over the baseline", "La precisión mejora un 4,1 % respecto a la línea base"),
    "tr": ("Accuracy improves by 4.1% over the baseline", "Doğruluk, temel modele göre %4,1 artar"),
}

LATIN_WORD = re.compile(r"[A-Za-z]{3,}")
# Cyrillic / CJK characters leaking into a Persian or Arabic answer.
FOREIGN_SCRIPT = re.compile(r"[\u0400-\u04FF\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]")
TRANSLATABLE_FIELDS = ("title", "key_message", "speaker_notes", "image_explanation")


def _wrong_script(language: str, text: str) -> bool:
    """True when the answer clearly stayed in English for a non-Latin target."""
    check = SCRIPT_CHECKS.get(language)
    if check is None or not text.strip():
        return False
    if language in {"fa", "ar", "ur", "he"} and FOREIGN_SCRIPT.search(text):
        return True
    return not check.search(text) and len(LATIN_WORD.findall(text)) >= 2


class Translator:
    """Translate short strings with a local model, caching repeated ones."""

    def __init__(self, llm: LLM, language: str) -> None:
        self.llm = llm
        self.language = language
        self.instruction = get_language_instruction(language)
        self.cache: dict[str, str] = {}
        self.failures: list[str] = []

    def __call__(self, text: str | None) -> str | None:
        if text is None:
            return None
        source = str(text).strip()
        if not source:
            return ""
        if source in self.cache:
            return self.cache[source]

        system = (
            f"You translate English academic slide text into {self.instruction}. "
            'Answer with JSON {"translation": "<translated text>"} only. Translate '
            "faithfully and naturally; keep numbers, percentages, units, model "
            "names, acronyms and other proper nouns unchanged; add nothing."
        )
        example = TRANSLATION_EXAMPLES.get(self.language)
        messages = [{"role": "system", "content": system}]
        if example:
            messages += [
                {"role": "user", "content": f"English: {example[0]}"},
                {
                    "role": "assistant",
                    "content": json.dumps({"translation": example[1]}, ensure_ascii=False),
                },
            ]
        user = f"English: {source}"

        def validate(data: dict[str, Any]) -> str | None:
            translation = (data.get("translation") or "").strip()
            if not translation:
                return "translation must not be empty"
            if _wrong_script(self.language, translation):
                return f"the translation must be written in {self.instruction}"
            return None

        try:
            result = self.llm.chat_json(
                system,
                user,
                TRANSLATION_SCHEMA,
                validate=validate,
                num_predict=768,
                history=messages[1:],
            )
            translation = result["translation"].strip()
        except LLMError as err:
            logger.warning("Translation failed, keeping English: %s (%s)", source[:60], err)
            self.failures.append(source)
            translation = source

        self.cache[source] = translation
        return translation


def translate_plan(
    llm: LLM,
    plan: dict[str, Any],
    language: str,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Return a copy of a (grounded) English plan with slide text translated."""
    translate = Translator(llm, language)
    translated_plan = deepcopy(plan)
    translated_plan["source_language"] = "en"
    translated_plan["presentation_title"] = translate(plan.get("presentation_title"))

    for number, slide in enumerate(translated_plan["slides"], start=1):
        slide["original_text"] = {
            "title": slide.get("title"),
            "key_message": slide.get("key_message"),
            "bullets": list(slide.get("bullets") or []),
            "speaker_notes": slide.get("speaker_notes"),
            "image_explanation": slide.get("image_explanation"),
        }
        for field in TRANSLATABLE_FIELDS:
            slide[field] = translate(slide.get(field))
        slide["bullets"] = [translate(bullet) for bullet in slide.get("bullets") or []]
        if progress:
            progress(f"  slide {number}: {slide['title']}")

    if translate.failures:
        translated_plan["untranslated_count"] = len(translate.failures)
        if progress:
            progress(
                f"  warning: {len(translate.failures)} text(s) stayed in English; "
                "a larger or more multilingual model (e.g. qwen3:8b) translates better"
            )
    return translated_plan
