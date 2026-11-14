"""Render the final presentation plan to PowerPoint via the Node renderer."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import Article

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RENDERER_DIR = PROJECT_ROOT / "renderer"
RENDERER_SCRIPT = RENDERER_DIR / "render_presentation.mjs"
RTL_LANGUAGES = {"fa", "ar", "he", "ur", "ps"}
LANGUAGE_TAGS = {"fa": "fa-IR", "ar": "ar-SA", "en": "en-US", "de": "de-DE", "fr": "fr-FR", "es": "es-ES"}


class RenderError(RuntimeError):
    """Raised when the PowerPoint file cannot be produced."""


def build_render_plan(plan: dict[str, Any], article: Article, language: str) -> dict[str, Any]:
    """Add layout metadata and absolute image paths for the renderer."""
    render_plan = json.loads(json.dumps(plan, ensure_ascii=False))
    render_plan["language"] = language
    render_plan["language_tag"] = LANGUAGE_TAGS.get(language, language)
    render_plan["direction"] = "rtl" if language in RTL_LANGUAGES else "ltr"
    render_plan["source_title"] = article.title
    render_plan["source_file"] = article.source_path.name
    render_plan["page_count"] = article.page_count
    render_plan["generated_at"] = datetime.now(UTC).isoformat(timespec="seconds")

    for slide in render_plan["slides"]:
        image_path = slide.get("image_path")
        if image_path:
            slide["image_path"] = str(Path(image_path).resolve())
    return render_plan


def check_renderer(renderer_dir: Path = RENDERER_DIR, node: str = "node") -> str:
    """Return the node executable, raising a helpful error if the renderer is unusable."""
    node_path = shutil.which(node)
    if node_path is None:
        raise RenderError("Node.js was not found on PATH. Install Node.js 18+ to render PowerPoint files.")
    if not (renderer_dir / "render_presentation.mjs").exists():
        raise RenderError(f"Renderer script not found in {renderer_dir}")
    if not (renderer_dir / "node_modules" / "pptxgenjs").exists():
        raise RenderError(
            "Renderer dependencies are missing. Run: cd renderer && npm install"
        )
    return node_path


def render_pptx(
    render_plan_path: Path,
    output_path: Path,
    renderer_dir: Path = RENDERER_DIR,
    node: str = "node",
) -> Path:
    node_path = check_renderer(renderer_dir, node)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed = subprocess.run(
        [node_path, str(renderer_dir / "render_presentation.mjs"), str(render_plan_path.resolve()), str(output_path.resolve())],
        cwd=renderer_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RenderError(
            "The PowerPoint renderer failed:\n" + (completed.stderr or completed.stdout).strip()
        )
    if not output_path.exists():
        raise RenderError("The renderer finished but no PowerPoint file was written")
    return output_path
