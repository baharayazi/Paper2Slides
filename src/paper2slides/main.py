"""Command-line interface for Paper2Slides."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .llm import LLM, LLMError
from .orchestrator import MAX_SLIDES, MIN_SLIDES, run_pipeline
from .render import RenderError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paper2slides",
        description="Turn a research paper PDF into an editable PowerPoint presentation.",
    )
    parser.add_argument("pdf_path", type=Path, help="Path to the paper (PDF)")
    parser.add_argument(
        "--slides",
        type=int,
        choices=range(MIN_SLIDES, MAX_SLIDES + 1),
        default=5,
        metavar=f"[{MIN_SLIDES}-{MAX_SLIDES}]",
        help="Number of slides to generate (default: 5)",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code for the slide text, e.g. en or fa (default: en)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the .pptx (default: outputs/<paper>.pptx)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("data"),
        help="Directory for extracted figures and intermediate JSON (default: data)",
    )
    parser.add_argument("--model", default=None, help="Ollama model name (default: $OLLAMA_MODEL or qwen3:4b)")
    parser.add_argument("--num-ctx", type=int, default=None, help="Context window in tokens (default: $OLLAMA_NUM_CTX or 16384)")
    parser.add_argument("--no-render", action="store_true", help="Stop after the grounded plan; do not build the .pptx")
    parser.add_argument("--verbose", action="store_true", help="Show model retry warnings")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.ERROR,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.pdf_path.exists():
        parser.error(f"PDF file '{args.pdf_path}' does not exist")
    if args.pdf_path.suffix.lower() != ".pdf":
        parser.error(f"'{args.pdf_path}' is not a PDF file")

    llm = LLM(model=args.model, num_ctx=args.num_ctx)

    try:
        result = run_pipeline(
            args.pdf_path,
            slide_count=args.slides,
            language=args.language,
            output_path=args.output,
            work_dir=args.work_dir,
            llm=llm,
            render=not args.no_render,
            progress=lambda message: print(message, flush=True),
        )
    except (LLMError, RenderError, ValueError, FileNotFoundError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130

    if result.pptx:
        print(f"\nPresentation: {result.pptx}")
    print(f"Plan: {result.final_plan_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
