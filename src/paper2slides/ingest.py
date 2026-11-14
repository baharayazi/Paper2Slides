"""
PDF ingestion: text, title and figure extraction.

Figures are extracted at the *figure level* rather than the embedded-image
level. For every caption that looks like ``Figure N ...`` we collect the
raster images and vector drawings sitting directly above it, take their
bounding box and render that region to a single PNG. This keeps composite
figures together and also captures vector plots that have no embedded bitmap.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pymupdf

from .models import Article, Figure

logger = logging.getLogger(__name__)

CAPTION_PATTERN = re.compile(
    r"^(?:Figure|Fig\.?)\s*(\d+)[a-zA-Z]?\s*(?P<sep>[|:.\-–—]|\n|\s)",
    re.IGNORECASE,
)
REFERENCES_PATTERN = re.compile(
    r"^\s*(?:\d+\.?\s*)?(?:References|Bibliography|REFERENCES)\s*$",
    re.MULTILINE,
)
APPENDIX_PATTERN = re.compile(
    r"^\s*(?:Appendix\b|Appendices\b|Supplementary Material\b|A\s+[A-Z][a-z]+)",
    re.MULTILINE,
)
PAGE_MARKER_PATTERN = re.compile(r"\n--- Page \d+ ---\n")

MIN_GRAPHIC_SIZE = 20  # points; anything smaller is an icon or a rule
MIN_DRAWING_SIZE = 40
MAX_CAPTION_GAP = 120  # points between the figure bottom and its caption
MAX_CHAIN_GAP = 90  # points between stacked parts of the same figure
MAX_RAW_IMAGE_CAPTION_GAP = 60
MIN_RAW_IMAGE_PIXELS = 200
MAX_CONTEXT_SENTENCES = 3
MAX_CONTEXT_CHARACTERS = 320


def clean_text(text: str) -> str:
    """Join hyphenated line breaks and collapse stray whitespace."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_pdf_text(doc: pymupdf.Document) -> str:
    """Read the text of every page, keeping page markers for source tracking."""
    pages = []
    for page_num, page in enumerate(doc, start=1):
        page_text = clean_text(page.get_text("text"))
        pages.append(f"\n--- Page {page_num} ---\n{page_text}")
    return "\n".join(pages)


def strip_references(text: str) -> str:
    """Drop the bibliography, which is long and useless for slides."""
    matches = list(REFERENCES_PATTERN.finditer(text))
    if not matches:
        return text

    for match in reversed(matches):
        if match.start() <= len(text) * 0.15:
            continue
        before = text[: match.start()].rstrip()
        appendix = APPENDIX_PATTERN.search(text, match.end())
        if appendix:
            return before + "\n\n" + text[appendix.start():]
        return before

    return text


def is_plausible_title(candidate: str) -> bool:
    candidate = candidate.strip()
    if not 10 <= len(candidate) <= 250:
        return False
    lowered = candidate.lower()
    if lowered.endswith((".pdf", ".dvi", ".tex", ".doc", ".docx")):
        return False
    if lowered.startswith(("untitled", "arxiv:", "microsoft word", "doi:")):
        return False
    return not re.fullmatch(r"[\W\d_]+", candidate)


def find_article_title(doc: pymupdf.Document, pdf_path: Path) -> str:
    """Pick the title from metadata, else the largest text on page one."""
    metadata_title = (doc.metadata or {}).get("title") or ""
    if is_plausible_title(metadata_title):
        return metadata_title.strip()

    if doc.page_count:
        title = _largest_font_text(doc[0])
        if is_plausible_title(title):
            return title

    return pdf_path.stem.replace("_", " ").replace("-", " ")


def _largest_font_text(page: pymupdf.Page) -> str:
    """Join the spans that share the largest font size on the page."""
    spans: list[tuple[float, float, str]] = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span["text"].strip()
                if not text:
                    continue
                # Skip rotated watermark-like text (arXiv stamps).
                if abs(line.get("dir", (1, 0))[1]) > 0.5:
                    continue
                spans.append((round(span["size"], 1), span["bbox"][1], text))

    if not spans:
        return ""

    largest = max(size for size, _, _ in spans)
    title_parts = [
        text for size, _, text in sorted(spans, key=lambda s: s[1])
        if size >= largest - 0.5
    ]
    return re.sub(r"\s+", " ", " ".join(title_parts)).strip()


def parse_caption(block_text: str) -> int | None:
    """Return the figure number if the block looks like a figure caption."""
    text = block_text.strip()
    match = CAPTION_PATTERN.match(text)
    if not match:
        return None

    separator = match.group("sep")
    if separator.strip() == "" and separator != "\n" and len(text) > 250:
        return None

    return int(match.group(1))


def _caption_blocks(page: pymupdf.Page) -> list[tuple[pymupdf.Rect, int, str]]:
    captions = []
    for block in page.get_text("blocks"):
        if len(block) < 5 or block[6] != 0:  # 0 = text block, 1 = image block
            continue
        text = block[4].strip()
        number = parse_caption(text)
        if number is None:
            continue
        rect = pymupdf.Rect(block[:4])
        captions.append((rect, number, clean_text(text).replace("\n", " ")))

    captions.sort(key=lambda item: (item[0].y0, item[0].x0))
    return captions


def _graphic_rects(page: pymupdf.Page) -> tuple[list[pymupdf.Rect], list[pymupdf.Rect]]:
    """
    Rectangles of raster images and vector drawing clusters on the page.

    Returns ``(graphics, frames)``: *frames* are large drawing clusters that
    may enclose a whole figure including its caption (boxed figures).
    """
    page_rect = page.rect
    page_area = page_rect.get_area() or 1.0
    rects: list[pymupdf.Rect] = []

    for image in page.get_images(full=True):
        for rect in page.get_image_rects(image[0]):
            rects.append(pymupdf.Rect(rect))

    try:
        for rect in page.cluster_drawings():
            rect = pymupdf.Rect(rect)
            if rect.width >= MIN_DRAWING_SIZE and rect.height >= MIN_DRAWING_SIZE:
                rects.append(rect)
    except Exception as err:  # noqa: BLE001 - depends on the PyMuPDF build
        logger.debug("cluster_drawings failed on page %d: %s", page.number + 1, err)

    usable, frames = [], []
    for rect in rects:
        clipped = rect & page_rect
        if clipped.is_empty:
            continue
        if clipped.width < MIN_GRAPHIC_SIZE or clipped.height < MIN_GRAPHIC_SIZE:
            continue
        if clipped.get_area() > 0.6 * page_area:
            if clipped.get_area() < 0.9 * page_area:
                frames.append(clipped)  # boxed figure, not a page border
            continue
        usable.append(clipped)

    return usable, frames


def _horizontally_overlaps(a: pymupdf.Rect, b: pymupdf.Rect, slack: float = 20) -> bool:
    return a.x1 > b.x0 - slack and a.x0 < b.x1 + slack


def _frame_region(
        caption_rect: pymupdf.Rect,
        frames: list[pymupdf.Rect],
        graphics: list[pymupdf.Rect],
        text_blocks: list[pymupdf.Rect],
        upper_bound: float,
) -> pymupdf.Rect | None:
    """
    Region of a boxed figure whose frame encloses the caption too.

    The frame may be a background fill that also sits behind the paragraph
    above the figure, so the top edge is tightened to the topmost graphic
    plus any text rows (table headers, labels) directly above it.
    """
    for frame in frames:
        if not _horizontally_overlaps(frame, caption_rect, slack=5):
            continue
        if frame.y0 > caption_rect.y0 - 40 or frame.y1 < caption_rect.y0 - 8:
            continue

        bottom = caption_rect.y0 - 4
        top = max(frame.y0, upper_bound)
        inside = [
            rect for rect in graphics
            if rect.y0 >= top - 2 and rect.y1 <= bottom + 2
               and _horizontally_overlaps(rect, frame, slack=0)
        ]
        if inside:
            top = min(rect.y0 for rect in inside)
            changed = True
            while changed:
                changed = False
                for block in text_blocks:
                    if (
                            top - 30 <= block.y1 <= top
                            and block.y0 >= frame.y0 - 2
                            and _horizontally_overlaps(block, frame, slack=0)
                    ):
                        top = min(top, block.y0)
                        changed = True

        region = pymupdf.Rect(frame.x0, top, frame.x1, bottom)
        if region.height >= 40:
            return region
    return None


def _figure_region(
        caption_rect: pymupdf.Rect,
        graphics: list[pymupdf.Rect],
        frames: list[pymupdf.Rect],
        text_blocks: list[pymupdf.Rect],
        page_rect: pymupdf.Rect,
        upper_bound: float,
) -> pymupdf.Rect | None:
    """Bounding box of the graphics that belong to a caption."""
    candidates = [
        rect
        for rect in graphics
        if rect.y1 <= caption_rect.y0 + 8
           and rect.y0 >= upper_bound - 2
           and _horizontally_overlaps(rect, caption_rect, slack=5)
    ]
    candidates.sort(key=lambda rect: rect.y1, reverse=True)

    if not candidates or caption_rect.y0 - candidates[0].y1 > MAX_CAPTION_GAP:
        region = _frame_region(caption_rect, frames, graphics, text_blocks, upper_bound)
        if region is None:
            return None
    else:
        region = pymupdf.Rect(candidates[0])
        for rect in candidates[1:]:
            if region.y0 - rect.y1 > MAX_CHAIN_GAP:
                break
            region |= rect

    region = pymupdf.Rect(
        region.x0 - 6, region.y0 - 6, region.x1 + 6, region.y1 + 4
    ) & page_rect
    if region.is_empty or region.width < 40 or region.height < 30:
        return None
    return region


def _render_region(page: pymupdf.Page, region: pymupdf.Rect, output_path: Path) -> tuple[int, int]:
    zoom = min(3.0, max(1.5, 1600 / max(region.width, 1)))
    pixmap = page.get_pixmap(clip=region, matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    pixmap.save(output_path)
    return pixmap.width, pixmap.height


def _nearest_text_below(page: pymupdf.Page, rect: pymupdf.Rect) -> str:
    best: tuple[float, str] | None = None
    for block in page.get_text("blocks"):
        if len(block) < 5 or block[6] != 0:
            continue
        block_rect = pymupdf.Rect(block[:4])
        gap = block_rect.y0 - rect.y1
        if gap < -5 or gap > MAX_RAW_IMAGE_CAPTION_GAP:
            continue
        if not _horizontally_overlaps(block_rect, rect):
            continue
        if best is None or gap < best[0]:
            best = (gap, clean_text(block[4]).replace("\n", " "))
    return best[1][:300] if best else ""


def extract_figures(doc: pymupdf.Document, output_dir: Path) -> list[Figure]:
    """Extract captioned figure regions, plus large uncaptioned bitmaps."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.png"):
        stale.unlink()

    figures: list[Figure] = []
    seen_xrefs: set[int] = set()

    for page_num, page in enumerate(doc, start=1):
        graphics, frames = _graphic_rects(page)
        captions = _caption_blocks(page)
        text_blocks = [
            pymupdf.Rect(block[:4])
            for block in page.get_text("blocks")
            if len(block) >= 7 and block[6] == 0 and block[4].strip()
        ]
        regions: list[pymupdf.Rect] = []

        for caption_rect, number, caption_text in captions:
            # Only captions in the same column(s) bound the region above us.
            upper_bound = 0.0
            for other_rect, _, _ in captions:
                if other_rect is caption_rect or other_rect.y1 > caption_rect.y0:
                    continue
                if _horizontally_overlaps(other_rect, caption_rect, slack=0):
                    upper_bound = max(upper_bound, other_rect.y1)

            region = _figure_region(
                caption_rect, graphics, frames, text_blocks, page.rect, upper_bound
            )
            if region is None:
                continue

            index = len(figures) + 1
            path = output_dir / f"figure_{index:02d}_page_{page_num}.png"
            width, height = _render_region(page, region, path)
            regions.append(region)
            figures.append(
                Figure(
                    figure_id=f"F{index}",
                    page_number=page_num,
                    path=path,
                    width=width,
                    height=height,
                    caption=caption_text,
                    figure_number=number,
                    kind="figure",
                )
            )

        # Fallback: big bitmaps that no caption claimed (scanned figures,
        # photos, or papers whose captions we failed to detect).
        for image in page.get_images(full=True):
            xref = image[0]
            if xref in seen_xrefs:
                continue
            rects = [pymupdf.Rect(r) & page.rect for r in page.get_image_rects(xref)]
            rects = [r for r in rects if not r.is_empty]
            if not rects:
                continue
            rect = rects[0]
            if any(region.contains(rect) or (region & rect).get_area() > 0.5 * rect.get_area() for region in regions):
                seen_xrefs.add(xref)
                continue
            try:
                info = doc.extract_image(xref)
            except Exception as err:  # noqa: BLE001 - corrupt image streams
                logger.debug("extract_image failed for xref %d: %s", xref, err)
                continue
            if not info or info.get("width", 0) < MIN_RAW_IMAGE_PIXELS or info.get("height", 0) < MIN_RAW_IMAGE_PIXELS:
                continue
            if rect.width < 60 or rect.height < 60:
                continue
            seen_xrefs.add(xref)

            index = len(figures) + 1
            path = output_dir / f"figure_{index:02d}_page_{page_num}.png"
            width, height = _render_region(page, rect, path)
            regions.append(rect)
            figures.append(
                Figure(
                    figure_id=f"F{index}",
                    page_number=page_num,
                    path=path,
                    width=width,
                    height=height,
                    caption=_nearest_text_below(page, rect),
                    figure_number=None,
                    kind="image",
                )
            )

    return figures


def attach_context(figures: list[Figure], text: str) -> None:
    """Attach the body sentences that mention each numbered figure."""
    body = PAGE_MARKER_PATTERN.sub(" ", text)
    body = re.sub(r"\s+", " ", body)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", body)

    for figure in figures:
        if figure.figure_number is None:
            continue
        mention = re.compile(
            rf"\b(?:Figure|Fig\.?)\s*{figure.figure_number}(?!\d)", re.IGNORECASE
        )
        caption_prefix = re.sub(r"\s+", " ", figure.caption)[:40]
        context: list[str] = []
        for sentence in sentences:
            if not mention.search(sentence):
                continue
            sentence = sentence.strip()
            if caption_prefix and caption_prefix in sentence:
                continue  # that is the caption itself
            if CAPTION_PATTERN.match(sentence) and CAPTION_PATTERN.match(sentence).group("sep") in "|:":
                continue
            if len(sentence) > MAX_CONTEXT_CHARACTERS:
                sentence = sentence[: MAX_CONTEXT_CHARACTERS - 1].rsplit(" ", 1)[0] + "…"
            context.append(sentence)
            if len(context) >= MAX_CONTEXT_SENTENCES:
                break
        figure.context = context


def build_article(pdf_path: Path, figures_dir: Path) -> Article:
    """Build one structured object containing the extracted paper data."""
    with pymupdf.open(pdf_path) as doc:
        if doc.needs_pass:
            raise ValueError(f"'{pdf_path}' is password protected")

        text = read_pdf_text(doc)
        title = find_article_title(doc, pdf_path)
        figures = extract_figures(doc, figures_dir)
        page_count = doc.page_count

    attach_context(figures, text)

    return Article(
        source_path=pdf_path,
        title=title,
        page_count=page_count,
        text=text,
        figures=figures,
    )


def save_json(data: object, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
