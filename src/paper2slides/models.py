"""Data structures shared across the Paper2Slides pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Figure:
    """
    One figure extracted from the paper.

    A figure is a *region* of a page (all sub-images, vector drawings and
    labels that belong to one caption), rendered to a single PNG file.
    """

    figure_id: str
    page_number: int
    path: Path
    width: int
    height: int
    caption: str
    figure_number: int | None = None
    context: list[str] = field(default_factory=list)
    kind: str = "figure"  # "figure" (captioned region) or "image" (raw bitmap)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = self.path.as_posix()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Figure:
        return cls(
            figure_id=data["figure_id"],
            page_number=data["page_number"],
            path=Path(data["path"]),
            width=data["width"],
            height=data["height"],
            caption=data.get("caption", ""),
            figure_number=data.get("figure_number"),
            context=list(data.get("context", [])),
            kind=data.get("kind", "figure"),
        )


@dataclass
class Article:
    source_path: Path
    title: str
    page_count: int
    text: str
    figures: list[Figure]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path.as_posix(),
            "title": self.title,
            "page_count": self.page_count,
            "text": self.text,
            "figures": [figure.to_dict() for figure in self.figures],
        }
