# Paper2Slides

Paper2Slides turns a research-paper PDF into a concise, editable PowerPoint
presentation. You choose how many slides you want (4 to 10) and the language,
and it builds a narrative from the paper, places the paper's own figures next
to the slide text they support, and writes speaker notes.

Everything runs locally: PDF parsing with PyMuPDF, a local LLM through Ollama,
and PowerPoint rendering with pptxgenjs.

## How it works

```text
PDF
 │
 ├─ 1. Ingest        text (references stripped), title, figure regions rendered to PNG,
 │                   captions and the body sentences that mention each figure
 ├─ 2. Reader agent  structured summary: question, method, findings, key numbers, limits
 ├─ 3. Planner agent exactly N slides with title, key message, bullets, speaker notes,
 │                   an English "visual need" and an optional figure pick (schema-constrained)
 ├─ 4. Grounding     per slide: TF-IDF shortlist of figures (caption + in-text mentions)
 │                   → LLM judge picks one figure or none, scores relevance 1-5,
 │                   writes the one-line explanation; each figure is used at most once
 └─ 5. Render        title slide + content slides, RTL layout for Persian/Arabic,
                     figures kept in aspect ratio, caption + page in speaker notes
```

Figures are extracted at the *figure level*: for every `Figure N` caption the
raster images and vector drawings above it are grouped and rendered as a single
image. Composite figures stay in one piece and vector plots (matplotlib, TikZ)
are captured even though they contain no embedded bitmap.

The figure picker is constrained to real figure IDs by the JSON schema sent to
the model, so the model cannot invent image paths.

## Requirements

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) with a local model (default `qwen3:4b`)
- Node.js 18+ (for the PowerPoint renderer)

## Installation

```bash
git clone git@github.com:baharaz78/Paper2Slides.git
cd Paper2Slides

uv sync
ollama pull qwen3:4b
cp .env.example .env

cd renderer && npm install && cd ..
```

## Usage

```bash
uv run paper2slides paper.pdf --slides 6 --language en --output outputs/paper.pptx
```

Persian presentation:

```bash
uv run paper2slides paper.pdf --slides 6 --language fa --output outputs/paper-fa.pptx
```

Options:

| Flag | Meaning |
| --- | --- |
| `--slides N` | number of content slides, 4 to 10 (default 5) |
| `--language CODE` | `en`, `fa`, ... (RTL layout for `fa`, `ar`, `he`, `ur`) |
| `--output PATH` | where to write the `.pptx` (default `outputs/<paper>.pptx`) |
| `--work-dir DIR` | where figures and intermediate JSON go (default `data`) |
| `--model NAME` | Ollama model (default `$OLLAMA_MODEL` or `qwen3:4b`) |
| `--num-ctx N` | context window in tokens (default `$OLLAMA_NUM_CTX` or 16384) |
| `--no-render` | stop after the grounded plan JSON |
| `--verbose` | show model retry warnings |

Intermediate files in the work dir let you inspect or hand-edit each stage:
`article.json`, `figures/*.png`, `article_summary.json`,
`presentation_plan.json`, `final_presentation_plan.json`.

To re-render a hand-edited plan without calling the model:

```bash
node renderer/render_presentation.mjs data/final_presentation_plan.json outputs/edited.pptx
```

## Local API

```bash
uv run uvicorn paper2slides.api:app --reload
```

```text
POST /jobs?slide_count=6&language=en      multipart upload, field "file"
GET  /jobs/{job_id}                       status, progress log, result
GET  /jobs/{job_id}/download              the .pptx when status is "done"
```

## Project structure

```text
src/paper2slides/
├── models.py        Article and Figure dataclasses
├── ingest.py        PDF text, title and figure-region extraction
├── llm.py           Ollama client + reader / planner / judge agents (JSON-schema outputs)
├── grounding.py     figure-to-slide matching (TF-IDF shortlist + LLM judge)
├── render.py        render plan + Node renderer invocation
├── orchestrator.py  the end-to-end pipeline
├── main.py          CLI
└── api.py           FastAPI job API
renderer/render_presentation.mjs   pptxgenjs renderer
tests/                             pytest suite (no model needed)
```

## Development

```bash
uv run python -m pytest -q
uv run ruff check src tests
```

## Languages

All reasoning (summary, planning, figure matching) runs in English, where small
local models are strongest. For any other `--language` the slide text is
translated afterwards, one string at a time, with a script check that rejects
answers that stayed in English. The English original of every slide is kept in
`final_presentation_plan.json` under `original_text`.

`qwen3:4b` produces acceptable Persian this way, but a larger or more
multilingual model gives noticeably better translations, for example:

```bash
ollama pull qwen3:8b
uv run paper2slides paper.pdf --slides 6 --language fa --model qwen3:8b
```

## Notes

- Only information found in the paper is used; the agents are told never to
  invent numbers and the JSON schemas keep the output structured.
- A figure is placed on a slide only when the judge rates its relevance at
  least 3/5; otherwise the slide is text-only.
- Larger models (`qwen3:8b`, `qwen3:14b`) give noticeably better summaries and
  figure choices if your machine can run them: `--model qwen3:8b`.
