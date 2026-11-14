"""Small local HTTP API around the pipeline.

    uv run uvicorn paper2slides.api:app --reload

    POST /jobs?slide_count=6&language=en   (multipart file upload)
    GET  /jobs/{job_id}
    GET  /jobs/{job_id}/download
"""

from __future__ import annotations

import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from .orchestrator import MAX_SLIDES, MIN_SLIDES, result_to_dict, run_pipeline

app = FastAPI(title="Paper2Slides", version="0.2.0")

JOBS_ROOT = Path("data") / "jobs"
_jobs: dict[str, dict[str, Any]] = {}
_lock = Lock()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _update(job_id: str, **fields: Any) -> None:
    with _lock:
        _jobs[job_id].update(fields)
        _jobs[job_id]["updated_at"] = _now()


def _log(job_id: str, message: str) -> None:
    with _lock:
        _jobs[job_id]["log"].append(message)
        _jobs[job_id]["updated_at"] = _now()


def _run_job(job_id: str, pdf_path: Path, slide_count: int, language: str, job_dir: Path) -> None:
    _update(job_id, status="running")
    try:
        result = run_pipeline(
            pdf_path,
            slide_count=slide_count,
            language=language,
            output_path=job_dir / "presentation.pptx",
            work_dir=job_dir,
            progress=lambda message: _log(job_id, message),
        )
    except Exception as err:  # noqa: BLE001 - the job must record any failure
        _update(job_id, status="failed", error=str(err))
        return
    _update(job_id, status="done", result=result_to_dict(result))


@app.post("/jobs", status_code=202)
async def create_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),  # noqa: B008 - FastAPI dependency idiom
    slide_count: int = Query(5, ge=MIN_SLIDES, le=MAX_SLIDES),
    language: str = Query("en", min_length=2, max_length=5),
) -> dict[str, Any]:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = job_dir / "paper.pdf"
    with pdf_path.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    with _lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "slide_count": slide_count,
            "language": language,
            "filename": file.filename,
            "created_at": _now(),
            "updated_at": _now(),
            "log": [],
            "error": None,
            "result": None,
        }

    background_tasks.add_task(_run_job, job_id, pdf_path, slide_count, language, job_dir)
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        return dict(job)


@app.get("/jobs/{job_id}/download")
async def download(job_id: str) -> FileResponse:
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    if job["status"] != "done" or not job["result"] or not job["result"].get("pptx"):
        raise HTTPException(status_code=409, detail=f"Job is {job['status']}")
    return FileResponse(
        job["result"]["pptx"],
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename=f"{Path(job['filename']).stem}.pptx",
    )
