#!/usr/bin/env python3
"""Yan Kuma — Mobile-friendly annotation with auto-cleanup to GitHub/HF."""

from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
import subprocess

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import yt_dlp

app = FastAPI()

# Config
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads")))
DB_PATH = str(BASE_DIR / "yan_kuma.db")
UPLOAD_DIR.mkdir(exist_ok=True)

# Storage limits (in MB)
MAX_STORAGE_MB = int(os.getenv("MAX_STORAGE_MB", "500"))

# GitHub config
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO = os.getenv("GITHUB_REPO", "ismobaga/yan_kuma")
GITHUB_USER = os.getenv("GITHUB_USER", "ismobaga")
GITHUB_EMAIL = os.getenv("GITHUB_EMAIL", "ismail@crommixmali.com")
GIT_DATA_BRANCH = "data"

# Hugging Face config
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_REPO_ID = os.getenv("HF_REPO_ID", "ismobaga/yan_kuma_bambara_asr")


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS batches (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE,
            created_at TEXT,
            exported_at TEXT,
            files_deleted INTEGER DEFAULT 0
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS segments (
            id INTEGER PRIMARY KEY,
            batch_id INTEGER,
            segment_id INTEGER,
            audio_path TEXT,
            frame_path TEXT,
            start_time REAL,
            end_time REAL,
            duration REAL,
            transcript_bm TEXT,
            translation_fr TEXT,
            show TEXT,
            episode TEXT,
            language TEXT DEFAULT 'Bambara',
            content_type TEXT DEFAULT 'Dialog',
            quality TEXT DEFAULT 'Clear',
            has_names INTEGER DEFAULT 0,
            annotated_at TEXT,
            FOREIGN KEY(batch_id) REFERENCES batches(id)
        )
    """
    )
    conn.commit()
    conn.close()


init_db()


def get_storage_usage_mb() -> float:
    total = sum(f.stat().st_size for f in UPLOAD_DIR.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def cleanup_batch_files(batch_id: int) -> None:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT audio_path, frame_path FROM segments WHERE batch_id = ?", (batch_id,)
    )
    for audio_path, frame_path in rows:
        for item in (audio_path, frame_path):
            try:
                if item and Path(item).exists():
                    Path(item).unlink()
            except OSError:
                pass
    conn.execute("UPDATE batches SET files_deleted = 1 WHERE id = ?", (batch_id,))
    conn.commit()
    conn.close()


def segment_by_silence(audio_path: str, min_dur: int = 20, sr: int = 16000) -> list[dict]:
    y, sr = librosa.load(audio_path, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)

    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    db = librosa.power_to_db(spectrogram, ref=np.max)
    silence_frames = np.mean(db, axis=0) < -40
    frame_times = librosa.frames_to_time(np.arange(len(silence_frames)), sr=sr)
    silence_times = frame_times[silence_frames][::10]

    segments = []
    seg_start = 0.0
    seg_idx = 0

    for silence_time in silence_times:
        seg_dur = silence_time - seg_start
        if seg_dur >= min_dur:
            seg_end = silence_time
            start_sample = int(seg_start * sr)
            end_sample = int(seg_end * sr)
            chunk = y[start_sample:end_sample]

            out_path = UPLOAD_DIR / f"seg_{seg_idx:05d}.wav"
            sf.write(out_path, chunk, sr)
            segments.append(
                {
                    "segment_id": seg_idx,
                    "audio_path": str(out_path),
                    "start_time": seg_start,
                    "end_time": seg_end,
                    "duration": seg_end - seg_start,
                }
            )
            seg_start = seg_end
            seg_idx += 1

    if duration - seg_start >= min_dur:
        chunk = y[int(seg_start * sr) :]
        out_path = UPLOAD_DIR / f"seg_{seg_idx:05d}.wav"
        sf.write(out_path, chunk, sr)
        segments.append(
            {
                "segment_id": seg_idx,
                "audio_path": str(out_path),
                "start_time": seg_start,
                "end_time": duration,
                "duration": duration - seg_start,
            }
        )

    return segments


def extract_frames(video_path: str, segments: list[dict]) -> list[dict]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 1

    for seg in segments:
        frame_num = int(seg["start_time"] * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if ret:
            frame_path = UPLOAD_DIR / f"frame_{seg['segment_id']:05d}.jpg"
            frame = cv2.resize(frame, (320, 180))
            cv2.imwrite(str(frame_path), frame)
            seg["frame_path"] = str(frame_path)

    cap.release()
    return segments


def push_to_github(parquet_path: Path, batch_name: str) -> bool:
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return False

    try:
        repo_dir = BASE_DIR / "yan_kuma_data"
        if not repo_dir.exists():
            git_url = f"https://{GITHUB_USER}:{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git"
            subprocess.run(
                f"git clone --branch {GIT_DATA_BRANCH} {git_url} {repo_dir} "
                f"2>/dev/null || git clone {git_url} {repo_dir} && cd {repo_dir} && "
                f"git checkout -b {GIT_DATA_BRANCH} 2>/dev/null || true",
                shell=True,
                check=False,
                capture_output=True,
            )

        (repo_dir / f"{batch_name}.parquet").write_bytes(parquet_path.read_bytes())

        subprocess.run(
            f"cd {repo_dir} && git config user.email {GITHUB_EMAIL} && git config user.name {GITHUB_USER}",
            shell=True,
            check=False,
        )
        subprocess.run(
            f"cd {repo_dir} && git add . && git commit -m 'Add batch: {batch_name}' 2>/dev/null",
            shell=True,
            check=False,
        )
        subprocess.run(
            f"cd {repo_dir} && git push -u origin {GIT_DATA_BRANCH} 2>/dev/null",
            shell=True,
            check=False,
            capture_output=True,
        )

        return True
    except Exception:
        return False


def push_to_huggingface(parquet_path: Path, batch_name: str) -> bool:
    if not HF_TOKEN:
        return False

    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(parquet_path),
            path_in_repo=f"data/{batch_name}.parquet",
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN,
            commit_message=f"Add batch: {batch_name}",
        )
        return True
    except Exception:
        return False


def _safe_upload_name(filename: str) -> str:
    name = Path(filename or "").name
    if not name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return name


@app.post("/upload-local")
async def upload_local(file: UploadFile = File(...)):
    video_path = UPLOAD_DIR / _safe_upload_name(file.filename or "")
    with open(video_path, "wb") as f:
        f.write(await file.read())

    audio_path = video_path.with_suffix(".wav")
    try:
        y, sr = librosa.load(str(video_path), sr=16000)
        sf.write(audio_path, y, sr)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio extraction failed: {e}") from e

    segments = extract_frames(str(video_path), segment_by_silence(str(audio_path)))

    batch_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "INSERT INTO batches (name, created_at) VALUES (?, ?)",
        (batch_name, datetime.now().isoformat()),
    )
    batch_id = cursor.lastrowid
    conn.commit()

    for seg in segments:
        conn.execute(
            """
            INSERT INTO segments
            (batch_id, segment_id, audio_path, frame_path, start_time, end_time, duration)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                batch_id,
                seg["segment_id"],
                seg["audio_path"],
                seg.get("frame_path", ""),
                seg["start_time"],
                seg["end_time"],
                seg["duration"],
            ),
        )
    conn.commit()
    conn.close()

    storage_mb = get_storage_usage_mb()
    return {
        "batch_id": batch_id,
        "segment_count": len(segments),
        "batch_name": batch_name,
        "storage_mb": round(storage_mb, 1),
        "storage_warning": storage_mb > MAX_STORAGE_MB,
    }


@app.post("/upload-youtube")
async def upload_youtube(url: str):
    ydl_opts = {
        "format": "best[height<=480]",
        "outtmpl": str(UPLOAD_DIR / "youtube_%(id)s"),
        "quiet": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"YouTube download failed: {e}") from e

    audio_path = Path(video_path).with_suffix(".wav")
    try:
        y, sr = librosa.load(video_path, sr=16000)
        sf.write(str(audio_path), y, sr)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio extraction failed: {e}") from e

    segments = extract_frames(video_path, segment_by_silence(str(audio_path)))

    batch_name = f"youtube_{info.get('id', 'unknown')}_{datetime.now().strftime('%H%M%S')}"
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "INSERT INTO batches (name, created_at) VALUES (?, ?)",
        (batch_name, datetime.now().isoformat()),
    )
    batch_id = cursor.lastrowid
    conn.commit()

    for seg in segments:
        conn.execute(
            """
            INSERT INTO segments
            (batch_id, segment_id, audio_path, frame_path, start_time, end_time, duration)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                batch_id,
                seg["segment_id"],
                seg["audio_path"],
                seg.get("frame_path", ""),
                seg["start_time"],
                seg["end_time"],
                seg["duration"],
            ),
        )
    conn.commit()
    conn.close()

    storage_mb = get_storage_usage_mb()
    return {
        "batch_id": batch_id,
        "segment_count": len(segments),
        "title": info.get("title", "Unknown"),
        "batch_name": batch_name,
        "storage_mb": round(storage_mb, 1),
        "storage_warning": storage_mb > MAX_STORAGE_MB,
    }


@app.get("/segments")
def get_segments(batch_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM segments WHERE batch_id = ? ORDER BY segment_id", (batch_id,)
    )
    segments = [dict(row) for row in rows.fetchall()]
    conn.close()
    return segments


@app.get("/storage")
def get_storage_info():
    used_mb = get_storage_usage_mb()
    return {
        "used_mb": round(used_mb, 1),
        "max_mb": MAX_STORAGE_MB,
        "percent": round(100 * used_mb / MAX_STORAGE_MB, 1),
        "warning": used_mb > MAX_STORAGE_MB,
    }


@app.post("/save")
def save_annotation(
    segment_id: int,
    transcript_bm: str,
    show: str = "",
    episode: str = "",
    language: str = "Bambara",
    content_type: str = "Dialog",
    quality: str = "Clear",
    translation_fr: str = "",
):
    has_names = 1 if "[" in transcript_bm else 0
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        UPDATE segments
        SET transcript_bm = ?, translation_fr = ?, show = ?, episode = ?, language = ?,
            content_type = ?, quality = ?, has_names = ?, annotated_at = ?
        WHERE id = ?
    """,
        (
            transcript_bm,
            translation_fr,
            show,
            episode,
            language,
            content_type,
            quality,
            has_names,
            datetime.now().isoformat(),
            segment_id,
        ),
    )
    conn.commit()
    conn.close()
    return {"ok": True}


@app.post("/export")
def export_parquet(batch_id: int, cleanup: bool = True):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM segments WHERE batch_id = ?", conn, params=(batch_id,))

    batch = conn.execute("SELECT name FROM batches WHERE id = ?", (batch_id,)).fetchone()
    batch_name = batch[0] if batch else f"batch_{batch_id}"
    conn.close()

    if df.empty:
        raise HTTPException(status_code=400, detail="No segments to export")

    export_path = UPLOAD_DIR / f"{batch_name}_export.parquet"
    df.to_parquet(export_path, index=False)

    metadata = {
        "batch_id": batch_id,
        "batch_name": batch_name,
        "exported_at": datetime.now().isoformat(),
        "segment_count": len(df),
        "annotated_count": (df["transcript_bm"].notna() & (df["transcript_bm"] != "")).sum(),
        "total_duration": float(df["duration"].sum()),
    }

    metadata_path = UPLOAD_DIR / f"{batch_name}_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    github_ok = push_to_github(export_path, batch_name)
    hf_ok = push_to_huggingface(export_path, batch_name)

    if cleanup:
        cleanup_batch_files(batch_id)

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE batches SET exported_at = ? WHERE id = ?",
        (datetime.now().isoformat(), batch_id),
    )
    conn.commit()
    conn.close()

    storage_mb = get_storage_usage_mb()
    return {
        "file": str(export_path),
        "metadata": metadata,
        "github_synced": github_ok,
        "huggingface_synced": hf_ok,
        "files_cleaned": cleanup,
        "storage_mb_after": round(storage_mb, 1),
    }


@app.get("/download/{batch_id}")
def download_export(batch_id: int):
    conn = sqlite3.connect(DB_PATH)
    batch = conn.execute("SELECT name FROM batches WHERE id = ?", (batch_id,)).fetchone()
    conn.close()

    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    export_path = UPLOAD_DIR / f"{batch[0]}_export.parquet"
    if not export_path.exists():
        raise HTTPException(status_code=404, detail="Export not found")

    return FileResponse(export_path, filename=f"yan_kuma_{batch[0]}.parquet")


@app.get("/", response_class=HTMLResponse)
def index():
    with open(BASE_DIR / "index.html", "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
