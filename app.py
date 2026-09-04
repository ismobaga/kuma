#!/usr/bin/env python3
"""Yan Kuma — Mobile-friendly annotation with auto-cleanup to GitHub/HF."""

from datetime import datetime
import json
import os
from pathlib import Path
import re
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
import time

app = FastAPI()

# Config
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads")))
DB_PATH = str(BASE_DIR / "yan_kuma.db")
UPLOAD_DIR.mkdir(exist_ok=True)
ASR_MODEL = os.getenv("ASR_MODEL", "facebook/wav2vec2-large-xlsr-53-bambara")
ASR_LOAD_ON_STARTUP = os.getenv("ASR_LOAD_ON_STARTUP", "false").lower() in {"1", "true", "yes", "on"}

# YouTube config
YOUTUBE_COOKIES_FILE = os.getenv("YOUTUBE_COOKIES_FILE", "")
YOUTUBE_COOKIES_FROM_BROWSER = os.getenv("YOUTUBE_COOKIES_FROM_BROWSER", "")
YOUTUBE_USER_AGENT = os.getenv("YOUTUBE_USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

cookies_content = os.getenv("YOUTUBE_COOKIES_CONTENT")
if cookies_content:
    with open("cookies.txt", "w") as f:
        f.write(cookies_content)
    os.environ["YOUTUBE_COOKIES_FILE"] = "./cookies.txt"
# ASR setup (optional, graceful fallback)
asr_model = None
asr_processor = None
asr_available = False


def load_asr_model() -> bool:
    global asr_model, asr_processor, asr_available

    if asr_available and asr_model is not None and asr_processor is not None:
        return True

    if not ASR_MODEL:
        print("⚠ ASR model not configured. ASR disabled.")
        return False

    try:
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

        asr_processor = Wav2Vec2Processor.from_pretrained(ASR_MODEL)
        asr_model = Wav2Vec2ForCTC.from_pretrained(ASR_MODEL)
        asr_available = True
        print(f"✓ ASR model loaded: {ASR_MODEL}")
        return True
    except ImportError:
        print("⚠ Transformers not installed. ASR disabled.")
        print("  Install with: pip install transformers torch")
        asr_available = False
        return False
    except Exception as e:
        print(f"⚠ ASR not available: {e}")
        asr_available = False
        return False


if ASR_LOAD_ON_STARTUP:
    load_asr_model()

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
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail=f"Unable to open video file: {video_path}")

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
        safe_batch_name = (
            batch_name
            if re.fullmatch(r"batch_\d+", batch_name or "")
            else f"batch_{int(datetime.now().timestamp())}"
        )
        repo_dir = BASE_DIR / "yan_kuma_data"
        if not repo_dir.exists():
            git_url = f"https://{GITHUB_USER}:{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git"
            cloned = subprocess.run(
                ["git", "clone", "--branch", GIT_DATA_BRANCH, git_url, str(repo_dir)],
                check=False,
                capture_output=True,
                text=True,
            )
            if cloned.returncode != 0:
                subprocess.run(
                    ["git", "clone", git_url, str(repo_dir)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                subprocess.run(
                    ["git", "checkout", "-b", GIT_DATA_BRANCH],
                    cwd=repo_dir,
                    check=False,
                    capture_output=True,
                    text=True,
                )

        (repo_dir / f"{safe_batch_name}.parquet").write_bytes(parquet_path.read_bytes())

        subprocess.run(
            ["git", "config", "user.email", GITHUB_EMAIL],
            cwd=repo_dir,
            check=False,
        )
        subprocess.run(
            ["git", "config", "user.name", GITHUB_USER],
            cwd=repo_dir,
            check=False,
        )
        subprocess.run(["git", "add", "."], cwd=repo_dir, check=False)
        subprocess.run(
            ["git", "commit", "-m", f"Add batch: {safe_batch_name}"],
            cwd=repo_dir,
            check=False,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "push", "-u", "origin", GIT_DATA_BRANCH],
            cwd=repo_dir,
            check=False,
            capture_output=True,
            text=True,
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


def _safe_batch_name(name: str, default: str = "batch") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name or "").strip("._-")
    if not cleaned:
        return default
    return cleaned[:120]


def _batch_export_candidates(batch_id: int, batch_name: str | None = None) -> list[Path]:
    candidates = []

    if batch_name:
        candidates.append(_safe_batch_name(batch_name, default=f"batch_{batch_id}"))

    candidates.extend([f"batch_{batch_id}", f"{_safe_batch_name(batch_name or 'batch', default=f'batch_{batch_id}')}"])

    ordered = []
    seen = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)

    return [UPLOAD_DIR / f"{candidate}_export.parquet" for candidate in ordered]


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


def youtube_download_with_retries(url, output_path, max_retries=3):
    """Download YouTube video with retry logic and cookie support"""
    
    ydl_opts = {
        'format': 'best[height<=480]',
        'outtmpl': str(output_path),
        'quiet': False,
        'no_warnings': False,
        'socket_timeout': 30,
        'http_headers': {
            'User-Agent': YOUTUBE_USER_AGENT
        },
        'retries': 5,
        'fragment_retries': 5,
        'skip_unavailable_fragments': True,
        # 'web' is the only client that honors cookies for the bot-check;
        # other clients (e.g. visionos) ignore cookies and always require sign-in.
        'extractor_args': {'youtube': {'player_client': ['web']}},
    }
    
    # Add cookies if available (file takes priority over browser extraction)
    if YOUTUBE_COOKIES_FILE and Path(YOUTUBE_COOKIES_FILE).exists():
        ydl_opts['cookiefile'] = YOUTUBE_COOKIES_FILE
        print(f"📍 Using cookies from: {YOUTUBE_COOKIES_FILE}")
    elif YOUTUBE_COOKIES_FROM_BROWSER:
        # Format: "browser" or "browser:profile", e.g. "chrome" or "firefox:default-release"
        browser, _, profile = YOUTUBE_COOKIES_FROM_BROWSER.partition(":")
        ydl_opts['cookiesfrombrowser'] = (browser, profile or None, None, None)
        print(f"📍 Using cookies from browser: {YOUTUBE_COOKIES_FROM_BROWSER}")
    else:
        print("⚠️  No cookies configured. If YouTube blocks, set YOUTUBE_COOKIES_FILE or YOUTUBE_COOKIES_FROM_BROWSER in .env")
    
    for attempt in range(max_retries):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                return ydl.prepare_filename(info), info
        except yt_dlp.utils.ExtractorError as e:
            error_msg = str(e)
            
            # Check for sign-in required error
            if "Sign in to confirm" in error_msg or "bot" in error_msg.lower():
                raise HTTPException(
                    status_code=400,
                    detail=f"YouTube requires authentication. Set YOUTUBE_COOKIES_FILE in .env or use direct video link. See YOUTUBE_COOKIES.md for setup."
                )
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"⚠️  Download failed (attempt {attempt+1}/{max_retries}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise HTTPException(status_code=400, detail=f"YouTube download failed: {error_msg}")
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"⚠️  Error (attempt {attempt+1}/{max_retries}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")

            
@app.post("/upload-youtube")
async def upload_youtube(url: str):
    """Download + process YouTube video with auth support"""
    
    try:
        video_path, info = youtube_download_with_retries(
            url,
            str(UPLOAD_DIR / 'youtube_%(id)s')
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"YouTube download failed: {str(e)}")
    
    audio_path = Path(video_path).with_suffix(".wav")
    try:
        y, sr = librosa.load(video_path, sr=16000)
        sf.write(str(audio_path), y, sr)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio extraction failed: {e}")
    
    segments = segment_by_silence(str(audio_path))
    segments = extract_frames(video_path, segments)
    
    batch_name = f"youtube_{info.get('id', 'unknown')}_{datetime.now().strftime('%H%M%S')}"
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO batches (name, created_at) VALUES (?, ?)", 
                 (batch_name, datetime.now().isoformat()))
    conn.commit()
    batch_id = conn.lastrowid
    conn.close()
    
    conn = sqlite3.connect(DB_PATH)
    for seg in segments:
        conn.execute("""
            INSERT INTO segments 
            (batch_id, segment_id, audio_path, frame_path, start_time, end_time, duration)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (batch_id, seg['segment_id'], seg['audio_path'], seg.get('frame_path', ''),
              seg['start_time'], seg['end_time'], seg['duration']))
    conn.commit()
    conn.close()
    
    storage_mb = get_storage_usage_mb()
    
    return {
        "batch_id": batch_id,
        "segment_count": len(segments),
        "title": info.get('title', 'Unknown'),
        "batch_name": batch_name,
        "storage_mb": round(storage_mb, 1),
        "storage_warning": storage_mb > MAX_STORAGE_MB
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


@app.post("/asr-transcribe")
def asr_transcribe(audio_path: str):
    """Auto-transcribe audio segment using ASR"""
    if not load_asr_model():
        raise HTTPException(status_code=400, detail="ASR model not available. Install transformers: pip install transformers torch")
    
    try:
        import torch
        
        # Load audio
        speech, sr = librosa.load(audio_path, sr=16000)
        
        # Prepare input
        input_values = asr_processor(speech, sampling_rate=16000, return_tensors="pt").input_values
        
        # Transcribe
        with torch.no_grad():
            logits = asr_model(input_values).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcript = asr_processor.batch_decode(predicted_ids)[0]
        
        return {
            "transcript": transcript,
            "confidence": "auto",
            "model": ASR_MODEL
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"ASR failed: {str(e)}")


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
    safe_batch_name = _safe_batch_name(batch_name, default=f"batch_{batch_id}")
    conn.close()

    if df.empty:
        raise HTTPException(status_code=400, detail="No segments to export")

    export_path = UPLOAD_DIR / f"batch_{batch_id}_export.parquet"
    df.to_parquet(export_path, index=False)

    metadata = {
        "batch_id": batch_id,
        "batch_name": batch_name,
        "exported_at": datetime.now().isoformat(),
        "segment_count": len(df),
        "annotated_count": (df["transcript_bm"].notna() & (df["transcript_bm"] != "")).sum(),
        "total_duration": float(df["duration"].sum()),
    }

    metadata_path = UPLOAD_DIR / f"batch_{batch_id}_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    github_ok = push_to_github(export_path, safe_batch_name)
    hf_ok = push_to_huggingface(export_path, safe_batch_name)

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

    export_candidates = _batch_export_candidates(batch_id, batch[0])
    export_path = next((candidate for candidate in export_candidates if candidate.exists()), None)
    if export_path is None:
        raise HTTPException(status_code=404, detail="Export not found")

    return FileResponse(export_path, filename=f"yan_kuma_batch_{batch_id}.parquet")


@app.get("/", response_class=HTMLResponse)
def index():
    with open(BASE_DIR / "index.html", "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
