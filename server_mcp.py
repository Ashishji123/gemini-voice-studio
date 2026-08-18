from __future__ import annotations

import io
import ipaddress
import json
import mimetypes
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from urllib.parse import unquote, urlparse
from urllib.request import HTTPRedirectHandler, Request as UrlRequest, build_opener

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Your existing voice engine. Keep gemini_voice_engine.py next to this server file.
from gemini_voice_engine import GeminiVoiceStudio, VOICE_PROFILES, TONE_PRESETS

# -----------------------------------------------------------------------------
# Paths / configuration
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
os.chdir(ROOT_DIR)  # preserves your existing engine's relative output paths
load_dotenv(ROOT_DIR / ".env")

OUTPUT_DIR = (ROOT_DIR / "output").resolve()
UPLOAD_DIR = (ROOT_DIR / "uploads").resolve()
TEMP_DIR = (ROOT_DIR / "temp").resolve()
STATIC_DIR = (ROOT_DIR / "static").resolve()
for directory in (OUTPUT_DIR, UPLOAD_DIR, TEMP_DIR, STATIC_DIR):
    directory.mkdir(parents=True, exist_ok=True)

DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "Charon")
PUBLIC_BASE_URL = os.getenv(
    "RENDER_EXTERNAL_URL",
    os.getenv("PUBLIC_URL", "http://127.0.0.1:8000"),
).rstrip("/")

FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "2048"))
MAX_REMOTE_IMPORT_MB = int(os.getenv("MAX_REMOTE_IMPORT_MB", "2048"))
PROCESS_TIMEOUT_SECONDS = int(os.getenv("PROCESS_TIMEOUT_SECONDS", "7200"))
SERVER_API_KEY = os.getenv("SERVER_API_KEY", "").strip()

GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
GOOGLE_OAUTH_CLIENT_JSON = os.getenv("GOOGLE_OAUTH_CLIENT_JSON", "").strip()
GOOGLE_OAUTH_TOKEN_JSON = os.getenv("GOOGLE_OAUTH_TOKEN_JSON", "google-drive-token.json").strip()
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "").strip()
GOOGLE_DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]

ALLOWED_MEDIA_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpeg", ".mpg", ".ts",
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma",
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp",
    ".srt", ".ass", ".vtt",
}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpeg", ".mpg", ".ts"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}

studio = GeminiVoiceStudio()

app = FastAPI(
    title="ChatGPT Gemini Media Studio",
    version="2.0.0",
    description=(
        "Charon/Gemini voice generation plus FFmpeg video/audio editing, media import, "
        "Google Drive import/export, Custom GPT Actions, and MCP tools."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Small compatibility helpers
# -----------------------------------------------------------------------------
def _model_dump(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()  # pydantic v2
    return model.dict()  # pydantic v1


def _public_base(request: Optional[Request] = None) -> str:
    if request is not None:
        base = str(request.base_url).rstrip("/")
        if "127.0.0.1" not in base and "localhost" not in base:
            return base
    return PUBLIC_BASE_URL


def _output_url(filename: str, request: Optional[Request] = None) -> str:
    return f"{_public_base(request)}/output/{filename}"


def _safe_filename(name: str, fallback: str = "media") -> str:
    name = Path(unquote(str(name or ""))).name.strip()
    if not name:
        name = fallback
    name = re.sub(r"[^A-Za-z0-9._()\- ]+", "_", name)
    name = re.sub(r"\s+", "_", name).strip("._")
    return name[:180] or fallback


def _ensure_extension(name: str, ext: str) -> str:
    safe = _safe_filename(name)
    ext = ext if ext.startswith(".") else f".{ext}"
    if Path(safe).suffix.lower() != ext.lower():
        safe = f"{Path(safe).stem}{ext}"
    return safe


def _unique_name(name: str, *, directory: Path = OUTPUT_DIR) -> str:
    safe = _safe_filename(name)
    candidate = directory / safe
    if not candidate.exists():
        return safe
    stem, suffix = candidate.stem, candidate.suffix
    return f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"


def _resolve_media_file(filename: str) -> Path:
    safe = _safe_filename(filename)
    for directory in (OUTPUT_DIR, UPLOAD_DIR):
        candidate = (directory / safe).resolve()
        if candidate.parent == directory and candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Media file not found: {safe}")


def _media_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    if ext in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}:
        return "image"
    if ext in {".srt", ".ass", ".vtt"}:
        return "subtitle"
    return "file"


def _run_process(cmd: List[str], *, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout or PROCESS_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required executable not found: {cmd[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Media command timed out after {timeout or PROCESS_TIMEOUT_SECONDS} seconds") from exc

    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "Unknown media processing error").strip()
        if len(stderr) > 6000:
            stderr = stderr[-6000:]
        raise RuntimeError(stderr)
    return result


def _ffmpeg_available() -> bool:
    return shutil.which(FFMPEG_BIN) is not None and shutil.which(FFPROBE_BIN) is not None


def probe_media_file(path: Path) -> dict:
    cmd = [
        FFPROBE_BIN,
        "-v", "error",
        "-show_entries",
        "format=duration,size,bit_rate,format_name:stream=index,codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels",
        "-of", "json",
        str(path),
    ]
    result = _run_process(cmd, timeout=120)
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    duration = None
    try:
        duration = float(fmt.get("duration"))
    except (TypeError, ValueError):
        pass

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    fps = None
    if video_stream:
        raw_fps = str(video_stream.get("r_frame_rate") or "")
        if "/" in raw_fps:
            try:
                numerator, denominator = raw_fps.split("/", 1)
                if float(denominator) != 0:
                    fps = round(float(numerator) / float(denominator), 4)
            except Exception:
                pass

    return {
        "filename": path.name,
        "type": _media_type(path),
        "duration_seconds": duration,
        "size_bytes": path.stat().st_size,
        "format": fmt.get("format_name"),
        "has_video": bool(video_stream),
        "has_audio": bool(audio_stream),
        "video": {
            "codec": video_stream.get("codec_name") if video_stream else None,
            "width": video_stream.get("width") if video_stream else None,
            "height": video_stream.get("height") if video_stream else None,
            "fps": fps,
        },
        "audio": {
            "codec": audio_stream.get("codec_name") if audio_stream else None,
            "sample_rate": audio_stream.get("sample_rate") if audio_stream else None,
            "channels": audio_stream.get("channels") if audio_stream else None,
        },
    }


def _atempo_filters(speed: float) -> List[str]:
    if speed <= 0:
        raise ValueError("speed must be greater than 0")
    filters: List[str] = []
    remaining = speed
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    if abs(remaining - 1.0) > 1e-6:
        filters.append(f"atempo={remaining:.6f}")
    return filters


# -----------------------------------------------------------------------------
# Authentication - optional API key for public deployments / GPT Actions
# -----------------------------------------------------------------------------
@app.middleware("http")
async def optional_api_key_auth(request: Request, call_next):
    if not SERVER_API_KEY:
        return await call_next(request)

    protected = request.url.path.startswith("/api/") or request.url.path == "/mcp"
    public_exceptions = {
        "/api/status",
        "/api/health",
        "/api/actions-schema",
        "/chatgpt-actions.json",
        "/chatgpt-schema",
    }
    if protected and request.url.path not in public_exceptions:
        auth = request.headers.get("authorization", "")
        x_key = request.headers.get("x-api-key", "")
        bearer_ok = auth == f"Bearer {SERVER_API_KEY}"
        key_ok = x_key == SERVER_API_KEY
        if not (bearer_ok or key_ok):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)


# -----------------------------------------------------------------------------
# Request models
# -----------------------------------------------------------------------------
class SingleGenerateRequest(BaseModel):
    text: str
    voice: str = DEFAULT_VOICE
    tone: Optional[str] = ""
    language: Optional[str] = "Hindi"
    filename: Optional[str] = None
    save_to_drive: bool = False
    drive_folder_id: Optional[str] = None


class SceneItem(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    timestamp: Optional[str] = ""
    start_seconds: Optional[float] = None
    tone: Optional[str] = ""
    text: str
    voice: Optional[str] = None
    language: Optional[str] = "Hindi"
    volume: float = 1.0


class TimelineGenerateRequest(BaseModel):
    scenes: List[SceneItem]
    default_voice: str = DEFAULT_VOICE
    merge_master: bool = True
    master_name: str = "master_voiceover"
    save_to_drive: bool = False
    drive_folder_id: Optional[str] = None


class ScriptBreakdownRequest(BaseModel):
    script: str
    target_audience: Optional[str] = "Indian Tech & Gaming YouTube"


class MergeRequest(BaseModel):
    filenames: List[str]
    master_name: str = "merged_voiceover"


class UrlImportRequest(BaseModel):
    url: str
    filename: Optional[str] = None


class ChatGPTFileRef(BaseModel):
    name: Optional[str] = None
    id: Optional[str] = None
    mime_type: Optional[str] = None
    download_link: Optional[str] = None


class ChatGPTFileImportRequest(BaseModel):
    openaiFileIdRefs: List[Union[ChatGPTFileRef, Dict[str, Any], str]] = Field(default_factory=list)


class DriveImportRequest(BaseModel):
    file_id: str
    filename: Optional[str] = None


class DriveExportRequest(BaseModel):
    filename: str
    folder_id: Optional[str] = None
    drive_filename: Optional[str] = None


class DriveSearchRequest(BaseModel):
    name_contains: str = ""
    folder_id: Optional[str] = None
    limit: int = 25


class AudioLayer(BaseModel):
    # Provide filename for an existing audio file OR text to generate a new Charon/Gemini voice file.
    filename: Optional[str] = None
    text: Optional[str] = None
    voice: Optional[str] = DEFAULT_VOICE
    tone: Optional[str] = ""
    language: Optional[str] = "Hindi"
    start_seconds: float = 0.0
    trim_start_seconds: float = 0.0
    trim_end_seconds: Optional[float] = None
    volume: float = 1.0
    speed: float = 1.0
    fade_in_seconds: float = 0.0
    fade_out_seconds: float = 0.0


class RenderVideoRequest(BaseModel):
    video_filename: str
    voice_layers: List[AudioLayer] = Field(default_factory=list)
    audio_layers: List[AudioLayer] = Field(default_factory=list)
    original_audio_volume: float = 0.18
    video_start_seconds: float = 0.0
    video_end_seconds: Optional[float] = None
    output_name: str = "final_youtube_video.mp4"
    video_preset: str = "veryfast"
    video_crf: int = 19
    audio_bitrate: str = "192k"
    save_to_drive: bool = False
    drive_folder_id: Optional[str] = None


class YouTubeVoiceoverRequest(BaseModel):
    video_filename: str
    scenes: List[SceneItem]
    default_voice: str = DEFAULT_VOICE
    original_audio_volume: float = 0.18
    output_name: str = "youtube_voiceover_final.mp4"
    save_to_drive: bool = False
    drive_folder_id: Optional[str] = None


class TrimVideoRequest(BaseModel):
    video_filename: str
    start_seconds: float = 0.0
    end_seconds: Optional[float] = None
    output_name: str = "trimmed_video.mp4"


class ExtractAudioRequest(BaseModel):
    media_filename: str
    output_name: str = "extracted_audio.mp3"


class ConcatVideosRequest(BaseModel):
    video_filenames: List[str]
    output_name: str = "joined_video.mp4"


# -----------------------------------------------------------------------------
# Google Drive helpers (optional)
# -----------------------------------------------------------------------------
def _drive_service():
    """Build a Drive client. Prefer the user's OAuth token; fall back to a service account."""
    try:
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive packages missing. Install google-api-python-client, google-auth, and google-auth-oauthlib."
        ) from exc

    # Personal Google account OAuth is the preferred mode for this local media studio.
    # Run setup_google_drive.py once to create the refreshable token file.
    token_path = Path(GOOGLE_OAUTH_TOKEN_JSON).expanduser()
    if not token_path.is_absolute():
        token_path = (ROOT_DIR / token_path).resolve()
    if token_path.exists():
        try:
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request as GoogleAuthRequest

            credentials = Credentials.from_authorized_user_file(str(token_path), GOOGLE_DRIVE_SCOPES)
            if credentials.expired and credentials.refresh_token:
                credentials.refresh(GoogleAuthRequest())
                token_path.write_text(credentials.to_json(), encoding="utf-8")
            if not credentials.valid:
                raise RuntimeError("Stored Google OAuth token is invalid. Run setup_google_drive.py again.")
            return build("drive", "v3", credentials=credentials, cache_discovery=False)
        except Exception as exc:
            raise RuntimeError(f"Could not load/refresh Google OAuth token: {exc}") from exc

    # Service account remains supported for Shared Drives / server-owned setups.
    if GOOGLE_SERVICE_ACCOUNT_JSON:
        credentials_path = Path(GOOGLE_SERVICE_ACCOUNT_JSON).expanduser()
        if not credentials_path.is_absolute():
            credentials_path = (ROOT_DIR / credentials_path).resolve()
        if not credentials_path.exists():
            raise RuntimeError(f"Google service-account JSON not found: {credentials_path}")
        try:
            from google.oauth2 import service_account
            credentials = service_account.Credentials.from_service_account_file(
                str(credentials_path), scopes=GOOGLE_DRIVE_SCOPES
            )
            return build("drive", "v3", credentials=credentials, cache_discovery=False)
        except Exception as exc:
            raise RuntimeError(f"Could not initialize Google service account: {exc}") from exc

    raise RuntimeError(
        "Google Drive is not authorized. Preferred: set GOOGLE_OAUTH_CLIENT_JSON and run setup_google_drive.py once. "
        "Alternative: set GOOGLE_SERVICE_ACCOUNT_JSON for a Shared Drive/service-account setup."
    )


def _drive_auth_mode() -> str:
    token_path = Path(GOOGLE_OAUTH_TOKEN_JSON).expanduser()
    if not token_path.is_absolute():
        token_path = (ROOT_DIR / token_path).resolve()
    if token_path.exists():
        return "oauth_user"
    if GOOGLE_SERVICE_ACCOUNT_JSON:
        return "service_account"
    return "not_configured"


def search_drive_files(name_contains: str = "", folder_id: Optional[str] = None, limit: int = 25) -> dict:
    service = _drive_service()
    limit = max(1, min(int(limit or 25), 100))
    clauses = ["trashed = false"]
    target_folder = (folder_id or GOOGLE_DRIVE_FOLDER_ID or "").strip()
    if target_folder:
        safe_folder = target_folder.replace("'", "\\'")
        clauses.append(f"'{safe_folder}' in parents")
    if name_contains:
        escaped = str(name_contains).replace("'", "\\'")
        clauses.append(f"name contains '{escaped}'")
    query = " and ".join(clauses)
    try:
        result = service.files().list(
            q=query,
            pageSize=limit,
            orderBy="modifiedTime desc",
            fields="files(id,name,mimeType,size,modifiedTime,webViewLink,webContentLink,parents)",
        ).execute()
    except Exception as exc:
        raise RuntimeError(f"Drive search failed: {exc}") from exc
    return {
        "success": True,
        "auth_mode": _drive_auth_mode(),
        "folder_id": target_folder or None,
        "files": result.get("files", []),
    }


def import_drive_file(file_id: str, filename: Optional[str] = None) -> dict:
    service = _drive_service()
    try:
        metadata = service.files().get(fileId=file_id, fields="id,name,mimeType,size").execute()
    except Exception as exc:
        raise RuntimeError(
            "Could not read the Drive file. Verify the connected Google account/service account can access it. "
            f"Drive error: {exc}"
        ) from exc

    remote_name = filename or metadata.get("name") or f"drive_{file_id}"
    safe_name = _safe_filename(remote_name, "drive_media")
    if Path(safe_name).suffix.lower() not in ALLOWED_MEDIA_EXTENSIONS:
        raise ValueError(f"Unsupported media extension: {Path(safe_name).suffix or '(none)'}")
    safe_name = _unique_name(safe_name, directory=UPLOAD_DIR)
    destination = UPLOAD_DIR / safe_name

    try:
        from googleapiclient.http import MediaIoBaseDownload
        request = service.files().get_media(fileId=file_id)
        with open(destination, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request, chunksize=8 * 1024 * 1024)
            done = False
            while not done:
                _, done = downloader.next_chunk()
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Drive download failed: {exc}") from exc

    return {
        "success": True,
        "filename": destination.name,
        "source": "google_drive",
        "drive_file_id": file_id,
        "mime_type": metadata.get("mimeType"),
        "size_bytes": destination.stat().st_size,
    }


def export_file_to_drive(filename: str, folder_id: Optional[str] = None, drive_filename: Optional[str] = None) -> dict:
    path = _resolve_media_file(filename)
    service = _drive_service()
    target_folder = (folder_id or GOOGLE_DRIVE_FOLDER_ID or "").strip()

    try:
        from googleapiclient.http import MediaFileUpload
        metadata: Dict[str, Any] = {"name": _safe_filename(drive_filename or path.name)}
        if target_folder:
            metadata["parents"] = [target_folder]
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        media = MediaFileUpload(str(path), mimetype=mime_type, resumable=True)
        result = service.files().create(
            body=metadata,
            media_body=media,
            fields="id,name,mimeType,size,webViewLink,webContentLink,parents",
        ).execute()
    except Exception as exc:
        raise RuntimeError(
            "Drive upload failed. Verify the connected Google account/service account can write to the destination folder. "
            f"Drive error: {exc}"
        ) from exc

    return {
        "success": True,
        "drive": result,
        "source_filename": path.name,
    }


# -----------------------------------------------------------------------------
# Safe remote / ChatGPT-file URL importing
# -----------------------------------------------------------------------------
def _is_public_ip(ip_text: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_text)
        return not (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        )
    except ValueError:
        return False


def _validate_remote_url(url: str) -> str:
    parsed = urlparse(str(url).strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http:// or https:// media URLs are accepted.")
    if not parsed.hostname:
        raise ValueError("URL has no hostname.")

    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve URL hostname: {parsed.hostname}") from exc

    for info in infos:
        ip_text = info[4][0]
        if not _is_public_ip(ip_text):
            raise ValueError("For security, remote imports cannot target localhost/private-network addresses.")
    return url


class _SafeRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_remote_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def import_media_url(url: str, filename: Optional[str] = None) -> dict:
    _validate_remote_url(url)
    parsed = urlparse(url)
    source_name = filename or Path(unquote(parsed.path)).name or f"remote_{uuid.uuid4().hex[:8]}.bin"
    safe_name = _safe_filename(source_name, f"remote_{uuid.uuid4().hex[:8]}")

    opener = build_opener(_SafeRedirectHandler())
    request = UrlRequest(url, headers={"User-Agent": "ChatGPT-Gemini-Media-Studio/2.0"})
    max_bytes = MAX_REMOTE_IMPORT_MB * 1024 * 1024

    try:
        with opener.open(request, timeout=60) as response:
            final_url = response.geturl()
            _validate_remote_url(final_url)
            content_type = response.headers.get_content_type()
            if Path(safe_name).suffix == "":
                guessed = mimetypes.guess_extension(content_type or "")
                if guessed:
                    safe_name += guessed
            if Path(safe_name).suffix.lower() not in ALLOWED_MEDIA_EXTENSIONS:
                raise ValueError(
                    f"Unsupported media extension: {Path(safe_name).suffix or '(none)'}. "
                    "Provide filename with a supported video/audio/image/subtitle extension."
                )
            safe_name = _unique_name(safe_name, directory=UPLOAD_DIR)
            destination = UPLOAD_DIR / safe_name

            total = 0
            with open(destination, "wb") as fh:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(f"Remote file exceeds MAX_REMOTE_IMPORT_MB={MAX_REMOTE_IMPORT_MB} MB")
                    fh.write(chunk)
    except Exception:
        if "destination" in locals():
            destination.unlink(missing_ok=True)
        raise

    return {
        "success": True,
        "filename": destination.name,
        "source": "url",
        "size_bytes": destination.stat().st_size,
    }


def import_chatgpt_file_refs(refs: List[Any]) -> dict:
    if not refs:
        raise ValueError(
            "No ChatGPT file references were provided. If attachment forwarding is unavailable, use Drive import or a public HTTPS URL."
        )

    imported = []
    errors = []
    for raw in refs:
        if isinstance(raw, BaseModel):
            raw = _model_dump(raw)
        if isinstance(raw, str):
            if raw.startswith("http://") or raw.startswith("https://"):
                raw = {"download_link": raw}
            else:
                errors.append({"ref": raw, "error": "No externally downloadable URL was included."})
                continue
        if not isinstance(raw, dict):
            errors.append({"ref": str(raw), "error": "Unsupported reference format."})
            continue

        link = raw.get("download_link") or raw.get("url")
        name = raw.get("name")
        if not link or not str(link).startswith(("http://", "https://")):
            errors.append(
                {
                    "ref": raw.get("id") or name or "unknown",
                    "error": "ChatGPT did not provide an externally fetchable download_link. Use Drive import or a public HTTPS URL as fallback.",
                }
            )
            continue
        try:
            imported.append(import_media_url(str(link), filename=name))
        except Exception as exc:
            errors.append({"ref": raw.get("id") or name or "unknown", "error": str(exc)})

    return {"success": bool(imported), "imported": imported, "errors": errors}


# -----------------------------------------------------------------------------
# Voice generation helpers
# -----------------------------------------------------------------------------
def generate_voice_audio(
    *,
    text: str,
    voice: str = DEFAULT_VOICE,
    tone: str = "",
    language: str = "Hindi",
    filename: Optional[str] = None,
    request: Optional[Request] = None,
) -> dict:
    if not str(text).strip():
        raise ValueError("Text cannot be empty.")

    tone_str = TONE_PRESETS.get(tone, tone) if tone else ""
    result = studio.generate_speech(
        text=text,
        voice_name=voice or DEFAULT_VOICE,
        tone_instruction=tone_str,
        output_filename=filename,
        language=language or "Hindi",
    )

    fn_wav = result.get("filename_wav")
    fn_mp3 = result.get("filename_mp3")
    if fn_wav:
        result["download_url_wav"] = _output_url(fn_wav, request)
    if fn_mp3:
        result["download_url_mp3"] = _output_url(fn_mp3, request)
        result["direct_download_link"] = _output_url(fn_mp3, request)
    return {"success": True, "data": result}


def generate_voice_timeline(
    *,
    scenes: List[dict],
    default_voice: str = DEFAULT_VOICE,
    merge_master: bool = True,
    master_name: str = "master_voiceover",
    request: Optional[Request] = None,
) -> dict:
    if not scenes:
        raise ValueError("No scenes provided.")

    clean_scenes = []
    for index, scene in enumerate(scenes, start=1):
        text = str(scene.get("text") or "")
        if not text.strip():
            raise ValueError(f"Scene {index} text cannot be empty.")
        clean_scenes.append(
            {
                "id": scene.get("id"),
                "name": scene.get("name"),
                "timestamp": str(scene.get("timestamp") or ""),
                "tone": str(scene.get("tone") or ""),
                "text": text,
                "voice": scene.get("voice"),
            }
        )

    result = studio.generate_scene_timeline(
        scenes=clean_scenes,
        default_voice=default_voice or DEFAULT_VOICE,
        merge_master=merge_master,
        master_name=master_name,
    )

    for scene in result.get("scenes", []):
        if scene.get("filename_wav"):
            scene["download_url_wav"] = _output_url(scene["filename_wav"], request)
        if scene.get("filename_mp3"):
            scene["download_url_mp3"] = _output_url(scene["filename_mp3"], request)

    if result.get("master_audio"):
        wav_name = f"{master_name}.wav"
        mp3_name = f"{master_name}.mp3"
        result["master_audio"]["download_master_wav_url"] = _output_url(wav_name, request)
        result["master_audio"]["download_master_mp3_url"] = _output_url(mp3_name, request)
        result["master_audio"]["direct_download_link"] = _output_url(mp3_name, request)

    return {"success": True, "data": result}


# -----------------------------------------------------------------------------
# FFmpeg editing helpers
# -----------------------------------------------------------------------------
def _prepare_audio_layer(layer: dict, index: int, request: Optional[Request] = None) -> tuple[Path, dict]:
    filename = layer.get("filename")
    text = str(layer.get("text") or "")
    generated_info: Dict[str, Any] = {}

    if text.strip():
        generated_name = f"layer_{index:02d}_{uuid.uuid4().hex[:8]}"
        generated = generate_voice_audio(
            text=text,
            voice=str(layer.get("voice") or DEFAULT_VOICE),
            tone=str(layer.get("tone") or ""),
            language=str(layer.get("language") or "Hindi"),
            filename=generated_name,
            request=request,
        )
        data = generated["data"]
        filename = data.get("filename_mp3") or data.get("filename_wav")
        generated_info = {
            "generated": True,
            "text": text,
            "voice": str(layer.get("voice") or DEFAULT_VOICE),
            "tone": str(layer.get("tone") or ""),
            "filename": filename,
            "download_url": data.get("download_url_mp3") or data.get("download_url_wav"),
        }

    if not filename:
        raise ValueError(f"Audio layer {index} needs either filename or text.")
    path = _resolve_media_file(str(filename))
    if path.suffix.lower() not in AUDIO_EXTENSIONS:
        raise ValueError(f"Audio layer {index} is not an audio file: {path.name}")
    return path, generated_info


def render_video_with_layers(data: dict, request: Optional[Request] = None) -> dict:
    if not _ffmpeg_available():
        raise RuntimeError("FFmpeg/ffprobe are required but were not found in PATH.")

    video_path = _resolve_media_file(str(data.get("video_filename") or ""))
    if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Base media is not a supported video: {video_path.name}")

    video_probe = probe_media_file(video_path)
    if not video_probe.get("has_video"):
        raise ValueError("Base file has no video stream.")

    original_audio_volume = float(data.get("original_audio_volume", 0.18))
    if original_audio_volume < 0 or original_audio_volume > 4:
        raise ValueError("original_audio_volume must be between 0 and 4.")

    start_seconds = max(0.0, float(data.get("video_start_seconds") or 0.0))
    end_raw = data.get("video_end_seconds")
    end_seconds = float(end_raw) if end_raw is not None else None
    if end_seconds is not None and end_seconds <= start_seconds:
        raise ValueError("video_end_seconds must be greater than video_start_seconds.")

    output_name = _ensure_extension(str(data.get("output_name") or "final_youtube_video.mp4"), ".mp4")
    output_name = _unique_name(output_name, directory=OUTPUT_DIR)
    output_path = OUTPUT_DIR / output_name

    voice_layers = list(data.get("voice_layers") or [])
    audio_layers = list(data.get("audio_layers") or [])
    layers = voice_layers + audio_layers

    prepared: List[dict] = []
    generated_layers: List[dict] = []
    for index, raw_layer in enumerate(layers, start=1):
        if isinstance(raw_layer, BaseModel):
            raw_layer = _model_dump(raw_layer)
        layer = dict(raw_layer)
        path, generated = _prepare_audio_layer(layer, index, request=request)
        prepared.append({"path": path, "layer": layer, "kind": "voice" if index <= len(voice_layers) else "audio"})
        if generated:
            generated_layers.append(generated)

    cmd: List[str] = [FFMPEG_BIN, "-y", "-hide_banner"]
    if start_seconds > 0:
        cmd += ["-ss", f"{start_seconds:.3f}"]
    if end_seconds is not None:
        cmd += ["-t", f"{end_seconds - start_seconds:.3f}"]
    cmd += ["-i", str(video_path)]

    for item in prepared:
        cmd += ["-i", str(item["path"])]

    filters: List[str] = []
    mix_labels: List[str] = []

    if video_probe.get("has_audio") and original_audio_volume > 0:
        filters.append(f"[0:a]volume={original_audio_volume:.6f}[orig]")
        mix_labels.append("[orig]")

    for idx, item in enumerate(prepared, start=1):
        layer = item["layer"]
        trim_start = max(0.0, float(layer.get("trim_start_seconds") or 0.0))
        trim_end_raw = layer.get("trim_end_seconds")
        trim_end = float(trim_end_raw) if trim_end_raw is not None else None
        if trim_end is not None and trim_end <= trim_start:
            raise ValueError(f"Audio layer {idx}: trim_end_seconds must be greater than trim_start_seconds.")

        volume = float(layer.get("volume", 1.0))
        if volume < 0 or volume > 8:
            raise ValueError(f"Audio layer {idx}: volume must be between 0 and 8.")

        speed = float(layer.get("speed", 1.0))
        if speed <= 0 or speed > 8:
            raise ValueError(f"Audio layer {idx}: speed must be > 0 and <= 8.")

        start = max(0.0, float(layer.get("start_seconds") or 0.0))
        fade_in = max(0.0, float(layer.get("fade_in_seconds") or 0.0))
        fade_out = max(0.0, float(layer.get("fade_out_seconds") or 0.0))

        chain: List[str] = []
        if trim_start > 0 or trim_end is not None:
            atrim = f"atrim=start={trim_start:.6f}"
            if trim_end is not None:
                atrim += f":end={trim_end:.6f}"
            chain.append(atrim)
        chain.append("asetpts=PTS-STARTPTS")
        chain.extend(_atempo_filters(speed))
        chain.append(f"volume={volume:.6f}")

        if fade_in > 0:
            chain.append(f"afade=t=in:st=0:d={fade_in:.6f}")

        if fade_out > 0:
            try:
                duration = probe_media_file(item["path"]).get("duration_seconds")
                if duration:
                    effective = max(0.05, (float(duration) - trim_start) / speed)
                    if trim_end is not None:
                        effective = max(0.05, (trim_end - trim_start) / speed)
                    fade_start = max(0.0, effective - fade_out)
                    chain.append(f"afade=t=out:st={fade_start:.6f}:d={fade_out:.6f}")
            except Exception:
                pass

        delay_ms = int(round(start * 1000))
        if delay_ms > 0:
            chain.append(f"adelay={delay_ms}:all=1")

        label = f"a{idx}"
        filters.append(f"[{idx}:a]{','.join(chain)}[{label}]")
        mix_labels.append(f"[{label}]")

    if mix_labels:
        if len(mix_labels) == 1:
            filters.append(f"{mix_labels[0]}anull[mixed]")
        else:
            filters.append(
                f"{''.join(mix_labels)}amix=inputs={len(mix_labels)}:duration=longest:dropout_transition=0:normalize=0,"
                "alimiter=limit=0.95[mixed]"
            )
        cmd += ["-filter_complex", ";".join(filters), "-map", "0:v:0", "-map", "[mixed]"]
    else:
        cmd += ["-map", "0:v:0", "-an"]

    preset = str(data.get("video_preset") or "veryfast")
    crf = int(data.get("video_crf") or 19)
    if crf < 0 or crf > 51:
        raise ValueError("video_crf must be between 0 and 51.")

    cmd += [
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
    ]
    if mix_labels:
        cmd += ["-c:a", "aac", "-b:a", str(data.get("audio_bitrate") or "192k"), "-ar", "48000"]
    cmd += ["-movflags", "+faststart", "-shortest", str(output_path)]

    started = time.time()
    _run_process(cmd)
    elapsed = round(time.time() - started, 2)
    output_probe = probe_media_file(output_path)

    payload: Dict[str, Any] = {
        "success": True,
        "filename": output_name,
        "download_url": _output_url(output_name, request),
        "duration_seconds": output_probe.get("duration_seconds"),
        "size_bytes": output_path.stat().st_size,
        "render_seconds": elapsed,
        "generated_voice_layers": generated_layers,
        "source_video": video_path.name,
        "original_audio_volume": original_audio_volume,
    }

    if bool(data.get("save_to_drive")):
        payload["drive_export"] = export_file_to_drive(
            output_name,
            folder_id=data.get("drive_folder_id"),
        )
    return payload


def create_youtube_voiceover(data: dict, request: Optional[Request] = None) -> dict:
    scenes = data.get("scenes") or []
    if not scenes:
        raise ValueError("At least one voiceover scene is required.")

    voice_layers = []
    default_voice = str(data.get("default_voice") or DEFAULT_VOICE)
    for index, raw_scene in enumerate(scenes, start=1):
        if isinstance(raw_scene, BaseModel):
            raw_scene = _model_dump(raw_scene)
        scene = dict(raw_scene)
        text = str(scene.get("text") or "")
        if not text.strip():
            raise ValueError(f"Scene {index} text cannot be empty.")

        start = scene.get("start_seconds")
        if start is None:
            timestamp = str(scene.get("timestamp") or "").strip()
            if timestamp:
                start = _timestamp_to_seconds(timestamp)
            else:
                start = 0.0

        voice_layers.append(
            {
                "text": text,
                "voice": scene.get("voice") or default_voice,
                "tone": scene.get("tone") or "",
                "language": scene.get("language") or "Hindi",
                "start_seconds": float(start),
                "volume": float(scene.get("volume") or 1.0),
            }
        )

    render_payload = {
        "video_filename": data.get("video_filename"),
        "voice_layers": voice_layers,
        "audio_layers": [],
        "original_audio_volume": float(data.get("original_audio_volume", 0.18)),
        "output_name": data.get("output_name") or "youtube_voiceover_final.mp4",
        "save_to_drive": bool(data.get("save_to_drive")),
        "drive_folder_id": data.get("drive_folder_id"),
    }
    result = render_video_with_layers(render_payload, request=request)
    result["workflow"] = "youtube_voiceover"
    result["default_voice"] = default_voice
    result["scene_count"] = len(scenes)
    return result


def _timestamp_to_seconds(value: str) -> float:
    value = value.strip()
    if not value:
        return 0.0
    parts = value.split(":")
    try:
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except ValueError as exc:
        raise ValueError(f"Invalid timestamp: {value}. Use seconds, MM:SS, or HH:MM:SS.") from exc


def trim_video_file(data: dict, request: Optional[Request] = None) -> dict:
    source = _resolve_media_file(str(data.get("video_filename") or ""))
    start = max(0.0, float(data.get("start_seconds") or 0.0))
    end_raw = data.get("end_seconds")
    end = float(end_raw) if end_raw is not None else None
    if end is not None and end <= start:
        raise ValueError("end_seconds must be greater than start_seconds.")
    output_name = _unique_name(_ensure_extension(str(data.get("output_name") or "trimmed_video.mp4"), ".mp4"))
    out = OUTPUT_DIR / output_name

    cmd = [FFMPEG_BIN, "-y", "-hide_banner", "-ss", f"{start:.3f}"]
    if end is not None:
        cmd += ["-t", f"{end - start:.3f}"]
    cmd += [
        "-i", str(source),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(out),
    ]
    _run_process(cmd)
    return {
        "success": True,
        "filename": output_name,
        "download_url": _output_url(output_name, request),
        "media": probe_media_file(out),
    }


def extract_audio_file(data: dict, request: Optional[Request] = None) -> dict:
    source = _resolve_media_file(str(data.get("media_filename") or ""))
    output_name = _unique_name(_ensure_extension(str(data.get("output_name") or "extracted_audio.mp3"), ".mp3"))
    out = OUTPUT_DIR / output_name
    cmd = [
        FFMPEG_BIN, "-y", "-hide_banner", "-i", str(source),
        "-vn", "-c:a", "libmp3lame", "-q:a", "2", str(out),
    ]
    _run_process(cmd)
    return {
        "success": True,
        "filename": output_name,
        "download_url": _output_url(output_name, request),
        "media": probe_media_file(out),
    }


def concat_video_files(data: dict, request: Optional[Request] = None) -> dict:
    filenames = data.get("video_filenames") or []
    if len(filenames) < 2:
        raise ValueError("Provide at least two videos to concatenate.")
    paths = [_resolve_media_file(str(name)) for name in filenames]
    first_probe = probe_media_file(paths[0])
    width = int((first_probe.get("video") or {}).get("width") or 1280)
    height = int((first_probe.get("video") or {}).get("height") or 720)
    fps = float((first_probe.get("video") or {}).get("fps") or 30.0)

    output_name = _unique_name(_ensure_extension(str(data.get("output_name") or "joined_video.mp4"), ".mp4"))
    out = OUTPUT_DIR / output_name

    cmd: List[str] = [FFMPEG_BIN, "-y", "-hide_banner"]
    probes = []
    for path in paths:
        cmd += ["-i", str(path)]
        probes.append(probe_media_file(path))

    filters: List[str] = []
    concat_inputs: List[str] = []
    for idx, probe in enumerate(probes):
        filters.append(
            f"[{idx}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps:.6f},format=yuv420p[v{idx}]"
        )
        if probe.get("has_audio"):
            filters.append(f"[{idx}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a{idx}]")
        else:
            duration = float(probe.get("duration_seconds") or 0.1)
            filters.append(f"anullsrc=r=48000:cl=stereo:d={duration:.6f}[a{idx}]")
        concat_inputs.append(f"[v{idx}][a{idx}]")

    filters.append(f"{''.join(concat_inputs)}concat=n={len(paths)}:v=1:a=1[vout][aout]")
    cmd += [
        "-filter_complex", ";".join(filters),
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(out),
    ]
    _run_process(cmd)
    return {
        "success": True,
        "filename": output_name,
        "download_url": _output_url(output_name, request),
        "media": probe_media_file(out),
        "sources": [p.name for p in paths],
    }


# -----------------------------------------------------------------------------
# Media-library helper
# -----------------------------------------------------------------------------
def list_media_files(request: Optional[Request] = None) -> dict:
    items = []
    for area, directory in (("output", OUTPUT_DIR), ("uploads", UPLOAD_DIR)):
        for path in directory.iterdir():
            if not path.is_file() or path.suffix.lower() not in ALLOWED_MEDIA_EXTENSIONS:
                continue
            item = {
                "filename": path.name,
                "area": area,
                "type": _media_type(path),
                "size_bytes": path.stat().st_size,
                "modified": path.stat().st_mtime,
            }
            if area == "output":
                item["download_url"] = _output_url(path.name, request)
            items.append(item)
    items.sort(key=lambda x: x["modified"], reverse=True)
    return {"success": True, "files": items}


# -----------------------------------------------------------------------------
# REST endpoints - usable by Custom GPT Actions, browser, curl, etc.
# -----------------------------------------------------------------------------
@app.get("/api/status")
def api_status():
    api_key = os.getenv("GEMINI_API_KEY", "")
    masked_key = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "Not configured"
    return {
        "status": "online",
        "version": "2.0.0",
        "gemini_api_key_configured": bool(api_key),
        "gemini_api_key_masked": masked_key,
        "default_voice": DEFAULT_VOICE,
        "supported_voices_count": len(VOICE_PROFILES),
        "ffmpeg_ready": _ffmpeg_available(),
        "google_drive_configured": _drive_auth_mode() != "not_configured",
        "google_drive_auth_mode": _drive_auth_mode(),
        "google_oauth_client_configured": bool(GOOGLE_OAUTH_CLIENT_JSON),
        "api_key_protection_enabled": bool(SERVER_API_KEY),
        "mcp_endpoint": "/mcp",
        "gpt_actions_schema": "/chatgpt-actions.json",
    }


@app.get("/api/voices")
def api_voices():
    return {"voices": [{"name": name, **info} for name, info in VOICE_PROFILES.items()]}


@app.get("/api/presets")
def api_presets():
    return {"presets": [{"id": key, "name": key.replace("_", " ").title(), "prompt": value} for key, value in TONE_PRESETS.items()]}


@app.post("/api/generate", operation_id="generateSingleAudio")
def api_generate(req: SingleGenerateRequest, request: Request):
    try:
        result = generate_voice_audio(
            text=req.text,
            voice=req.voice,
            tone=req.tone or "",
            language=req.language or "Hindi",
            filename=req.filename,
            request=request,
        )
        if req.save_to_drive:
            filename = result["data"].get("filename_mp3") or result["data"].get("filename_wav")
            result["drive_export"] = export_file_to_drive(filename, folder_id=req.drive_folder_id)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/generate-timeline", operation_id="generateTimeline")
def api_generate_timeline(req: TimelineGenerateRequest, request: Request):
    try:
        result = generate_voice_timeline(
            scenes=[_model_dump(s) for s in req.scenes],
            default_voice=req.default_voice,
            merge_master=req.merge_master,
            master_name=req.master_name,
            request=request,
        )
        if req.save_to_drive and req.merge_master:
            result["drive_export"] = export_file_to_drive(f"{req.master_name}.mp3", folder_id=req.drive_folder_id)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/breakdown", operation_id="breakdownVoiceoverScript")
def api_breakdown(req: ScriptBreakdownRequest):
    try:
        scenes = studio.ai_breakdown_script(
            raw_script=req.script,
            target_audience=req.target_audience or "Indian Tech & Gaming YouTube",
        )
        return {"success": True, "scenes": scenes}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/import-url", operation_id="importMediaFromUrl")
def api_import_url(req: UrlImportRequest):
    try:
        return import_media_url(req.url, req.filename)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/import-chatgpt-files", operation_id="importChatGPTFiles")
def api_import_chatgpt_files(req: ChatGPTFileImportRequest):
    try:
        return import_chatgpt_file_refs(req.openaiFileIdRefs)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/import-drive", operation_id="importGoogleDriveFile")
def api_import_drive(req: DriveImportRequest):
    try:
        return import_drive_file(req.file_id, req.filename)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/search-drive", operation_id="searchGoogleDriveMedia")
def api_search_drive(req: DriveSearchRequest):
    try:
        return search_drive_files(req.name_contains, req.folder_id, req.limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/save-drive", operation_id="saveMediaToGoogleDrive")
def api_save_drive(req: DriveExportRequest):
    try:
        return export_file_to_drive(req.filename, req.folder_id, req.drive_filename)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/render-video", operation_id="renderVideoWithAudioLayers")
def api_render_video(req: RenderVideoRequest, request: Request):
    try:
        return render_video_with_layers(_model_dump(req), request=request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/youtube-voiceover", operation_id="createYouTubeVideoVoiceover")
def api_youtube_voiceover(req: YouTubeVoiceoverRequest, request: Request):
    try:
        return create_youtube_voiceover(_model_dump(req), request=request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/trim-video", operation_id="trimVideo")
def api_trim_video(req: TrimVideoRequest, request: Request):
    try:
        return trim_video_file(_model_dump(req), request=request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/extract-audio", operation_id="extractAudio")
def api_extract_audio(req: ExtractAudioRequest, request: Request):
    try:
        return extract_audio_file(_model_dump(req), request=request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/concat-videos", operation_id="concatVideos")
def api_concat_videos(req: ConcatVideosRequest, request: Request):
    try:
        return concat_video_files(_model_dump(req), request=request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/media", operation_id="listMediaLibrary")
def api_list_media(request: Request):
    return list_media_files(request=request)


@app.get("/api/probe/{filename}", operation_id="inspectMediaFile")
def api_probe(filename: str):
    try:
        return {"success": True, "media": probe_media_file(_resolve_media_file(filename))}
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/api/upload-media")
async def api_upload_media(file: UploadFile = File(...)):
    safe_name = _safe_filename(file.filename or f"upload_{uuid.uuid4().hex[:8]}")
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_MEDIA_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported media extension: {ext or '(none)'}")
    safe_name = _unique_name(safe_name, directory=UPLOAD_DIR)
    destination = UPLOAD_DIR / safe_name
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    total = 0
    try:
        with open(destination, "wb") as fh:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(status_code=413, detail=f"File exceeds MAX_UPLOAD_MB={MAX_UPLOAD_MB} MB")
                fh.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return {"success": True, "filename": destination.name, "size_bytes": destination.stat().st_size}


# Backwards-compatible endpoint from your old server.
@app.post("/api/attach-video")
async def api_attach_video(
    request: Request,
    video_file: UploadFile = File(...),
    audio_filename: str = Form(...),
    replace_original: bool = Form(True),
):
    safe_video = _safe_filename(video_file.filename or f"video_{uuid.uuid4().hex[:8]}.mp4")
    safe_video = _unique_name(safe_video, directory=UPLOAD_DIR)
    destination = UPLOAD_DIR / safe_video
    with open(destination, "wb") as fh:
        shutil.copyfileobj(video_file.file, fh)

    payload = {
        "video_filename": safe_video,
        "audio_layers": [{"filename": audio_filename, "start_seconds": 0.0, "volume": 1.0}],
        "voice_layers": [],
        "original_audio_volume": 0.0 if replace_original else 1.0,
        "output_name": f"{Path(safe_video).stem}_voiceover.mp4",
    }
    try:
        return render_video_with_layers(payload, request=request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# -----------------------------------------------------------------------------
# Custom GPT Actions OpenAPI schema
# -----------------------------------------------------------------------------
def _action_schema() -> dict:
    # Keep this intentionally focused. Binary browser upload is not exposed as a GPT action;
    # GPT can import a forwarded ChatGPT file reference, Drive file ID, or HTTPS URL instead.
    components = {
        "schemas": {
            "Scene": {
                "type": "object",
                "properties": {
                    "start_seconds": {"type": "number", "description": "Voice start time on the video timeline."},
                    "timestamp": {"type": "string", "description": "Alternative start time such as 00:35."},
                    "text": {"type": "string"},
                    "voice": {"type": "string", "default": DEFAULT_VOICE},
                    "tone": {"type": "string", "description": "Natural-language delivery direction or preset."},
                    "language": {"type": "string", "default": "Hindi"},
                    "volume": {"type": "number", "default": 1.0},
                },
                "required": ["text"],
            },
            "AudioLayer": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Existing server-side audio filename."},
                    "text": {"type": "string", "description": "If provided, generate this text as speech before mixing."},
                    "voice": {"type": "string", "default": DEFAULT_VOICE},
                    "tone": {"type": "string"},
                    "language": {"type": "string", "default": "Hindi"},
                    "start_seconds": {"type": "number", "default": 0},
                    "trim_start_seconds": {"type": "number", "default": 0},
                    "trim_end_seconds": {"type": "number"},
                    "volume": {"type": "number", "default": 1},
                    "speed": {"type": "number", "default": 1},
                    "fade_in_seconds": {"type": "number", "default": 0},
                    "fade_out_seconds": {"type": "number", "default": 0},
                },
            },
        }
    }

    def json_post(operation_id: str, summary: str, description: str, schema: dict):
        return {
            "post": {
                "operationId": operation_id,
                "summary": summary,
                "description": description,
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": schema}},
                },
                "responses": {
                    "200": {"description": "Successful operation"},
                    "400": {"description": "Invalid input"},
                    "500": {"description": "Generation or media processing error"},
                },
            }
        }

    paths = {
        "/api/generate": json_post(
            "generateSingleAudio",
            "Generate Charon voice audio",
            "Generate a downloadable MP3/WAV voiceover. Use Charon unless the user asks for another voice.",
            {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "voice": {"type": "string", "default": DEFAULT_VOICE},
                    "tone": {"type": "string"},
                    "language": {"type": "string", "default": "Hindi"},
                    "filename": {"type": "string"},
                    "save_to_drive": {"type": "boolean", "default": False},
                    "drive_folder_id": {"type": "string"},
                },
                "required": ["text"],
            },
        ),
        "/api/youtube-voiceover": json_post(
            "createYouTubeVideoVoiceover",
            "Create a YouTube video with scene-timed voiceover",
            "Generate Charon voice for each scene, place each voice layer at its timestamp, mix with the original video audio, export MP4, and optionally save to Drive.",
            {
                "type": "object",
                "properties": {
                    "video_filename": {"type": "string"},
                    "scenes": {"type": "array", "items": {"$ref": "#/components/schemas/Scene"}, "minItems": 1},
                    "default_voice": {"type": "string", "default": DEFAULT_VOICE},
                    "original_audio_volume": {"type": "number", "default": 0.18},
                    "output_name": {"type": "string", "default": "youtube_voiceover_final.mp4"},
                    "save_to_drive": {"type": "boolean", "default": False},
                    "drive_folder_id": {"type": "string"},
                },
                "required": ["video_filename", "scenes"],
            },
        ),
        "/api/render-video": json_post(
            "renderVideoWithAudioLayers",
            "Render video with voice, music, or sound layers",
            "Mix generated voice layers and/or existing audio layers over a base video and export MP4.",
            {
                "type": "object",
                "properties": {
                    "video_filename": {"type": "string"},
                    "voice_layers": {"type": "array", "items": {"$ref": "#/components/schemas/AudioLayer"}},
                    "audio_layers": {"type": "array", "items": {"$ref": "#/components/schemas/AudioLayer"}},
                    "original_audio_volume": {"type": "number", "default": 0.18},
                    "video_start_seconds": {"type": "number", "default": 0},
                    "video_end_seconds": {"type": "number"},
                    "output_name": {"type": "string", "default": "final_youtube_video.mp4"},
                    "save_to_drive": {"type": "boolean", "default": False},
                    "drive_folder_id": {"type": "string"},
                },
                "required": ["video_filename"],
            },
        ),
        "/api/import-url": json_post(
            "importMediaFromUrl",
            "Import media from an HTTPS URL",
            "Download a video/audio/image/subtitle file into the server media library.",
            {
                "type": "object",
                "properties": {"url": {"type": "string"}, "filename": {"type": "string"}},
                "required": ["url"],
            },
        ),
        "/api/import-chatgpt-files": json_post(
            "importChatGPTFiles",
            "Import files attached in the GPT conversation",
            "Accept ChatGPT's openaiFileIdRefs file-reference objects when the Actions runtime provides externally downloadable links. If not available, use Drive or HTTPS URL import.",
            {
                "type": "object",
                "properties": {
                    "openaiFileIdRefs": {
                        "type": "array",
                        "items": {
                            "oneOf": [
                                {"type": "string"},
                                {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "id": {"type": "string"},
                                        "mime_type": {"type": "string"},
                                        "download_link": {"type": "string"},
                                    },
                                },
                            ]
                        },
                    }
                },
                "required": ["openaiFileIdRefs"],
            },
        ),
        "/api/search-drive": json_post(
            "searchGoogleDriveMedia",
            "Search Google Drive media",
            "Search the configured Google Drive account/folder by filename so ChatGPT can find a video or audio file before importing it.",
            {
                "type": "object",
                "properties": {
                    "name_contains": {"type": "string", "default": ""},
                    "folder_id": {"type": "string"},
                    "limit": {"type": "integer", "default": 25, "minimum": 1, "maximum": 100},
                },
            },
        ),
        "/api/import-drive": json_post(
            "importGoogleDriveFile",
            "Import a media file from Google Drive",
            "Download a Drive file using the server's configured service account. The file or parent folder must be shared with that service account.",
            {
                "type": "object",
                "properties": {"file_id": {"type": "string"}, "filename": {"type": "string"}},
                "required": ["file_id"],
            },
        ),
        "/api/save-drive": json_post(
            "saveMediaToGoogleDrive",
            "Save generated media to Google Drive",
            "Upload an existing server-side generated audio/video file into the configured Drive folder.",
            {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "folder_id": {"type": "string"},
                    "drive_filename": {"type": "string"},
                },
                "required": ["filename"],
            },
        ),
        "/api/trim-video": json_post(
            "trimVideo",
            "Trim a video",
            "Create a new MP4 from a selected time range.",
            {
                "type": "object",
                "properties": {
                    "video_filename": {"type": "string"},
                    "start_seconds": {"type": "number", "default": 0},
                    "end_seconds": {"type": "number"},
                    "output_name": {"type": "string"},
                },
                "required": ["video_filename"],
            },
        ),
        "/api/extract-audio": json_post(
            "extractAudio",
            "Extract audio from media",
            "Extract the audio track from a video/audio file into MP3.",
            {
                "type": "object",
                "properties": {"media_filename": {"type": "string"}, "output_name": {"type": "string"}},
                "required": ["media_filename"],
            },
        ),
        "/api/concat-videos": json_post(
            "concatVideos",
            "Join multiple videos",
            "Normalize and concatenate multiple videos into one MP4.",
            {
                "type": "object",
                "properties": {
                    "video_filenames": {"type": "array", "items": {"type": "string"}, "minItems": 2},
                    "output_name": {"type": "string"},
                },
                "required": ["video_filenames"],
            },
        ),
    }

    return {
        "openapi": "3.1.0",
        "info": {
            "title": "ChatGPT Gemini Media Studio Actions",
            "description": "Control Charon/Gemini voice generation and FFmpeg video editing directly from a custom GPT.",
            "version": "2.0.0",
        },
        "servers": [{"url": PUBLIC_BASE_URL}],
        "paths": paths,
        "components": components,
    }


@app.get("/chatgpt-actions.json")
def chatgpt_actions_schema():
    schema = _action_schema()
    schema["servers"] = [{"url": PUBLIC_BASE_URL}]
    return schema


@app.get("/chatgpt-schema")
def chatgpt_schema_legacy_alias():
    return chatgpt_actions_schema()


@app.get("/api/actions-schema")
def api_actions_schema_alias():
    return chatgpt_actions_schema()


# Legacy plugin manifest endpoint kept for compatibility with older setups.
@app.get("/.well-known/ai-plugin.json")
def legacy_ai_plugin_manifest():
    manifest_path = ROOT_DIR / ".well-known" / "ai-plugin.json"
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "schema_version": "v1",
        "name_for_human": "ChatGPT Gemini Media Studio",
        "name_for_model": "chatgpt_gemini_media_studio",
        "description_for_human": "Charon voice generation and video/audio editing.",
        "description_for_model": "Generate Charon voice audio, edit video audio layers, import media, and save results to Google Drive.",
        "auth": {"type": "none"},
        "api": {"type": "openapi", "url": f"{PUBLIC_BASE_URL}/chatgpt-actions.json"},
        "logo_url": f"{PUBLIC_BASE_URL}/logo.png",
        "contact_email": "",
        "legal_info_url": "",
    }


# -----------------------------------------------------------------------------
# MCP bridge - same abilities as Actions, for ChatGPT custom apps / MCP clients
# -----------------------------------------------------------------------------
MCP_SERVER_NAME = "chatgpt-gemini-media-studio"
MCP_SERVER_VERSION = "2.0.0"
MCP_DEFAULT_PROTOCOL = "2025-06-18"


def _mcp_ok(request_id, result):
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _mcp_error(request_id, code: int, message: str, data=None):
    error: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def _mcp_tool_result(payload, *, is_error: bool = False):
    return {
        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
        "structuredContent": payload,
        "isError": is_error,
    }


def _tool_annotations(read_only: bool = False, destructive: bool = False):
    return {
        "readOnlyHint": read_only,
        "destructiveHint": destructive,
        "idempotentHint": read_only,
        "openWorldHint": False,
    }


def _mcp_tools() -> List[dict]:
    scene_schema = {
        "type": "object",
        "properties": {
            "start_seconds": {"type": ["number", "null"]},
            "timestamp": {"type": "string", "default": ""},
            "text": {"type": "string"},
            "voice": {"type": ["string", "null"]},
            "tone": {"type": "string", "default": ""},
            "language": {"type": "string", "default": "Hindi"},
            "volume": {"type": "number", "default": 1.0},
        },
        "required": ["text"],
        "additionalProperties": True,
    }
    layer_schema = {
        "type": "object",
        "properties": {
            "filename": {"type": ["string", "null"]},
            "text": {"type": ["string", "null"]},
            "voice": {"type": ["string", "null"], "default": DEFAULT_VOICE},
            "tone": {"type": "string", "default": ""},
            "language": {"type": "string", "default": "Hindi"},
            "start_seconds": {"type": "number", "default": 0},
            "trim_start_seconds": {"type": "number", "default": 0},
            "trim_end_seconds": {"type": ["number", "null"]},
            "volume": {"type": "number", "default": 1},
            "speed": {"type": "number", "default": 1},
            "fade_in_seconds": {"type": "number", "default": 0},
            "fade_out_seconds": {"type": "number", "default": 0},
        },
        "additionalProperties": False,
    }

    return [
        {
            "name": "generate_single_audio",
            "title": "Generate Charon voice audio",
            "description": "Generate downloadable Gemini speech. Use Charon by default for YouTube voiceovers.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "voice": {"type": "string", "default": DEFAULT_VOICE},
                    "tone": {"type": "string", "default": ""},
                    "language": {"type": "string", "default": "Hindi"},
                    "filename": {"type": ["string", "null"]},
                    "save_to_drive": {"type": "boolean", "default": False},
                    "drive_folder_id": {"type": ["string", "null"]},
                },
                "required": ["text"],
                "additionalProperties": False,
            },
            "annotations": _tool_annotations(),
        },
        {
            "name": "create_youtube_voiceover",
            "title": "Create YouTube voiceover video",
            "description": "Generate scene-timed Charon voice layers, mix them with the video's original audio, and export a downloadable MP4.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "video_filename": {"type": "string"},
                    "scenes": {"type": "array", "items": scene_schema, "minItems": 1},
                    "default_voice": {"type": "string", "default": DEFAULT_VOICE},
                    "original_audio_volume": {"type": "number", "default": 0.18},
                    "output_name": {"type": "string", "default": "youtube_voiceover_final.mp4"},
                    "save_to_drive": {"type": "boolean", "default": False},
                    "drive_folder_id": {"type": ["string", "null"]},
                },
                "required": ["video_filename", "scenes"],
                "additionalProperties": False,
            },
            "annotations": _tool_annotations(),
        },
        {
            "name": "render_video_layers",
            "title": "Render video audio layers",
            "description": "Mix generated voice, music, sound effects, or existing audio layers over a video with exact start times, trims, gains, fades, and speed controls.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "video_filename": {"type": "string"},
                    "voice_layers": {"type": "array", "items": layer_schema, "default": []},
                    "audio_layers": {"type": "array", "items": layer_schema, "default": []},
                    "original_audio_volume": {"type": "number", "default": 0.18},
                    "video_start_seconds": {"type": "number", "default": 0},
                    "video_end_seconds": {"type": ["number", "null"]},
                    "output_name": {"type": "string", "default": "final_youtube_video.mp4"},
                    "save_to_drive": {"type": "boolean", "default": False},
                    "drive_folder_id": {"type": ["string", "null"]},
                },
                "required": ["video_filename"],
                "additionalProperties": False,
            },
            "annotations": _tool_annotations(),
        },
        {
            "name": "import_media_url",
            "title": "Import media URL",
            "description": "Import video/audio/image/subtitle media from a public HTTP(S) URL into the server library.",
            "inputSchema": {
                "type": "object",
                "properties": {"url": {"type": "string"}, "filename": {"type": ["string", "null"]}},
                "required": ["url"],
                "additionalProperties": False,
            },
            "annotations": _tool_annotations(),
        },
        {
            "name": "search_google_drive",
            "title": "Search Google Drive media",
            "description": "Search the configured Drive account/folder by filename so a media file can be found before import.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name_contains": {"type": "string", "default": ""},
                    "folder_id": {"type": ["string", "null"]},
                    "limit": {"type": "integer", "default": 25, "minimum": 1, "maximum": 100},
                },
                "additionalProperties": False,
            },
            "annotations": _tool_annotations(read_only=True),
        },
        {
            "name": "import_google_drive_file",
            "title": "Import Google Drive media",
            "description": "Download a Drive media file using the configured service account. The file/folder must be shared with the service account.",
            "inputSchema": {
                "type": "object",
                "properties": {"file_id": {"type": "string"}, "filename": {"type": ["string", "null"]}},
                "required": ["file_id"],
                "additionalProperties": False,
            },
            "annotations": _tool_annotations(),
        },
        {
            "name": "save_to_google_drive",
            "title": "Save media to Google Drive",
            "description": "Upload a generated server-side media file to the configured Drive folder.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "folder_id": {"type": ["string", "null"]},
                    "drive_filename": {"type": ["string", "null"]},
                },
                "required": ["filename"],
                "additionalProperties": False,
            },
            "annotations": _tool_annotations(),
        },
        {
            "name": "list_media",
            "title": "List media library",
            "description": "List uploaded and generated media files available to this media studio.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
            "annotations": _tool_annotations(read_only=True),
        },
        {
            "name": "probe_media",
            "title": "Inspect media",
            "description": "Inspect media duration, codecs, resolution, fps, and audio presence.",
            "inputSchema": {
                "type": "object",
                "properties": {"filename": {"type": "string"}},
                "required": ["filename"],
                "additionalProperties": False,
            },
            "annotations": _tool_annotations(read_only=True),
        },
        {
            "name": "trim_video",
            "title": "Trim video",
            "description": "Create a new MP4 containing a selected time range of an existing video.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "video_filename": {"type": "string"},
                    "start_seconds": {"type": "number", "default": 0},
                    "end_seconds": {"type": ["number", "null"]},
                    "output_name": {"type": "string", "default": "trimmed_video.mp4"},
                },
                "required": ["video_filename"],
                "additionalProperties": False,
            },
            "annotations": _tool_annotations(),
        },
        {
            "name": "extract_audio",
            "title": "Extract audio",
            "description": "Extract an MP3 audio track from an existing video or media file.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "media_filename": {"type": "string"},
                    "output_name": {"type": "string", "default": "extracted_audio.mp3"},
                },
                "required": ["media_filename"],
                "additionalProperties": False,
            },
            "annotations": _tool_annotations(),
        },
        {
            "name": "concat_videos",
            "title": "Join videos",
            "description": "Normalize and concatenate two or more videos into one MP4.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "video_filenames": {"type": "array", "items": {"type": "string"}, "minItems": 2},
                    "output_name": {"type": "string", "default": "joined_video.mp4"},
                },
                "required": ["video_filenames"],
                "additionalProperties": False,
            },
            "annotations": _tool_annotations(),
        },
        {
            "name": "list_voices",
            "title": "List Gemini voices",
            "description": "List voices supported by the existing Gemini voice engine.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
            "annotations": _tool_annotations(read_only=True),
        },
    ]


@app.post("/mcp")
async def mcp_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(_mcp_error(None, -32700, "Parse error: body must be valid JSON."), status_code=400)

    if not isinstance(body, dict):
        return JSONResponse(_mcp_error(None, -32600, "Invalid Request."), status_code=400)

    request_id = body.get("id")
    method = body.get("method")
    params = body.get("params") or {}

    if request_id is None and isinstance(method, str) and method.startswith("notifications/"):
        return Response(status_code=202)

    if body.get("jsonrpc") != "2.0" or not isinstance(method, str):
        return JSONResponse(_mcp_error(request_id, -32600, "Invalid Request."), status_code=400)

    if method == "initialize":
        requested = str(params.get("protocolVersion") or "") if isinstance(params, dict) else ""
        return JSONResponse(
            _mcp_ok(
                request_id,
                {
                    "protocolVersion": requested or MCP_DEFAULT_PROTOCOL,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": MCP_SERVER_NAME, "version": MCP_SERVER_VERSION},
                    "instructions": (
                        "This is the user's private media studio. Default voice is Charon. "
                        "For a normal YouTube request: import/locate the video, call create_youtube_voiceover with scene timestamps, "
                        "then return the MP4 download URL. Save to Google Drive only when requested."
                    ),
                },
            )
        )

    if method == "ping":
        return JSONResponse(_mcp_ok(request_id, {}))
    if method == "tools/list":
        return JSONResponse(_mcp_ok(request_id, {"tools": _mcp_tools()}))

    if method == "tools/call":
        if not isinstance(params, dict):
            return JSONResponse(_mcp_error(request_id, -32602, "Invalid params."), status_code=400)
        name = params.get("name")
        args = params.get("arguments") or {}
        if not isinstance(args, dict):
            return JSONResponse(_mcp_error(request_id, -32602, "Tool arguments must be an object."), status_code=400)
        try:
            if name == "generate_single_audio":
                payload = generate_voice_audio(
                    text=str(args.get("text") or ""),
                    voice=str(args.get("voice") or DEFAULT_VOICE),
                    tone=str(args.get("tone") or ""),
                    language=str(args.get("language") or "Hindi"),
                    filename=args.get("filename"),
                    request=request,
                )
                if args.get("save_to_drive"):
                    filename = payload["data"].get("filename_mp3") or payload["data"].get("filename_wav")
                    payload["drive_export"] = export_file_to_drive(filename, args.get("drive_folder_id"))
            elif name == "create_youtube_voiceover":
                payload = create_youtube_voiceover(args, request=request)
            elif name == "render_video_layers":
                payload = render_video_with_layers(args, request=request)
            elif name == "import_media_url":
                payload = import_media_url(str(args.get("url") or ""), args.get("filename"))
            elif name == "search_google_drive":
                payload = search_drive_files(
                    str(args.get("name_contains") or ""),
                    args.get("folder_id"),
                    int(args.get("limit") or 25),
                )
            elif name == "import_google_drive_file":
                payload = import_drive_file(str(args.get("file_id") or ""), args.get("filename"))
            elif name == "save_to_google_drive":
                payload = export_file_to_drive(str(args.get("filename") or ""), args.get("folder_id"), args.get("drive_filename"))
            elif name == "list_media":
                payload = list_media_files(request=request)
            elif name == "probe_media":
                payload = {"success": True, "media": probe_media_file(_resolve_media_file(str(args.get("filename") or "")))}
            elif name == "trim_video":
                payload = trim_video_file(args, request=request)
            elif name == "extract_audio":
                payload = extract_audio_file(args, request=request)
            elif name == "concat_videos":
                payload = concat_video_files(args, request=request)
            elif name == "list_voices":
                payload = {
                    "success": True,
                    "default_voice": DEFAULT_VOICE,
                    "voices": [{"name": n, **info} for n, info in VOICE_PROFILES.items()],
                }
            else:
                return JSONResponse(_mcp_error(request_id, -32602, f"Unknown tool: {name}"), status_code=400)
            return JSONResponse(_mcp_ok(request_id, _mcp_tool_result(payload)))
        except Exception as exc:
            return JSONResponse(_mcp_ok(request_id, _mcp_tool_result({"success": False, "error": str(exc)}, is_error=True)))

    return JSONResponse(_mcp_error(request_id, -32601, f"Method not found: {method}"), status_code=404)


@app.get("/mcp/health")
def mcp_health():
    return {
        "status": "online",
        "mcp_endpoint": "/mcp",
        "default_voice": DEFAULT_VOICE,
        "tool_count": len(_mcp_tools()),
        "tools": [tool["name"] for tool in _mcp_tools()],
    }


# -----------------------------------------------------------------------------
# Download serving / home
# -----------------------------------------------------------------------------
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

if (STATIC_DIR / "index.html").exists():
    app.mount("/studio", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


@app.get("/")
def root():
    return {
        "name": "ChatGPT Gemini Media Studio",
        "version": "2.0.0",
        "status": "online",
        "default_voice": DEFAULT_VOICE,
        "docs": "/docs",
        "actions_schema": "/chatgpt-actions.json",
        "mcp": "/mcp",
        "media": "/api/media",
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    print(f"Starting ChatGPT Gemini Media Studio at http://127.0.0.1:{port}")
    print(f"Default voice: {DEFAULT_VOICE}")
    print("Custom GPT Actions schema: /chatgpt-actions.json")
    print("MCP endpoint: /mcp")
    uvicorn.run(app, host="0.0.0.0", port=port)
