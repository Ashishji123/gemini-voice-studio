import os
import json
import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from gemini_voice_engine import GeminiVoiceStudio, VOICE_PROFILES, TONE_PRESETS

load_dotenv()

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(parents=True, exist_ok=True)

studio = GeminiVoiceStudio()

PUBLIC_BASE_URL = os.getenv("RENDER_EXTERNAL_URL", os.getenv("PUBLIC_URL", "http://127.0.0.1:8000")).rstrip("/")

# Initialize FastAPI App
app = FastAPI(
    title="Gemini Voice Studio Local",
    description="Local Google Gemini Voiceover & Video Audio Studio with Direct Cloud Download Links"
)

from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Gemini Voice Studio Local",
        version="1.0.0",
        description="Local Google Gemini Voiceover Studio with Direct Download Links",
        routes=app.routes,
        servers=[{"url": PUBLIC_BASE_URL}]
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- REST Models & Endpoints -----------------
class SingleGenerateRequest(BaseModel):
    text: str
    voice: str = "Charon"
    tone: Optional[str] = ""
    language: Optional[str] = "Hindi"
    filename: Optional[str] = None

class SceneItem(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    timestamp: Optional[str] = ""
    tone: Optional[str] = ""
    text: str
    voice: Optional[str] = None

class TimelineGenerateRequest(BaseModel):
    scenes: List[SceneItem]
    default_voice: str = "Charon"
    merge_master: bool = True
    master_name: str = "master_voiceover"

class ScriptBreakdownRequest(BaseModel):
    script: str
    target_audience: Optional[str] = "Indian Tech & Gaming YouTube"

class MergeRequest(BaseModel):
    filenames: List[str]
    master_name: str = "merged_voiceover"

@app.get("/.well-known/ai-plugin.json")
def get_ai_plugin():
    manifest_path = Path(".well-known/ai-plugin.json")
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {}

@app.get("/chatgpt-schema")
def get_chatgpt_schema():
    """Serve the clean ChatGPT Action schema with dynamic server URL."""
    schema_path = Path("chatgpt_action_schema.json")
    if schema_path.exists():
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        # Dynamically inject the correct public URL
        schema["servers"] = [{"url": PUBLIC_BASE_URL}]
        return schema
    return {}

@app.get("/api/status")
def get_status():
    api_key = os.getenv("GEMINI_API_KEY", "")
    masked_key = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "Not configured"
    return {
        "status": "online",
        "api_key_configured": bool(api_key),
        "api_key_masked": masked_key,
        "default_voice": "Charon",
        "supported_voices_count": len(VOICE_PROFILES)
    }

@app.get("/api/voices")
def get_voices():
    return {
        "voices": [
            {"name": name, **info}
            for name, info in VOICE_PROFILES.items()
        ]
    }

@app.get("/api/presets")
def get_presets():
    return {
        "presets": [
            {"id": k, "name": k.replace("_", " ").title(), "prompt": v}
            for k, v in TONE_PRESETS.items()
        ]
    }

@app.post("/api/generate")
def generate_single_audio(req: SingleGenerateRequest, request: Request):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    tone_str = TONE_PRESETS.get(req.tone, req.tone) if req.tone else ""
    try:
        result = studio.generate_speech(
            text=req.text,
            voice_name=req.voice,
            tone_instruction=tone_str,
            output_filename=req.filename,
            language=req.language or "Hindi"
        )
        base = str(request.base_url).rstrip("/")
        if "127.0.0.1" in base or "localhost" in base:
            base = PUBLIC_BASE_URL

        fn_wav = result["filename_wav"]
        fn_mp3 = result["filename_mp3"]
        result["download_url_wav"] = f"{base}/output/{fn_wav}"
        result["download_url_mp3"] = f"{base}/output/{fn_mp3}"
        result["direct_drive_download_link"] = f"{base}/output/{fn_mp3}"
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/breakdown")
def breakdown_script(req: ScriptBreakdownRequest):
    if not req.script.strip():
        raise HTTPException(status_code=400, detail="Script text cannot be empty.")
    try:
        scenes = studio.ai_breakdown_script(
            raw_script=req.script,
            target_audience=req.target_audience or "Indian Tech & Gaming YouTube"
        )
        return {"success": True, "scenes": scenes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-timeline")
def generate_timeline(req: TimelineGenerateRequest, request: Request):
    if not req.scenes:
        raise HTTPException(status_code=400, detail="No scenes provided.")
    try:
        scenes_dicts = [s.model_dump() for s in req.scenes]
        result = studio.generate_scene_timeline(
            scenes=scenes_dicts,
            default_voice=req.default_voice,
            merge_master=req.merge_master,
            master_name=req.master_name
        )
        base = str(request.base_url).rstrip("/")
        if "127.0.0.1" in base or "localhost" in base:
            base = PUBLIC_BASE_URL

        for sc in result["scenes"]:
            sc["download_url_wav"] = f"{base}/output/{sc['filename_wav']}"
            sc["download_url_mp3"] = f"{base}/output/{sc['filename_mp3']}"
        if result.get("master_audio"):
            result["master_audio"]["download_master_wav_url"] = f"{base}/output/{req.master_name}.wav"
            result["master_audio"]["download_master_mp3_url"] = f"{base}/output/{req.master_name}.mp3"
            result["master_audio"]["direct_drive_download_link"] = f"{base}/output/{req.master_name}.mp3"

        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/merge")
def merge_audio(req: MergeRequest, request: Request):
    if not req.filenames:
        raise HTTPException(status_code=400, detail="No files provided to merge.")
    paths = []
    for fn in req.filenames:
        p = OUTPUT_DIR / fn
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"File {fn} not found in output folder.")
        paths.append(p)
    try:
        result = studio.merge_audio_files(paths, output_name=req.master_name)
        base = str(request.base_url).rstrip("/")
        if "127.0.0.1" in base or "localhost" in base:
            base = PUBLIC_BASE_URL
        result["download_master_wav"] = f"{base}/output/{req.master_name}.wav"
        result["download_master_mp3"] = f"{base}/output/{req.master_name}.mp3"
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/attach-video")
async def attach_video(
    request: Request,
    video_file: UploadFile = File(...),
    audio_filename: str = Form(...),
    replace_original: bool = Form(True)
):
    audio_path = OUTPUT_DIR / audio_filename
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file {audio_filename} not found.")

    vid_path = UPLOAD_DIR / video_file.filename
    with open(vid_path, "wb") as f:
        shutil.copyfileobj(video_file.file, f)

    out_name = f"{vid_path.stem}_voiceover.mp4"
    out_video_path = OUTPUT_DIR / out_name

    try:
        studio.attach_audio_to_video(
            video_path=str(vid_path),
            audio_path=str(audio_path),
            output_video_path=str(out_video_path),
            replace_original_audio=replace_original
        )
        base = str(request.base_url).rstrip("/")
        if "127.0.0.1" in base or "localhost" in base:
            base = PUBLIC_BASE_URL
        return {
            "success": True,
            "filename": out_name,
            "path": str(out_video_path),
            "url": f"/output/{out_name}",
            "download_url": f"{base}/output/{out_name}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/files")
def list_files(request: Request):
    base = str(request.base_url).rstrip("/")
    if "127.0.0.1" in base or "localhost" in base:
        base = PUBLIC_BASE_URL
    files = []
    for p in sorted(OUTPUT_DIR.glob("*"), key=os.path.getmtime, reverse=True):
        if p.is_file() and p.suffix.lower() in [".wav", ".mp3", ".mp4", ".pcm"]:
            size_kb = round(p.stat().st_size / 1024, 1)
            files.append({
                "filename": p.name,
                "type": "audio" if p.suffix.lower() in [".wav", ".mp3"] else "video" if p.suffix.lower() == ".mp4" else "raw",
                "size_kb": size_kb,
                "url": f"/output/{p.name}",
                "download_url": f"{base}/output/{p.name}",
                "modified": p.stat().st_mtime
            })
    return {"files": files}

@app.delete("/api/files/{filename}")
def delete_file(filename: str):
    p = OUTPUT_DIR / filename
    if p.exists():
        p.unlink()
        return {"success": True, "deleted": filename}
    raise HTTPException(status_code=404, detail="File not found")

# Static files
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")
if (STATIC_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
else:
    @app.get("/")
    def root():
        return {
            "name": "Gemini Voice Studio API",
            "status": "online",
            "docs": "/docs",
            "chatgpt_schema": "/chatgpt-schema"
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"Starting Gemini Voice Studio at http://127.0.0.1:{port} ...")
    uvicorn.run(app, host="0.0.0.0", port=port)
