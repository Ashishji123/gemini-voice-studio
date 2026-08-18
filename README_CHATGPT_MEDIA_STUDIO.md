# ChatGPT Gemini Media Studio v2

This is a drop-in upgrade for the existing Gemini voice server. It keeps **Charon** as the default voice and adds a controlled media-editing layer around FFmpeg.

## What it can do from chat

- Generate Charon MP3/WAV voiceovers.
- Generate scene-by-scene narration and place each scene at an exact video timestamp.
- Mix several generated voices, existing audio clips, music, and SFX.
- Control each layer's start time, volume, trim, speed, fade-in, and fade-out.
- Keep, lower, or completely remove the video's original audio.
- Export a final H.264/AAC MP4 with a direct download URL.
- Trim video, extract MP3, and join multiple videos.
- Import media from an HTTPS URL.
- Search/import/export Google Drive media using personal OAuth (preferred) or an optional service account.
- Accept ChatGPT Action file references when the Action runtime supplies an externally downloadable `download_link`.
- Expose both a **Custom GPT Actions** API and an **MCP** endpoint.

## Files you need to keep together

Keep these files inside the same project folder:

- `server_mcp.py` — new main server.
- your existing `gemini_voice_engine.py` — unchanged.
- your existing Gemini dependencies/config.
- `.env` — copy from `.env.media.example` and fill it in.

The server automatically creates `output/`, `uploads/`, `temp/`, and `static/` if missing.

## 1. Install Python additions

On Windows run:

```bat
install_media_addons.bat
```

or:

```bash
python -m pip install -r requirements_media_addons.txt
```

FFmpeg and ffprobe must also be installed and visible in PATH.

## 2. Configure `.env`

Copy `.env.media.example` to `.env`.

At minimum set:

```env
GEMINI_API_KEY=YOUR_KEY
DEFAULT_VOICE=Charon
PUBLIC_URL=https://YOUR-HTTPS-ADDRESS
```

For local browser testing only, `PUBLIC_URL=http://127.0.0.1:8000` is fine. ChatGPT itself needs an address it can reach.

Optionally set `SERVER_API_KEY`. When enabled, the server accepts either `Authorization: Bearer <key>` or `X-API-Key: <key>` for protected API/MCP calls.

## 3. Run

Double-click:

```text
start_media_studio.bat
```

or run:

```bash
python server_mcp.py
```

Useful URLs:

- Health/status: `/api/status`
- Swagger docs: `/docs`
- Custom GPT Actions schema: `/chatgpt-actions.json`
- MCP endpoint: `/mcp`
- MCP health: `/mcp/health`
- Generated downloads: `/output/<filename>`

## 4A. Recommended personal setup: Custom GPT Actions

In the GPT editor, create an Action and import:

```text
https://YOUR-HTTPS-ADDRESS/chatgpt-actions.json
```

If `SERVER_API_KEY` is blank, use no authentication. If you set it, configure the Action with the same Bearer API key.

Paste the contents of `GPT_INSTRUCTIONS.txt` into the GPT's Instructions.

Then you can ask things like:

```text
Use Charon. Create Hindi YouTube audio for this text and give me the MP3.
```

```text
Use Charon and add these voice scenes to video.mp4:
0:00 intro...
0:18 gaming test...
0:42 battery result...
Keep original sound at 15% and give me the final MP4.
```

```text
Use my Drive video, add Charon voiceover, save the final video back to Drive, and give me the download link.
```

### Chat attachment path

The Actions schema includes `importChatGPTFiles`, which accepts `openaiFileIdRefs`. When the ChatGPT Actions runtime provides an externally downloadable `download_link`, the server imports the attachment automatically. If the runtime does not provide a usable public link, use Drive import or an HTTPS media URL as the fallback.

## 4B. MCP / custom app setup

The same server exposes:

```text
https://YOUR-HTTPS-ADDRESS/mcp
```

Tools include:

- `generate_single_audio`
- `create_youtube_voiceover`
- `render_video_layers`
- `import_media_url`
- `search_google_drive`
- `import_google_drive_file`
- `save_to_google_drive`
- `list_media`
- `probe_media`
- `trim_video`
- `extract_audio`
- `concat_videos`
- `list_voices`

No separate MCP Python package is required; the endpoint speaks JSON-RPC/MCP directly.

## Google Drive setup

Drive integration is optional. **For a personal Google account, OAuth is the recommended mode** because the server can search, import, and upload as your own signed-in Drive account.

1. In Google Cloud, enable the Google Drive API.
2. Configure the OAuth consent screen and create an OAuth Client ID of type **Desktop app**.
3. Download the client JSON into the project folder as `google-oauth-client.json` (or use another name and set it in `.env`).
4. Set:

```env
GOOGLE_OAUTH_CLIENT_JSON=google-oauth-client.json
GOOGLE_OAUTH_TOKEN_JSON=google-drive-token.json
```

5. Run once on your PC:

```bash
python setup_google_drive.py
```

Your browser opens for Google authorization. The resulting refreshable token is stored locally in `google-drive-token.json`.

Optionally set a default media folder:

```env
GOOGLE_DRIVE_FOLDER_ID=YOUR_FOLDER_ID
```

After that ChatGPT can search your connected Drive by filename, import a video/audio file, edit it, and upload the result back to Drive.

### Service-account alternative

`GOOGLE_SERVICE_ACCOUNT_JSON` is still supported for Shared Drive/server environments. If you use it, make sure the target Drive location grants that service account the required access.

## Typical fully automatic flow

1. Video enters server through a ChatGPT Action attachment reference, Drive, HTTPS URL, or `/api/upload-media`.
2. ChatGPT decides the narration/script/timing from the user's instructions and available video context.
3. `createYouTubeVideoVoiceover` generates Charon speech for every scene.
4. FFmpeg places each generated clip at the exact start time and mixes it with the video's original audio.
5. Server exports final MP4 to `output/` and returns a direct download URL.
6. If requested, the same output is uploaded to Google Drive.
7. The output can be imported/reused in a later edit without regenerating it.

## Important safety/security behavior

- Remote URL import blocks localhost/private-network targets to reduce SSRF risk.
- File paths are restricted to the server's `uploads/` and `output/` directories.
- Source files are not overwritten; edits create new files.
- There is no arbitrary shell/PC-file tool exposed to ChatGPT.
- If you expose the server to the public internet, use an API key/authentication and an HTTPS tunnel/domain you control.
