"""One-time Google Drive OAuth setup for ChatGPT Gemini Media Studio.

1. Enable Google Drive API in a Google Cloud project.
2. Create an OAuth Client ID of type "Desktop app" and download its JSON.
3. Put the JSON next to this script (default: google-oauth-client.json).
4. Set GOOGLE_OAUTH_CLIENT_JSON in .env if you used another filename.
5. Run: python setup_google_drive.py

A browser opens once. After approval, a refreshable token is saved locally and
server_mcp.py can search/import/upload Drive files as the signed-in user.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

SCOPES = ["https://www.googleapis.com/auth/drive"]
CLIENT_NAME = os.getenv("GOOGLE_OAUTH_CLIENT_JSON", "google-oauth-client.json").strip()
TOKEN_NAME = os.getenv("GOOGLE_OAUTH_TOKEN_JSON", "google-drive-token.json").strip()


def resolve(name: str) -> Path:
    path = Path(name).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def main():
    client_path = resolve(CLIENT_NAME)
    token_path = resolve(TOKEN_NAME)

    if not client_path.exists():
        raise SystemExit(
            f"OAuth client JSON not found: {client_path}\n"
            "Download a Google OAuth Desktop-app client JSON and set GOOGLE_OAUTH_CLIENT_JSON in .env."
        )

    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise SystemExit(
            "Missing Drive packages. Run install_media_addons.bat or:\n"
            "python -m pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
        ) from exc

    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(client_path), SCOPES)
            creds = flow.run_local_server(port=0, prompt="consent")
        token_path.write_text(creds.to_json(), encoding="utf-8")

    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    about = service.about().get(fields="user(displayName,emailAddress)").execute()
    user = about.get("user", {})

    print("\nGoogle Drive connected successfully.")
    print(f"Account: {user.get('displayName') or ''} <{user.get('emailAddress') or ''}>")
    print(f"Token saved: {token_path}")
    print("You can now start server_mcp.py and use Drive search/import/export from ChatGPT.")


if __name__ == "__main__":
    main()
