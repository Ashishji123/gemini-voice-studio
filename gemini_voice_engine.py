import os
import json
import wave
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Prebuilt voice profiles
VOICE_PROFILES = {
    "Charon": {"gender": "Male", "description": "Deep, confident, engaging. Perfect for tech reviews, storytelling, and energetic hooks."},
    "Puck": {"gender": "Male", "description": "Youthful, friendly, dynamic. Great for gaming, casual vlogs, and comedy."},
    "Fenrir": {"gender": "Male", "description": "Authoritative, intense, resonant. Ideal for dramatic narrations and trailers."},
    "Aoede": {"gender": "Female", "description": "Warm, melodious, expressive. Great for tutorials, explainer videos, and storytelling."},
    "Kore": {"gender": "Female", "description": "Clear, professional, soothing. Excellent for informative guides and documentaries."},
    "Zephyr": {"gender": "Female", "description": "Brisk, cheerful, modern. Good for shorts, reels, and fast updates."},
    "Orus": {"gender": "Male", "description": "Direct, conversational, natural. Great for podcasts and reviews."},
    "Leda": {"gender": "Female", "description": "Elegant, gentle, articulate. Perfect for audiobooks and educational content."}
}

# Tone and Style Presets for Hindi / Hinglish content
TONE_PRESETS = {
    "tech_youtuber": "Excited Indian tech YouTuber. Natural human conversational delivery. Medium-fast pace. Sound genuinely impressed and energetic. Do not sound like a monotone news reader.",
    "gaming_energetic": "High energy Indian gaming streamer. Fast-paced, hyped up, reactive and fun. Full of excitement and spontaneous emotion.",
    "analytical_review": "Calm, analytical, objective tech reviewer. Slower, precise, articulate and thoughtful pacing.",
    "curious_hook": "Curious, mysterious, hook the audience immediately with an intriguing and energetic opening question.",
    "cinematic_dramatic": "Dramatic, cinematic narrator. Deep emotion, dramatic pauses, impactful and epic delivery.",
    "calm_tutorial": "Friendly, patient teacher. Clear, step-by-step, soothing and easily understandable pace.",
    "confident_outro": "Warm, confident, engaging outro. Call to action with high positivity and genuine smile in the voice."
}

TTS_MODELS = [
    "gemini-3.1-flash-tts-preview",
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-pro-preview-tts"
]

TEXT_MODELS = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash"
]

class GeminiVoiceStudio:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set. Please add it to your .env file.")
        self.client = genai.Client(api_key=self.api_key)
        self.output_dir = Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for Google Drive folder
        self.gdrive_dir = Path("G:/My Drive/GeminiVoiceovers")
        if not self.gdrive_dir.exists():
            try:
                self.gdrive_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                self.gdrive_dir = None

    def _sync_to_gdrive(self, file_path: Path):
        """Automatically copies file to Google Drive if available."""
        if self.gdrive_dir and file_path.exists():
            try:
                shutil.copy2(file_path, self.gdrive_dir / file_path.name)
            except Exception:
                pass

    def _convert_pcm_to_wav(self, pcm_bytes: bytes, wav_path: Path, sample_rate: int = 24000) -> Path:
        """Saves raw 16-bit linear PCM bytes into a standard WAV file."""
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setnchannels(1)       # Mono
            wav_file.setsampwidth(2)      # 16-bit = 2 bytes
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_bytes)
        return wav_path

    def _convert_wav_to_mp3(self, wav_path: Path, mp3_path: Optional[Path] = None) -> Path:
        """Converts WAV to high quality MP3 using FFmpeg."""
        if mp3_path is None:
            mp3_path = wav_path.with_suffix(".mp3")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(wav_path),
            "-codec:a", "libmp3lame",
            "-b:a", "192k",
            str(mp3_path)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return mp3_path

    def generate_speech(
        self,
        text: str,
        voice_name: str = "Charon",
        tone_instruction: str = "",
        output_filename: Optional[str] = None,
        language: str = "Hindi"
    ) -> Dict[str, str]:
        """
        Generates TTS speech using Google Gemini API.
        
        Args:
            text: The Hindi / Hinglish / English narration text.
            voice_name: Prebuilt voice (e.g. Charon, Puck, Aoede, Fenrir, Kore).
            tone_instruction: Style/tone instructions (e.g. "Excited tech YouTuber").
            output_filename: Base name for the audio file (without extension).
            language: Language hint (Hindi, Hinglish, English).
        
        Returns:
            Dict containing wav_path, mp3_path, and duration_seconds.
        """
        if not output_filename:
            safe_prefix = "".join(c for c in text[:15] if c.isalnum() or c in (' ', '_', '-')).strip()
            safe_prefix = safe_prefix.replace(' ', '_') or "voiceover"
            output_filename = f"{safe_prefix}_{voice_name.lower()}"

        # Construct prompt with tone and delivery instructions
        prompt_parts = []
        if tone_instruction:
            prompt_parts.append(f"Style & Delivery Instructions: {tone_instruction}")
        if language:
            prompt_parts.append(f"Language: {language}")
        prompt_parts.append(f"Please read the following text with exact natural pronunciation, appropriate pacing, and human emotion:\n\n{text}")
        full_prompt = "\n".join(prompt_parts)

        # Attempt generation across supported TTS models
        last_err = None
        pcm_data = None

        for model_name in TTS_MODELS:
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=voice_name
                                )
                            )
                        )
                    )
                )
                for part in response.candidates[0].content.parts:
                    if part.inline_data and part.inline_data.data:
                        pcm_data = part.inline_data.data
                        break
                if pcm_data:
                    break
            except Exception as e:
                last_err = e
                continue

        if not pcm_data:
            raise RuntimeError(f"Failed to generate audio with Gemini TTS: {last_err}")

        # Save to WAV and MP3
        wav_path = self.output_dir / f"{output_filename}.wav"
        self._convert_pcm_to_wav(pcm_data, wav_path, sample_rate=24000)
        
        mp3_path = self.output_dir / f"{output_filename}.mp3"
        try:
            self._convert_wav_to_mp3(wav_path, mp3_path)
        except Exception:
            mp3_path = wav_path  # Fallback if ffmpeg mp3 fails

        # Automatically copy to Google Drive
        self._sync_to_gdrive(wav_path)
        if mp3_path != wav_path:
            self._sync_to_gdrive(mp3_path)

        duration = len(pcm_data) / (24000 * 2)  # 24kHz, 16-bit (2 bytes) mono

        return {
            "wav_path": str(wav_path),
            "mp3_path": str(mp3_path),
            "filename_wav": wav_path.name,
            "filename_mp3": mp3_path.name,
            "duration_seconds": round(duration, 2),
            "voice": voice_name,
            "text": text
        }

    def generate_scene_timeline(
        self,
        scenes: List[Dict[str, str]],
        default_voice: str = "Charon",
        merge_master: bool = True,
        master_name: str = "master_voiceover"
    ) -> Dict:
        """
        Generates individual audio files for a list of scenes and optionally merges them into one master track.
        
        scenes: List of dicts, each having:
          - id or name: e.g. "01_hook"
          - timestamp: e.g. "0:00"
          - tone: e.g. "curious_hook" or custom string
          - text: Narration text
          - voice (optional): Specific voice override
        """
        results = []
        wav_files_to_merge = []

        for idx, scene in enumerate(scenes):
            scene_name = scene.get("name") or scene.get("id") or f"scene_{idx+1:02d}"
            scene_voice = scene.get("voice") or default_voice
            scene_tone = scene.get("tone") or ""
            # Expand preset tone if key exists
            tone_desc = TONE_PRESETS.get(scene_tone, scene_tone)
            text = scene.get("text", "").strip()
            if not text:
                continue

            out_prefix = f"{idx+1:02d}_{scene_name}_{scene_voice.lower()}"
            res = self.generate_speech(
                text=text,
                voice_name=scene_voice,
                tone_instruction=tone_desc,
                output_filename=out_prefix
            )
            res["scene_id"] = scene.get("id", str(idx+1))
            res["scene_name"] = scene_name
            res["timestamp"] = scene.get("timestamp", "")
            res["tone"] = scene_tone
            results.append(res)
            wav_files_to_merge.append(Path(res["wav_path"]))

        master_info = None
        if merge_master and wav_files_to_merge:
            master_info = self.merge_audio_files(wav_files_to_merge, master_name)

        return {
            "scenes": results,
            "master_audio": master_info
        }

    def merge_audio_files(self, audio_paths: List[Path], output_name: str = "master_voiceover") -> Dict[str, str]:
        """Merges multiple WAV files sequentially into a single WAV and MP3 file."""
        concat_txt = self.output_dir / f"{output_name}_concat.txt"
        with open(concat_txt, "w", encoding="utf-8") as f:
            for p in audio_paths:
                # Use forward slashes for FFmpeg concat list
                clean_path = str(p.resolve()).replace("\\", "/")
                f.write(f"file '{clean_path}'\n")

        master_wav = self.output_dir / f"{output_name}.wav"
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_txt),
            "-c", "copy",
            str(master_wav)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if concat_txt.exists():
            concat_txt.unlink()

        master_mp3 = self._convert_wav_to_mp3(master_wav)

        # Get total duration
        with wave.open(str(master_wav), "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            duration = frames / float(rate)

        return {
            "master_wav": str(master_wav),
            "master_mp3": str(master_mp3),
            "duration_seconds": round(duration, 2)
        }

    def attach_audio_to_video(
        self,
        video_path: str,
        audio_path: str,
        output_video_path: Optional[str] = None,
        replace_original_audio: bool = True
    ) -> str:
        """
        Combines or replaces the audio stream of a video file with the generated voiceover audio using FFmpeg.
        """
        in_vid = Path(video_path)
        in_aud = Path(audio_path)
        if not in_vid.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not in_aud.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if not output_video_path:
            output_video_path = str(self.output_dir / f"{in_vid.stem}_with_voiceover{in_vid.suffix}")

        if replace_original_audio:
            # Completely replace audio
            cmd = [
                "ffmpeg", "-y",
                "-i", str(in_vid),
                "-i", str(in_aud),
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                output_video_path
            ]
        else:
            # Mix both original and new voiceover audio together
            cmd = [
                "ffmpeg", "-y",
                "-i", str(in_vid),
                "-i", str(in_aud),
                "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=2[a]",
                "-map", "0:v:0",
                "-map", "[a]",
                "-c:v", "copy",
                "-c:a", "aac",
                output_video_path
            ]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_video_path

    def ai_breakdown_script(self, raw_script: str, target_audience: str = "Indian Tech & Gaming YouTube") -> List[Dict]:
        """
        Uses Gemini to analyze a raw script or topic and break it into scene-by-scene Hindi voiceover cards with timestamps and emotion/tone tags.
        """
        prompt = f"""
You are an expert YouTube video director and Hindi voiceover scriptwriter.
Take this video script or topic and break it down into timed scenes suitable for voiceover generation.

Target Audience / Style: {target_audience}
Input Script/Idea:
{raw_script}

For each scene, output:
1. "name": Short scene tag (e.g., "hook", "gameplay", "fps_test", "heat_test", "conclusion", "outro")
2. "timestamp": Estimated timestamp (e.g., "0:00", "0:20", "0:55", "1:30", "2:10")
3. "tone": Clear tone description (e.g., "Excited Indian tech YouTuber, energetic hook", "Analytical & focused, medium pace", "Hyped gaming streamer", "Confident outro with warm smile")
4. "text": The exact natural Hindi / Hinglish dialogue for the narrator to speak. Write in Romanized Hindi (Hinglish) or Devanagari Hindi that sounds 100% natural, conversational, and energetic.

Return ONLY a valid JSON array of objects with the keys: name, timestamp, tone, text. No markdown ticks, no commentary.
"""
        response_text = ""
        for model_name in TEXT_MODELS:
            try:
                res = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                if res.text:
                    response_text = res.text.strip()
                    break
            except Exception:
                continue

        # Clean JSON markdown if wrapped
        if response_text.startswith("```"):
            lines = response_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            response_text = "\n".join(lines).strip()

        try:
            return json.loads(response_text)
        except Exception:
            # Fallback if json parsing fails
            return [
                {
                    "name": "full_script",
                    "timestamp": "0:00",
                    "tone": "Excited Indian tech YouTuber. Natural human delivery.",
                    "text": raw_script
                }
            ]
