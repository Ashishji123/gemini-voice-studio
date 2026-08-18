import argparse
import json
import sys
from pathlib import Path
from gemini_voice_engine import GeminiVoiceStudio, VOICE_PROFILES, TONE_PRESETS

def main():
    parser = argparse.ArgumentParser(description="Google Gemini Hindi & Multi-Voice Studio CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Command: generate
    gen_p = subparsers.add_parser("generate", help="Generate single audio file from text")
    gen_p.add_argument("--text", "-t", required=True, help="Text to speak (Hindi/Hinglish/English)")
    gen_p.add_argument("--voice", "-v", default="Charon", choices=list(VOICE_PROFILES.keys()), help="Voice name")
    gen_p.add_argument("--tone", help="Tone instruction or preset name")
    gen_p.add_argument("--lang", default="Hindi", help="Language hint")
    gen_p.add_argument("--output", "-o", help="Output filename prefix (saved in output/)")

    # Command: breakdown
    break_p = subparsers.add_parser("breakdown", help="Break down a script into timed scenes using Gemini AI")
    break_p.add_argument("--script", "-s", required=True, help="Path to text script file or raw text string")
    break_p.add_argument("--save-json", help="Path to save output scenes JSON file")

    # Command: batch
    batch_p = subparsers.add_parser("batch", help="Generate audio for a JSON scenes file")
    batch_p.add_argument("--file", "-f", required=True, help="Path to scenes.json file")
    batch_p.add_argument("--voice", "-v", default="Charon", help="Default voice name")
    batch_p.add_argument("--merge", action="store_true", help="Merge all scenes into one master audio track")
    batch_p.add_argument("--master-name", default="master_voiceover", help="Master audio track name")

    # Command: attach-video
    vid_p = subparsers.add_parser("attach-video", help="Attach generated audio to a video file")
    vid_p.add_argument("--video", required=True, help="Path to source MP4 video")
    vid_p.add_argument("--audio", required=True, help="Path to source WAV or MP3 audio")
    vid_p.add_argument("--output", "-o", help="Output video path")
    vid_p.add_argument("--keep-original-audio", action="store_true", help="Mix with original audio instead of replacing")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        studio = GeminiVoiceStudio()
    except Exception as e:
        print(f"Error initializing Voice Studio: {e}")
        sys.exit(1)

    if args.command == "generate":
        print(f"🎙️ Generating voiceover with voice '{args.voice}'...")
        tone_str = TONE_PRESETS.get(args.tone, args.tone) if args.tone else ""
        res = studio.generate_speech(
            text=args.text,
            voice_name=args.voice,
            tone_instruction=tone_str,
            output_filename=args.output,
            language=args.lang
        )
        print(f"✅ Success! Audio generated ({res['duration_seconds']}s):")
        print(f"   WAV: {res['wav_path']}")
        print(f"   MP3: {res['mp3_path']}")

    elif args.command == "breakdown":
        raw_text = args.script
        if Path(args.script).exists():
            raw_text = Path(args.script).read_text(encoding="utf-8")
        print("🧠 Analyzing script with Gemini AI...")
        scenes = studio.ai_breakdown_script(raw_text)
        print(f"✅ Generated {len(scenes)} scenes:")
        print(json.dumps(scenes, indent=2, ensure_ascii=False))
        if args.save_json:
            Path(args.save_json).write_text(json.dumps(scenes, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"💾 Saved to {args.save_json}")

    elif args.command == "batch":
        scenes_data = json.loads(Path(args.file).read_text(encoding="utf-8"))
        print(f"🎬 Processing {len(scenes_data)} scenes...")
        result = studio.generate_scene_timeline(
            scenes=scenes_data,
            default_voice=args.voice,
            merge_master=args.merge,
            master_name=args.master_name
        )
        print(f"✅ Generated {len(result['scenes'])} scene clips.")
        if result.get("master_audio"):
            print(f"🎶 Master audio merged: {result['master_audio']['master_wav']} ({result['master_audio']['duration_seconds']}s)")

    elif args.command == "attach-video":
        print(f"🎥 Attaching audio to video...")
        out_vid = studio.attach_audio_to_video(
            video_path=args.video,
            audio_path=args.audio,
            output_video_path=args.output,
            replace_original_audio=not args.keep_original_audio
        )
        print(f"✅ Video ready: {out_vid}")

if __name__ == "__main__":
    main()
