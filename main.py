import os
from pathlib import Path
import subprocess
import whisper

def load_media_file(filename: str) -> Path:
    media_path = Path("recordings") / filename
    if not media_path.exists():
        raise FileNotFoundError(f"File not found: {media_path}")
    return media_path

def extract_audio(video_path: Path) -> Path:
    audio_path = video_path.with_suffix(".wav")
    command = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-ac", "1",      # mono channel
        "-ar", "16000",  # 16 kHz
        str(audio_path)
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return audio_path

def transcribe_audio(audio_path: Path) -> str:
    model = whisper.load_model("base")  # or "small", "medium", "large"
    result = model.transcribe(str(audio_path))
    text = result.get("text", "").strip()
    print("Transcription result:", text)
    return text

def save_transcript(text: str, original_file: str):
    output_dir = Path("transcripts")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / (Path(original_file).stem + ".txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Transcript saved to {output_file}")

def main():
    filename = input("Enter the filename in recordings/: ").strip()
    media_path = load_media_file(filename)

    if media_path.suffix.lower() in [".mp4", ".mkv"]:
        print("Extracting audio from video...")
        audio_path = extract_audio(media_path)
    elif media_path.suffix.lower() in [".wav", ".mp3"]:
        audio_path = media_path
    else:
        raise ValueError("Unsupported file type")

    print("Transcribing audio...")
    transcript = transcribe_audio(audio_path)

    print("Saving transcript...")
    save_transcript(transcript, filename)

if __name__ == "__main__":
    main()
