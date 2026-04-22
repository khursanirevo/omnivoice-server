#!/usr/bin/env python3
"""
Scrape short spoken clips for Malaysian artists and register them as voice profiles.

Usage:
    uv run python scripts/scrape_voices.py              # all artists
    uv run python scripts/scrape_voices.py --artist siti_nurhaliza
    uv run python scripts/scrape_voices.py --list

Requirements:
    pip install yt-dlp  (or: uv add yt-dlp --dev)

Each clip is:
  - Sourced from a known interview/talk segment (not a song)
  - Trimmed to ~20s from a timestamp where the artist speaks clearly
  - Saved to voice_samples/<id>.wav + voice_samples/<id>.txt
  - Optionally registered as a persistent profile on a running server
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Artist registry
# Each entry:
#   id          : profile_id / filename stem (alphanumeric + underscore)
#   name        : display name
#   gender      : for reference
#   search      : yt-dlp ytsearch query string (targets spoken segments)
#   start_sec   : seconds into the video to clip from
#   duration    : how many seconds to extract
#   ref_text    : MUST be filled in manually after reviewing the clip —
#                 left empty here, filled by --transcribe or manual edit
# ---------------------------------------------------------------------------
ARTISTS = [
    # ── Female ───────────────────────────────────────────────────────────────
    {
        "id": "siti_nurhaliza",
        "name": "Siti Nurhaliza",
        "gender": "female",
        "search": "ytsearch1:Siti Nurhaliza interview 2023 berbicara",
        "start_sec": 30,
        "duration": 25,
        "ref_text": "",
    },
    {
        "id": "yuna",
        "name": "Yuna",
        "gender": "female",
        "search": "ytsearch1:Yuna singer Malaysia interview speaking 2023",
        "start_sec": 20,
        "duration": 25,
        "ref_text": "",
    },
    {
        "id": "zee_avi",
        "name": "Zee Avi",
        "gender": "female",
        "search": "ytsearch1:Zee Avi interview Malaysia speaking",
        "start_sec": 15,
        "duration": 25,
        "ref_text": "",
    },
    {
        "id": "shila_hamzah",
        "name": "Shila Hamzah",
        "gender": "female",
        "search": "ytsearch1:Shila Hamzah interview berbicara Melayu 2022",
        "start_sec": 20,
        "duration": 25,
        "ref_text": "",
    },
    {
        "id": "dayang_nurfaizah",
        "name": "Dayang Nurfaizah",
        "gender": "female",
        "search": "ytsearch1:Dayang Nurfaizah interview temu bual 2022",
        "start_sec": 15,
        "duration": 25,
        "ref_text": "",
    },
    {
        "id": "ziana_zain",
        "name": "Ziana Zain",
        "gender": "female",
        "search": "ytsearch1:Ziana Zain interview berbicara temu bual",
        "start_sec": 20,
        "duration": 25,
        "ref_text": "",
    },
    {
        "id": "jaclyn_victor",
        "name": "Jaclyn Victor",
        "gender": "female",
        "search": "ytsearch1:Jaclyn Victor Malaysia interview speaking 2022",
        "start_sec": 15,
        "duration": 25,
        "ref_text": "",
    },
    {
        "id": "elizabeth_tan",
        "name": "Elizabeth Tan",
        "gender": "female",
        "search": "ytsearch1:Elizabeth Tan Malaysia interview berbicara 2023",
        "start_sec": 15,
        "duration": 25,
        "ref_text": "",
    },
    # ── Male ─────────────────────────────────────────────────────────────────
    {
        "id": "anuar_zain",
        "name": "Anuar Zain",
        "gender": "male",
        "search": "ytsearch1:Anuar Zain interview temu bual berbicara",
        "start_sec": 20,
        "duration": 25,
        "ref_text": "",
    },
    {
        "id": "faizal_tahir",
        "name": "Faizal Tahir",
        "gender": "male",
        "search": "ytsearch1:Faizal Tahir interview berbicara temu bual 2023",
        "start_sec": 20,
        "duration": 25,
        "ref_text": "",
    },
    {
        "id": "hael_husaini",
        "name": "Hael Husaini",
        "gender": "male",
        "search": "ytsearch1:Hael Husaini interview berbicara 2023",
        "start_sec": 15,
        "duration": 25,
        "ref_text": "",
    },
    {
        "id": "ismail_izzani",
        "name": "Ismail Izzani",
        "gender": "male",
        "search": "ytsearch1:Ismail Izzani interview berbicara Malaysia 2023",
        "start_sec": 15,
        "duration": 25,
        "ref_text": "",
    },
    {
        "id": "aizat_amdan",
        "name": "Aizat Amdan",
        "gender": "male",
        "search": "ytsearch1:Aizat Amdan interview temu bual berbicara",
        "start_sec": 15,
        "duration": 25,
        "ref_text": "",
    },
    {
        "id": "aliff_aziz",
        "name": "Aliff Aziz",
        "gender": "male",
        "search": "ytsearch1:Aliff Aziz interview berbicara Malaysia 2022",
        "start_sec": 15,
        "duration": 25,
        "ref_text": "",
    },
    {
        "id": "wany_hasrita",
        "name": "Wany Hasrita",
        "gender": "female",
        "search": "ytsearch1:Wany Hasrita interview berbicara 2023",
        "start_sec": 15,
        "duration": 25,
        "ref_text": "",
    },
    {
        "id": "naim_daniel",
        "name": "Naim Daniel",
        "gender": "male",
        "search": "ytsearch1:Naim Daniel interview berbicara Malaysia 2023",
        "start_sec": 15,
        "duration": 25,
        "ref_text": "",
    },
    {
        "id": "amy_search",
        "name": "Amy Search",
        "gender": "male",
        "search": "ytsearch1:Amy Search interview berbicara temu bual 2022",
        "start_sec": 20,
        "duration": 25,
        "ref_text": "",
    },
    {
        "id": "hafiz_suip",
        "name": "Hafiz Suip",
        "gender": "male",
        "search": "ytsearch1:Hafiz Suip interview berbicara Malaysia 2023",
        "start_sec": 15,
        "duration": 25,
        "ref_text": "",
    },
]


def _require_tools() -> None:
    missing = []
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        missing.append("yt-dlp  →  pip install yt-dlp")
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        missing.append("ffmpeg  →  sudo apt install ffmpeg")
    if missing:
        print("Missing dependencies:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)


def download_and_clip(artist: dict, out_dir: Path, verbose: bool = False) -> Path | None:
    """Download, trim, and resample to 24kHz mono WAV. Returns output path or None on failure."""
    import yt_dlp

    wav_out = out_dir / f"{artist['id']}.wav"
    if wav_out.exists():
        print(f"  [skip] {artist['id']}.wav already exists")
        return wav_out

    start = artist["start_sec"]
    duration = artist["duration"]

    # Download best audio, postprocess with ffmpeg to trimmed 24kHz mono WAV
    postprocessor_args = {
        "ffmpeg": [
            "-ss", str(start),
            "-t", str(duration),
            "-ar", "24000",
            "-ac", "1",
        ]
    }

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / f"{artist['id']}_raw.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "postprocessor_args": postprocessor_args,
        "quiet": not verbose,
        "no_warnings": not verbose,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([artist["search"]])
    except Exception as e:
        print(f"  [error] {artist['id']}: {e}")
        return None

    # yt-dlp writes <id>_raw.wav
    raw = out_dir / f"{artist['id']}_raw.wav"
    if raw.exists():
        raw.rename(wav_out)
    else:
        # find whatever it wrote
        candidates = list(out_dir.glob(f"{artist['id']}_raw.*"))
        if candidates:
            # re-encode with ffmpeg to ensure correct format
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(start), "-t", str(duration),
                 "-i", str(candidates[0]),
                 "-ar", "24000", "-ac", "1", str(wav_out)],
                capture_output=not verbose, check=True,
            )
            candidates[0].unlink()
        else:
            print(f"  [error] {artist['id']}: no output file found")
            return None

    print(f"  [ok]   {wav_out.name}  ({wav_out.stat().st_size // 1024} KB)")
    return wav_out


def write_placeholder_txt(artist: dict, out_dir: Path) -> None:
    txt = out_dir / f"{artist['id']}.txt"
    if txt.exists():
        return
    if artist["ref_text"]:
        txt.write_text(artist["ref_text"] + "\n")
    else:
        txt.write_text(
            f"# TODO: listen to {artist['id']}.wav and fill in the exact transcript\n"
            f"# Then run: scripts/scrape_voices.py --register-all\n"
        )
        print(f"  [todo] Fill in voice_samples/{artist['id']}.txt with the exact transcript")


def register_profile(artist: dict, out_dir: Path, api_base: str) -> None:
    import urllib.request

    wav = out_dir / f"{artist['id']}.wav"
    txt = out_dir / f"{artist['id']}.txt"
    if not wav.exists():
        print(f"  [skip] {artist['id']}: no wav")
        return
    if not txt.exists() or txt.read_text().startswith("#"):
        print(f"  [skip] {artist['id']}: ref_text not filled in")
        return

    import urllib.parse
    import http.client

    ref_text = txt.read_text().strip()
    boundary = "----FormBoundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="profile_id"\r\n\r\n'
        f"{artist['id']}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="ref_text"\r\n\r\n'
        f"{ref_text}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="ref_audio"; filename="ref_audio.wav"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + wav.read_bytes() + f"\r\n--{boundary}--\r\n".encode()

    parsed = urllib.parse.urlparse(api_base)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 8880)
    conn.request(
        "POST", "/v1/voices/profiles",
        body=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    resp = conn.getresponse()
    if resp.status in (200, 201):
        print(f"  [registered] {artist['id']}")
    elif resp.status == 409:
        print(f"  [exists]     {artist['id']}")
    else:
        print(f"  [failed]     {artist['id']}: HTTP {resp.status} {resp.read().decode()[:120]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="List all artists and exit")
    parser.add_argument("--artist", help="Only process this artist id")
    parser.add_argument("--out-dir", default="voice_samples", help="Output directory (default: voice_samples)")
    parser.add_argument("--register-all", action="store_true", help="Register completed profiles with running server")
    parser.add_argument("--api-base", default="http://localhost:8880", help="Server base URL")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.list:
        print(f"{'ID':<20} {'Name':<22} Gender")
        print("-" * 52)
        for a in ARTISTS:
            print(f"{a['id']:<20} {a['name']:<22} {a['gender']}")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    targets = [a for a in ARTISTS if not args.artist or a["id"] == args.artist]
    if not targets:
        print(f"Artist '{args.artist}' not found. Use --list to see valid ids.")
        sys.exit(1)

    if args.register_all:
        print(f"Registering {len(targets)} profiles at {args.api_base} ...")
        for a in targets:
            register_profile(a, out_dir, args.api_base)
        return

    _require_tools()

    print(f"Scraping {len(targets)} artist(s) → {out_dir}/")
    for a in targets:
        print(f"\n→ {a['name']} ({a['id']})")
        wav = download_and_clip(a, out_dir, verbose=args.verbose)
        if wav:
            write_placeholder_txt(a, out_dir)

    print("\nDone. Next steps:")
    print("  1. Listen to each clip and fill in the exact transcript in voice_samples/<id>.txt")
    print("  2. Run:  uv run python scripts/scrape_voices.py --register-all")
    print("     to push all completed profiles to the running server.")


if __name__ == "__main__":
    main()
