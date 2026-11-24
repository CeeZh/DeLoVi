"""Turn a video file into a DeLoVi memory: one caption per clip, followed by the clip's speech after `Speech:`.

    python -m video_qa.memory video.mp4              # writes cache/videos/<name>/{meta,clips}.json

The video is cut into clips of CLIP_SECONDS. For each clip, GPT-6 Luna captions one frame per second (the
captioner of the benchmarks). The audio is transcribed with the OpenAI Whisper API, and the transcript segments
that overlap a clip are appended to its caption. The memory is cached per video, so later questions on the same
video reuse it. Needs ffmpeg and ffprobe.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import random
import shutil
import subprocess
from pathlib import Path

from delovi.config import CACHE_DIR, PRICES
from delovi.data import Clip, Memory

# Hyper-parameters
CLIP_SECONDS = 10  # length of one memory entry
CAPTION_MODEL, CAPTION_REASONING = "gpt-6-luna", "none"
FRAME_MAX_SIZE = 512  # longer side of a frame, in pixels
WHISPER_MODEL = "whisper-1"
WHISPER_PRICE_PER_MIN = 0.006  # USD
AUDIO_CHUNK_SECONDS = 600  # audio per Whisper request (the API accepts files up to 25 MB)
MAX_SPEECH_SPAN = 15.0  # Whisper sometimes stretches a short line over a long silence
CONCURRENCY = 16  # caption requests in flight

CAPTION_PROMPT = """You are an expert video captioner.

You will receive a short video segment represented by ordered frames.
Write a caption describing the visual content of the segment.

Guidelines:
- Describe visible actions, people, objects, and environment.
- Keep the caption factual and neutral.
- Do not mention frames, timestamps, or that the input came from frames.
- Avoid speculation about emotions or intentions unless clearly visible.

Output only the final caption text."""


def clock(sec: float) -> str:
    ms = max(0, int(round(sec * 1000)))
    h, rest = divmod(ms, 3_600_000)
    m, rest = divmod(rest, 60_000)
    s, ms = divmod(rest, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}" + (f".{ms:03d}" if ms else "")


def probe(video: Path) -> tuple[float, bool]:
    """(duration in seconds, whether the video has an audio track)."""
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type", "-of", "json",
                          str(video)], capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    return float(info["format"]["duration"]), any(s.get("codec_type") == "audio" for s in info.get("streams", []))


def extract_frames(video: Path, out: Path) -> None:
    """One JPEG per second of video, out/000001.jpg showing second 0, scaled so the longer side is at most
    FRAME_MAX_SIZE."""
    out.mkdir(parents=True, exist_ok=True)
    scale = (f"scale='min({FRAME_MAX_SIZE},iw)':'min({FRAME_MAX_SIZE},ih)':force_original_aspect_ratio=decrease")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", f"fps=1,{scale}", "-q:v", "3",
                    str(out / "%06d.jpg")], check=True)


def caption_messages(start: float, end: float, frames: list[tuple[float, Path]]) -> list[dict]:
    content = [{"type": "input_text", "text": f"Segment window: {clock(start)} to {clock(end)}\n"
                                              "The following frames are ordered chronologically within the segment."}]
    for t, path in frames:
        content.append({"type": "input_text", "text": f"Frame timestamp: {clock(t)}"})
        content.append({"type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64.b64encode(path.read_bytes()).decode()}"})
    return [{"role": "system", "content": CAPTION_PROMPT}, {"role": "user", "content": content}]


async def caption_clips(client, spans: list[tuple[float, float]], frame_dir: Path, done: dict[int, str],
                        log: Path) -> float:
    """Caption every clip not in `done` (appended to `log` as it finishes); returns the cost in USD."""
    n_frames = len(list(frame_dir.glob("*.jpg")))
    if not n_frames:
        raise RuntimeError(f"ffmpeg extracted no frames into {frame_dir}")
    sem, cost = asyncio.Semaphore(CONCURRENCY), [0.0]
    price = PRICES[CAPTION_MODEL]

    async def one(i: int, start: float, end: float) -> None:
        secs = [s for s in range(int(math.floor(start)), int(math.ceil(end))) if start <= s < end] or [int(start)]
        frames = [(float(s), frame_dir / f"{min(s, n_frames - 1) + 1:06d}.jpg") for s in secs]
        async with sem:
            for attempt in range(8):
                try:
                    resp = await client.responses.create(model=CAPTION_MODEL, input=caption_messages(start, end, frames),
                                                         reasoning={"effort": CAPTION_REASONING})
                    break
                except Exception:  # noqa: BLE001 -- rate limits, timeouts, 5xx
                    if attempt == 7:
                        raise
                    await asyncio.sleep(min(60.0, 2 ** attempt * 2) * (0.5 + random.random()))
        done[i] = resp.output_text.strip()
        cost[0] += (resp.usage.input_tokens * price[0] + resp.usage.output_tokens * price[2]) / 1e6
        with log.open("a") as fh:
            fh.write(json.dumps({"index": i, "text": done[i]}, ensure_ascii=False) + "\n")
        if len(done) % 50 == 0:
            print(f"  {len(done)}/{len(spans)} clips captioned", flush=True)

    await asyncio.gather(*(one(i, s, e) for i, (s, e) in enumerate(spans) if i not in done))
    return cost[0]


async def transcribe(client, video: Path, duration: float, work: Path, translate: bool) -> list[tuple[float, float, str]]:
    """Speech segments (start, end, text) of the whole video, from the Whisper API in chunks."""
    segments = []
    for k in range(int(math.ceil(duration / AUDIO_CHUNK_SECONDS))):
        start = k * AUDIO_CHUNK_SECONDS
        audio = work / f"audio_{k:03d}.mp3"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(AUDIO_CHUNK_SECONDS), "-i", str(video),
                        "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", str(audio)], check=True)
        if not audio.exists() or audio.stat().st_size < 2000:  # no audio left in this chunk
            continue
        with audio.open("rb") as f:
            if translate:  # into English
                resp = await client.audio.translations.create(model=WHISPER_MODEL, file=f, response_format="verbose_json")
            else:
                resp = await client.audio.transcriptions.create(model=WHISPER_MODEL, file=f, response_format="verbose_json",
                                                                timestamp_granularities=["segment"])
        for s in resp.segments or []:
            text = " ".join(s.text.split())
            if text:
                segments.append((start + s.start, min(start + s.end, start + s.start + MAX_SPEECH_SPAN), text))
        audio.unlink()
    return segments


def describe(duration: float, clip_seconds: int, speech: bool) -> str:
    length = f"about {round(duration / 60)} minutes long" if duration >= 90 else f"about {round(duration)} seconds long"
    tail = ("what is said in the video follows `Speech:`" if speech
            else "the video has no speech, so the captions describe only what is seen")
    return f"a video {length}. The shortest segments are {clip_seconds}-second clips, and {tail}."


async def build_memory(video: Path, clip_seconds: int = CLIP_SECONDS, translate: bool = False,
                       description: str | None = None, dry_run: bool = False) -> Memory:
    """The memory of `video`, built once and cached under cache/videos/."""
    video = video.resolve()
    st = video.stat()
    key = hashlib.sha1(f"{video}|{st.st_size}|{st.st_mtime}|{clip_seconds}|{translate}|{dry_run}".encode()).hexdigest()[:8]
    out = CACHE_DIR / "videos" / f"{video.stem}_{key}"
    if not (out / "clips.json").exists():
        out.mkdir(parents=True, exist_ok=True)
        duration, has_audio = probe(video)
        n = max(1, math.ceil(duration / clip_seconds))
        spans = [(float(i * clip_seconds), min(float((i + 1) * clip_seconds), duration)) for i in range(n)]
        print(f"{video.name}: {duration / 60:.1f} min, {n} clips of {clip_seconds} s", flush=True)
        log = out / "captions.jsonl"
        done = {r["index"]: r["text"] for r in map(json.loads, log.read_text().splitlines())} if log.exists() else {}
        cost, speech = 0.0, []
        if dry_run:
            captions = {i: f"A fake caption of the clip at {clock(s)}." for i, (s, _) in enumerate(spans)}
        else:
            from openai import AsyncOpenAI

            client = AsyncOpenAI()
            if len(done) < n:
                extract_frames(video, out / "frames")
                cost += await caption_clips(client, spans, out / "frames", done, log)
                shutil.rmtree(out / "frames", ignore_errors=True)
            captions = done
            if has_audio:
                print(f"{video.name}: transcribing the audio with {WHISPER_MODEL}", flush=True)
                speech = await transcribe(client, video, duration, out, translate)
                cost += duration / 60 * WHISPER_PRICE_PER_MIN
        clips = []
        for i, (s, e) in enumerate(spans):
            said = " ".join(t for a, b, t in speech if b > s and a < e)
            clips.append({"start": s, "end": e, "text": captions[i] + (f"\nSpeech: {said}" if said else "")})
        meta = {"memory_id": f"video/{out.name}", "dataset": "video", "video": video.name, "n_clips": n,
                "start": 0.0, "end": duration, "timeline": {"type": "plain"}, "clip_seconds": clip_seconds,
                "description": description or describe(duration, clip_seconds, bool(speech)), "subject": None,
                "has_speech": bool(speech)}
        (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
        (out / "clips.json").write_text(json.dumps(clips, indent=1, ensure_ascii=False))
        print(f"{video.name}: memory written to {out} (API cost ${cost:.3f})", flush=True)
    meta = json.loads((out / "meta.json").read_text())
    if description:
        meta["description"] = description
    raw = json.loads((out / "clips.json").read_text())
    clips = [Clip(i, c["start"], c["end"], c["text"]) for i, c in enumerate(raw)]
    return Memory(meta["memory_id"], "video", meta["timeline"], meta["description"], None, clips, meta)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path)
    p.add_argument("--clip-seconds", type=int, default=CLIP_SECONDS)
    p.add_argument("--translate", action="store_true", help="transcribe the speech into English")
    p.add_argument("--dry-run", action="store_true", help="fake captions and no speech; no API calls")
    a = p.parse_args()
    m = asyncio.run(build_memory(a.video, a.clip_seconds, a.translate, dry_run=a.dry_run))
    print(f"{len(m.clips)} clips; first: {m.clips[0].text[:200]!r}")
