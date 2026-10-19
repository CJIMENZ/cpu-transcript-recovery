"""Portable Whisper transcription. No network inference API or Mac required."""
import argparse
import json
import math
from pathlib import Path


def normalize_segments(source):
    result = []
    for segment in source:
        text = segment.text.strip()
        if not text:
            continue
        start, end = float(segment.start), float(segment.end)
        if (not math.isfinite(start) or not math.isfinite(end)
                or start < 0 or end < start
                or (result and start < result[-1]["start"])):
            raise ValueError("Invalid transcript timeline; audio must be retained")
        result.append(dict(index=len(result), start=start, end=end, text=text))
    if not result:
        raise ValueError("No speech detected; audio must be retained")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("output")
    parser.add_argument("--model", default="small.en")
    args = parser.parse_args()
    from faster_whisper import WhisperModel
    model = WhisperModel(args.model, device="cpu", compute_type="int8", cpu_threads=4)
    source, info = model.transcribe(args.audio, beam_size=5, vad_filter=True,
                                    condition_on_previous_text=False)
    result = {"model": "faster-whisper/" + args.model,
              "language": info.language, "segments": normalize_segments(source)}
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
