"""Portable Whisper transcription. No network inference API or Mac required."""
import argparse
import json
import math
import os
from tempfile import NamedTemporaryFile
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


def write_transcript(destination, result):
    """Publish a complete transcript only if the destination is unused."""
    destination = Path(destination)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent) as temporary:
        json.dump(result, temporary, ensure_ascii=False, indent=2, allow_nan=False)
        temporary.write("\n")
        temporary.flush()
        # A hard link is atomic and fails if another run claimed the output.
        os.link(temporary.name, destination)


def main():
    parser = argparse.ArgumentParser(description="Transcribe local audio on a CPU.")
    parser.add_argument("audio", type=Path, help="Input audio file")
    parser.add_argument("output", type=Path, help="New JSON file in an existing directory")
    parser.add_argument("--model", default="small.en", help="Model name or local model directory")
    parser.add_argument("--offline", action="store_true", help="Use only locally available model files")
    args = parser.parse_args()

    if not args.audio.is_file():
        parser.error("input must be an existing audio file")
    if os.path.lexists(args.output):
        parser.error("output path already exists; choose a new filename")
    if not args.output.parent.is_dir():
        parser.error("output directory does not exist")

    from faster_whisper import WhisperModel
    model = WhisperModel(args.model, device="cpu", compute_type="int8", cpu_threads=4,
                         local_files_only=args.offline)
    source, info = model.transcribe(str(args.audio), beam_size=5, vad_filter=True,
                                    condition_on_previous_text=False)
    model_label = "local-model" if Path(args.model).is_dir() else args.model
    result = {"model": "faster-whisper/" + model_label,
              "language": info.language, "segments": normalize_segments(source)}
    try:
        write_transcript(args.output, result)
    except FileExistsError:
        parser.error("output path already exists; choose a new filename")


if __name__ == "__main__":
    main()
