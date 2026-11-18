import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from transcribe_cpu import write_transcript


class TranscriptOutputTests(unittest.TestCase):
    def test_writes_complete_unicode_json(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "transcript.json"
            result = {"language": "es", "segments": [{"text": "¡Hola!"}]}
            write_transcript(output, result)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), result)

    def test_existing_file_survives_competing_write(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output.json"
            write_transcript(output, {"run": 1})
            with self.assertRaises(FileExistsError):
                write_transcript(output, {"run": 2})
            self.assertEqual(json.loads(output.read_text()), {"run": 1})
            self.assertEqual(list(Path(directory).iterdir()), [output])

    def test_invalid_json_leaves_no_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "transcript.json"
            with self.assertRaises(ValueError):
                write_transcript(output, {"start": float("nan")})
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_cli_rejects_input_as_output_before_loading_model(self):
        script = Path(__file__).with_name("transcribe_cpu.py")
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "original.wav"
            audio.write_bytes(b"source audio sentinel")
            proc = subprocess.run(
                [sys.executable, str(script), str(audio), str(audio), "--offline"],
                capture_output=True, text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("output path already exists", proc.stderr)
            self.assertEqual(audio.read_bytes(), b"source audio sentinel")
