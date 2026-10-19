import unittest
import importlib.util
import tempfile
import wave
from types import SimpleNamespace
from transcribe_cpu import normalize_segments


class CpuTranscriptTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("faster_whisper"), "CPU dependencies not installed")
    def test_decoder_accepts_a_real_wav(self):
        from faster_whisper.audio import decode_audio
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio:
            with wave.open(audio.name, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\x00\x00" * 16000)
            self.assertEqual(len(decode_audio(audio.name)), 16000)

    def test_ignores_empty_segments_and_reindexes(self):
        source = [SimpleNamespace(start=0, end=1, text="  "),
                  SimpleNamespace(start=2, end=3, text=" Actual speech. ")]
        self.assertEqual(normalize_segments(source),
                         [dict(index=0, start=2.0, end=3.0, text="Actual speech.")])

    def test_rejects_silence_and_invalid_timestamps(self):
        for source in [[], [SimpleNamespace(start=float("nan"), end=1, text="x")],
                       [SimpleNamespace(start=2, end=1, text="x")]]:
            with self.assertRaises(ValueError):
                normalize_segments(source)


if __name__ == "__main__":
    unittest.main()
