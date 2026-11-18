# CPU transcript recovery

Transcribe a local audio file on a CPU when the usual transcription worker is unavailable. This command-line component grew out of a meeting recorder. It uses faster-whisper with voice activity detection and writes the detected language and timestamped text segments to JSON. The source audio stays on disk.

## Run it

With Python 3.12 installed, run these commands from this directory.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python -m unittest -v test_transcribe_cpu
python transcribe_cpu.py recording.wav transcript.json
```

Replace `recording.wav` with your audio file. Use a new output path to retain an earlier transcript. The default model is `small.en`. Its first use downloads model weights. Pass `--model /path/to/model` to use an existing local faster-whisper model directory.

Blank segments are skipped. The command raises an error if the transcript is empty or segment timestamps are nonfinite, negative, reversed, or out of start-time order. The tests use synthetic inputs and decode a generated WAV file.
