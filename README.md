# CPU transcript recovery

I built this command as the fallback for a meeting recorder when its usual transcription worker was unavailable. It runs faster-whisper on a CPU and turns a local recording into JSON segments with start times, end times and text. A downstream app can use those times to display subtitles, index a recording or jump to a spoken passage.

## Run it

Use Python 3.12 and an ordinary local filesystem. These commands create a Python environment and install the CPU inference and audio decoding dependencies.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python transcribe_cpu.py recording.wav transcript.json
```

The default model is `small.en` for English speech. Its first use downloads model weights. Inference and voice activity detection run locally. The output must be a new filename in an existing directory. The command retains the audio, and a failed or competing run cannot replace an existing transcript.

## Try it with synthetic speech

On macOS, run these commands from the cloned repository with the Python environment active. Use a fresh demo directory. The built-in speech tool generates a short WAV file.

```sh
mkdir demo
say --file-format=WAVE --data-format=LEI16@16000 -o demo/sample.wav \
  "The greenhouse sensor is ready. Check the temperature at noon."
python transcribe_cpu.py demo/sample.wav demo/sample.json --model tiny.en
cat demo/sample.json
```

On Linux with eSpeak NG installed, generate the same input with `espeak-ng -w demo/sample.wav "The greenhouse sensor is ready. Check the temperature at noon."` after creating the demo directory. The inference command is the same. Choose a fresh output name when repeating the example.

## Read the result

A transcript has the following shape. Timestamps are seconds from the start of the audio. Segment boundaries and wording depend on the model and recording.

```json
{
  "model": "faster-whisper/tiny.en",
  "language": "en",
  "segments": [
    {"index": 0, "start": 0.0, "end": 2.1, "text": "The greenhouse sensor is ready."}
  ]
}
```

The command skips blank text, numbers segments from zero and checks that timestamps are finite, nonnegative and in start-time order. Silence or an invalid timeline raises an error before the transcript is published.

## Work offline

Once the model is cached, add `--offline` to disable remote model lookup. For an air-gapped environment, preinstall the Python dependencies and copy a complete faster-whisper model directory to the machine. Then run the command with its local path.

```sh
python transcribe_cpu.py recording.wav offline.json --model /path/to/model --offline
```

When loading a model directory, the JSON uses `faster-whisper/local-model` to keep the filesystem path private. Output is assembled in a temporary file and published with a hard link. The destination filesystem must support local hard links.

## Verify

```sh
python -m unittest discover -v
```

The test suite decodes a generated WAV, exercises transcript normalization and checks safe output publication. CI runs the tests with the pinned dependencies. Tests require no model download.
