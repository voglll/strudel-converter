# Techno to Strudel Converter

First MVP for a rule-based audio-to-Strudel converter for techno and hardtechno.

## Goal

The tool analyses a track, estimates tempo and beat grid, extracts coarse drum activity, and emits a Strudel draft that is easy to inspect and edit.

Current scope

- single audio file input or a folder with audio files
- beat-stable techno and hardtechno
- synthetic Strudel output using built-in synths, not sample packs
- kick, hat, snare, and simple bass heuristics
- Strudel draft output as a `.txt` file next to the audio file by default

## Run

```bash
python -m strudel_converter.cli "Toter Schmetterling.mp3"
```

Experimental Fourier-only mode (keeps current mode unchanged):

```bash
python -m strudel_converter.cli "Toter Schmetterling.mp3" --mode fourier
```

Optional tuning:

```bash
python -m strudel_converter.cli "Toter Schmetterling.mp3" --mode fourier --fourier-seconds 16 --fourier-partials 18 --fourier-bins 72
```

In Fourier mode, the converter also writes two WAV files by default:
- `<name>_fourier_recon.wav` (sparse spectral reconstruction)
- `<name>_fourier_residual.wav` (residual difference)

If you give a folder instead of one file, the tool will convert every audio file in that folder and write a matching `.txt` file for each one.
