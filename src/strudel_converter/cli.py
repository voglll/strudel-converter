from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from .analyze import analyze_track
from .export import analysis_to_strudel
from .fourier_mode import fourier_to_strudel
from .stem_workflow import run_stem_workflow
from .loop_optimizer import run_loop

AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert techno audio into a Strudel draft.")
    parser.add_argument("input", help="Path to an audio file or a folder with audio files")
    parser.add_argument("-o", "--output", help="Optional output file or output folder.")
    parser.add_argument(
        "--mode",
        choices=("grid", "fourier", "stems", "loop"),
        default="grid",
        help="Conversion mode: 'grid' (beat/event), 'fourier' (sinusoidal), 'stems' (stem separation), or 'loop' (closed-loop beat optimizer).",
    )
    parser.add_argument(
        "--fourier-seconds",
        type=float,
        default=12.0,
        help="Seconds to analyze in Fourier mode (default: 12.0).",
    )
    parser.add_argument(
        "--fourier-partials",
        type=int,
        default=14,
        help="Number of sinusoidal partial tracks in Fourier mode (default: 14).",
    )
    parser.add_argument(
        "--fourier-bins",
        type=int,
        default=56,
        help="Top FFT bins kept per frame in Fourier mode (default: 56).",
    )
    parser.add_argument(
        "--fourier-max-frames",
        type=int,
        default=128,
        help="Max frame count emitted into Strudel patterns in Fourier mode (default: 128).",
    )
    parser.add_argument(
        "--fourier-tempo-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to Fourier mode cps (default: 1.0).",
    )
    parser.add_argument(
        "--no-fourier-wav",
        action="store_true",
        help="In Fourier mode, skip writing the reconstructed WAV file.",
    )
    parser.add_argument(
        "--stem-output-dir",
        help="Optional output directory for stem-first workflow artifacts.",
    )
    parser.add_argument(
        "--stem-iterations",
        type=int,
        default=4,
        help="Optimization iterations for the stem-first workflow (default: 4).",
    )
    parser.add_argument(
        "--loop-beats",
        type=int,
        default=64,
        help="Number of beats to analyze in closed-loop mode (default: 64).",
    )
    return parser


def _find_audio_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(
            candidate
            for candidate in path.iterdir()
            if candidate.is_file() and candidate.suffix.lower() in AUDIO_EXTENSIONS
        )
    raise FileNotFoundError(f"Input path does not exist: {path}")


def _default_output_path(input_file: Path) -> Path:
    return input_file.with_suffix(".txt")


def _resolve_output_paths(input_files: Iterable[Path], output_arg: str | None) -> list[Path]:
    input_files = list(input_files)
    if not input_files:
        raise FileNotFoundError("No audio files found.")

    if output_arg is None:
        return [_default_output_path(input_file) for input_file in input_files]

    output_path = Path(output_arg)
    if len(input_files) == 1 and output_path.suffix:
        return [output_path]

    if len(input_files) > 1 and output_path.suffix:
        raise ValueError("When converting multiple audio files, --output must be a folder.")

    output_path.mkdir(parents=True, exist_ok=True)
    return [output_path / _default_output_path(input_file).name for input_file in input_files]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    audio_files = _find_audio_files(input_path)
    output_paths = _resolve_output_paths(audio_files, args.output)

    for audio_file, output_path in zip(audio_files, output_paths, strict=True):
        if args.mode == "fourier":
            strudel_code = fourier_to_strudel(
                audio_file,
                output_txt_path=output_path,
                analyze_seconds=args.fourier_seconds,
                partial_count=args.fourier_partials,
                top_bins_per_frame=args.fourier_bins,
                max_frames=args.fourier_max_frames,
                tempo_scale=args.fourier_tempo_scale,
                write_reconstruction_wav=not args.no_fourier_wav,
            )
        elif args.mode == "stems":
            strudel_code = run_stem_workflow(
                audio_file,
                output_txt_path=output_path,
                output_dir=args.stem_output_dir,
                iterations=args.stem_iterations,
            )
        elif args.mode == "loop":
            strudel_code = run_loop(
                audio_file,
                output_txt_path=output_path,
                focus_beats=args.loop_beats,
            )
        else:
            analysis = analyze_track(audio_file)
            strudel_code = analysis_to_strudel(analysis)
        output_path.write_text(strudel_code, encoding="utf-8")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
