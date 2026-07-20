from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Event:
    time: float
    bar: int
    step_in_bar: int
    velocity: float = 1.0


@dataclass(slots=True)
class PatternTrack:
    name: str
    events: list[Event] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Section:
    start_time: float
    end_time: float
    start_bar: int
    end_bar: int
    label: str
    confidence: float


@dataclass(slots=True)
class BeatGrid:
    tempo: float
    beat_times: list[float]
    downbeat_times: list[float]
    bars: int
    beats_per_bar: int = 4
    steps_per_bar: int = 64
    loop_bars: int = 2


@dataclass(slots=True)
class TrackAnalysis:
    path: str
    sample_rate: int
    duration: float
    grid: BeatGrid
    sections: list[Section]
    tracks: dict[str, PatternTrack]
    notes: list[str] = field(default_factory=list)
