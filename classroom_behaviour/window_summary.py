"""Summarize confirmed gestures in consecutive source-time windows.

Durations use the union of event intervals, restricted to observable frame
intervals. Completed head cycles use their motion span rather than the single
completion-frame display. This module uses only the Python standard library and
can summarize existing analysis outputs without rerunning recognition models.
"""

import argparse
from bisect import bisect_right
import csv
import json
import math
from pathlib import Path


COLUMNS = ("window_index", "window_start_s", "window_end_s_exclusive", "window_duration_s",
           "behaviour", "active_seconds", "inactive_seconds", "observed_seconds", "unknown_seconds",
           "active_percent_observed", "event_count", "count_basis", "events_overlapping")
EPSILON = 1e-9


def _positive(value, name):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def merge_intervals(intervals):
    """Return disjoint intervals; adjacent positive categories form one episode."""
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + EPSILON:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _intersect(intervals, observed):
    ends = [end for _, end in observed]
    result = []
    for start, end in intervals:
        i = bisect_right(ends, start)
        while i < len(observed) and observed[i][0] < end:
            a, b = max(start, observed[i][0]), min(end, observed[i][1])
            if b > a:
                result.append((a, b))
            i += 1
    return result


def _window_parts(start, end, window_seconds, duration_s, count):
    first = max(0, int(math.floor(start / window_seconds)))
    last = min(count - 1, int(math.ceil(end / window_seconds)) - 1)
    for index in range(first, last + 1):
        a = max(start, index * window_seconds)
        b = min(end, (index + 1) * window_seconds, duration_s)
        if b > a:
            yield index, b - a


def write_window_summary(frames_path, events_path, output_path, fps, duration_s,
                         window_seconds=30.0, expected_frames=None):
    """Write one row per behaviour/window, with seconds, coverage and counts."""
    for value, name in ((fps, "fps"), (duration_s, "duration_s"), (window_seconds, "window_seconds")):
        _positive(value, name)
    frames_path, events_path, output_path = map(Path, (frames_path, events_path, output_path))
    if output_path.resolve() in (frames_path.resolve(), events_path.resolve()):
        raise ValueError("Window output must not replace frame or event evidence")
    window_count = max(1, math.ceil((duration_s - EPSILON) / window_seconds))

    with frames_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        names = reader.fieldnames or []
        labels = [name[4:] for name in names if name.startswith("raw_") and name[4:] in names]
        if not labels or len(set(names)) != len(names) or not {"frame_index", "timestamp_s"}.issubset(names):
            raise ValueError("Invalid frame CSV schema: expected frame/time and raw/confirmed behaviour columns")
        observed = {label: [] for label in labels}
        open_start = dict.fromkeys(labels)
        frame_count = 0
        for row in reader:
            t = float(row["timestamp_s"])
            if int(row["frame_index"]) != frame_count or not math.isfinite(t) or abs(t - frame_count / fps) > 1e-5:
                raise ValueError(f"Frame CSV is not a consecutive source-time sequence at row {frame_count + 1}")
            start = min(frame_count / fps, duration_s)
            end = min((frame_count + 1) / fps, duration_s)
            for label in labels:
                state = row[label]
                if state not in ("", "0", "1"):
                    raise ValueError(f"Invalid confirmed state for {label}: {state!r}")
                if end <= start:
                    continue
                if state != "" and open_start[label] is None:
                    open_start[label] = start
                elif state == "" and open_start[label] is not None:
                    observed[label].append((open_start[label], start))
                    open_start[label] = None
            frame_count += 1
        if (expected_frames is not None and frame_count != expected_frames) or frame_count / fps < duration_s - 1e-6:
            raise ValueError("Frame CSV does not cover the requested summary duration/frame count")
        for label in labels:
            if open_start[label] is not None:
                observed[label].append((open_start[label], duration_s))

    native_events = {label: [] for label in labels}
    cycle_labels = {label for label in labels if label in ("head_nodding", "head_shaking")}
    kinds = {}
    with events_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"behaviour", "start_s", "end_s_exclusive"}.issubset(reader.fieldnames or []):
            raise ValueError("Invalid event CSV schema")
        for row in reader:
            label = row["behaviour"]
            if label not in native_events:
                raise ValueError(f"Event label is absent from the frame CSV: {label}")
            start, end = float(row["start_s"]), float(row["end_s_exclusive"])
            if not all(math.isfinite(v) for v in (start, end)) or start < 0 or end <= start or end > frame_count / fps + 1e-5:
                raise ValueError(f"Invalid event timing for {label}")
            is_cycle = row.get("timing_type") == "completed_cycle"
            if label in kinds and kinds[label] != is_cycle:
                raise ValueError(f"Mixed cycle/posture event types for {label}")
            kinds[label] = is_cycle
            completion = None
            if is_cycle:
                cycle_labels.add(label)
                completion = float(row["completion_s"])
                if not math.isfinite(completion) or completion < start:
                    raise ValueError(f"Invalid completion time for {label}")
                # A longer imported sequence must not contribute uncompleted
                # cycles from beyond the selected source prefix.
                if completion >= duration_s:
                    continue
                if abs(completion - end) > 1e-5:
                    raise ValueError(f"Invalid completion time for {label}")
            a, b = start, min(end, duration_s)
            if b > a:
                native_events[label].append((a, b, completion))

    stats = {label: [{"observed": 0.0, "active": 0.0, "count": 0, "overlap": 0}
                     for _ in range(window_count)] for label in labels}
    for label in labels:
        intervals = observed[label]
        for start, end in intervals:
            for index, amount in _window_parts(start, end, window_seconds, duration_s, window_count):
                stats[label][index]["observed"] += amount
        events = native_events[label]
        active = merge_intervals(_intersect(merge_intervals([(a, b) for a, b, _ in events]), intervals))
        for start, end in active:
            for index, amount in _window_parts(start, end, window_seconds, duration_s, window_count):
                stats[label][index]["active"] += amount
        if label in cycle_labels:
            # Count individual completed cycles, even for adjacent motion spans.
            for start, end, completion in sorted(set(events)):
                parts = _intersect([(start, end)], intervals)
                if not parts:
                    continue
                index = min(window_count - 1, int(completion // window_seconds))
                stats[label][index]["count"] += 1
                overlapping = {i for a, b in parts for i, _ in _window_parts(a, b, window_seconds, duration_s, window_count)}
                for index in overlapping:
                    stats[label][index]["overlap"] += 1
        else:
            for start, end in active:
                index = min(window_count - 1, int(start // window_seconds))
                stats[label][index]["count"] += 1
                for index, _ in _window_parts(start, end, window_seconds, duration_s, window_count):
                    stats[label][index]["overlap"] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for index in range(window_count):
            start, end = index * window_seconds, min((index + 1) * window_seconds, duration_s)
            length = end - start
            for label in labels:
                values = stats[label][index]
                known = min(length, max(0.0, values["observed"]))
                active = min(known, max(0.0, values["active"]))
                writer.writerow({"window_index": index + 1, "window_start_s": f"{start:.6f}",
                    "window_end_s_exclusive": f"{end:.6f}", "window_duration_s": f"{length:.6f}",
                    "behaviour": label, "active_seconds": f"{active:.6f}" if known else "",
                    "inactive_seconds": f"{known - active:.6f}" if known else "",
                    "observed_seconds": f"{known:.6f}", "unknown_seconds": f"{length - known:.6f}",
                    "active_percent_observed": f"{100 * active / known:.6f}" if known else "",
                    "event_count": values["count"] if known else "",
                    "count_basis": "cycle_completion" if label in cycle_labels else "episode_start",
                    "events_overlapping": values["overlap"] if known else ""})
    return {"window_seconds": window_seconds, "duration_s": duration_s, "window_count": window_count,
            "behaviour_count": len(labels), "row_count": window_count * len(labels),
            "duration_basis": "quality-masked union of confirmed event intervals",
            "time_basis": "source_video_seconds", "unknown_policy": "blank activity when wholly unobserved"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True, help="An existing *_metadata.json")
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--duration-seconds", type=float, help="Optional source-prefix duration, e.g. 120 for four exact windows")
    parser.add_argument("--output", type=Path, help="Defaults to *_windows.csv beside the frame CSV")
    args = parser.parse_args(argv)
    metadata = json.loads(args.metadata.read_text())
    frames = Path(metadata["outputs"]["frames"])
    events = Path(metadata["outputs"]["events"])
    output = args.output or frames.with_name(frames.stem.removesuffix("_frames") + "_windows.csv")
    if output.resolve() == args.metadata.resolve():
        parser.error("Window output must not replace metadata")
    duration = args.duration_seconds if args.duration_seconds is not None else metadata.get("window_summary", {}).get("duration_s", metadata["duration_s"])
    if duration > metadata["duration_s"] + 1e-6:
        parser.error("Summary duration exceeds the analysed source")
    report = write_window_summary(frames, events, output, metadata["fps"], duration,
                                  args.window_seconds, metadata["processed_frames"])
    print(f"Saved {report['window_count']} windows: {output}")
    return report


if __name__ == "__main__":
    main()
