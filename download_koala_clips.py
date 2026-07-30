#!/usr/bin/env python3
"""Download timestamped video clips from the Koala-36M metadata."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Iterator


DATASET_REPO = "Koala-36M/Koala-36M-v1"
ALL_CSV_FILES = [f"Koala_36M_{index}.csv" for index in range(1, 11)]
SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class Clip:
    video_id: str
    url: str
    start: str
    end: str
    caption: str
    motion_score: float


@dataclass(frozen=True)
class DownloadResult:
    video_id: str
    status: str
    output: str
    error: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download only the timestamped sections listed in Koala-36M CSV files."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--csv",
        type=Path,
        nargs="+",
        help="One or more already-downloaded Koala CSV files.",
    )
    source.add_argument(
        "--hf-file",
        action="append",
        help=(
            "CSV shard to fetch from Hugging Face, e.g. Koala_36M_1.csv. "
            "Repeat to use multiple shards."
        ),
    )
    source.add_argument(
        "--hf-all",
        action="store_true",
        help="Fetch and process all 10 CSV shards (about 49 GB total).",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("koala_metadata"),
        help="Directory in which Hugging Face CSV files are stored.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("koala_clips"))
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum clips to process; 0 means all matching clips (default: 0).",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=0,
        help="Skip this many data rows before processing.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-height", type=int, default=720)
    parser.add_argument(
        "--min-clarity",
        type=float,
        default=0.9,
        help="Require clarity_score to be greater than this value (default: 0.9).",
    )
    parser.add_argument(
        "--min-motion",
        type=float,
        default=26.7,
        help="Require motion_score to be greater than this value (default: 26.7).",
    )
    parser.add_argument("--max-duration", type=float, default=None)
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Only count rows matching the score thresholds; do not download videos.",
    )
    authentication = parser.add_mutually_exclusive_group()
    authentication.add_argument(
        "--cookies-from-browser",
        help="Browser name passed to yt-dlp, e.g. chrome or firefox.",
    )
    authentication.add_argument(
        "--cookies",
        type=Path,
        help="Netscape-format cookies.txt exported from a logged-in browser.",
    )
    parser.add_argument(
        "--sleep-interval",
        type=float,
        default=1.0,
        help="Minimum delay before each download in seconds (default: 1).",
    )
    parser.add_argument(
        "--max-sleep-interval",
        type=float,
        default=5.0,
        help="Maximum randomized download delay in seconds (default: 5).",
    )
    parser.add_argument(
        "--no-exact-cuts",
        action="store_true",
        help="Faster cuts at nearby keyframes instead of exact boundaries.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def timestamp_to_seconds(timestamp: str) -> float:
    parts = timestamp.strip().split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"Unsupported timestamp: {timestamp!r}")
    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"Invalid timestamp: {timestamp!r}") from exc
    if len(values) == 2:
        minutes, seconds = values
        return minutes * 60 + seconds
    hours, minutes, seconds = values
    return hours * 3600 + minutes * 60 + seconds


def parse_timestamp_range(value: str) -> tuple[str, str]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid timestamp field: {value!r}") from exc
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 2:
        raise ValueError(f"Expected [start, end], got: {value!r}")
    start, end = str(parsed[0]).strip(), str(parsed[1]).strip()
    if timestamp_to_seconds(end) <= timestamp_to_seconds(start):
        raise ValueError(f"End must be after start: {value!r}")
    return start, end


def iter_clips(
    csv_paths: list[Path],
    start_row: int,
    limit: int,
    min_clarity: float | None,
    min_motion: float,
    max_duration: float | None,
    completed_stems: set[str],
) -> Iterator[Clip]:
    visited_rows = 0
    yielded_clips = 0
    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            required_columns = {"motion_score", "timestamp", "url", "videoID"}
            missing_columns = required_columns.difference(reader.fieldnames or [])
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise ValueError(f"{csv_path} is missing columns: {missing}")

            for row in reader:
                if visited_rows < start_row:
                    visited_rows += 1
                    continue
                visited_rows += 1

                if min_clarity is not None:
                    try:
                        if float(row.get("clarity_score", 0)) <= min_clarity:
                            continue
                    except (TypeError, ValueError):
                        continue

                try:
                    motion_score = float(row["motion_score"])
                except (TypeError, ValueError):
                    continue
                if motion_score <= min_motion:
                    continue

                try:
                    start, end = parse_timestamp_range(row["timestamp"])
                except ValueError as exc:
                    print(
                        f"Skipping row {visited_rows}: {exc}",
                        file=sys.stderr,
                    )
                    continue
                duration = timestamp_to_seconds(end) - timestamp_to_seconds(start)
                if max_duration is not None and duration > max_duration:
                    continue
                if safe_stem(row["videoID"]) in completed_stems:
                    continue

                yield Clip(
                    video_id=row["videoID"],
                    url=row["url"],
                    start=start,
                    end=end,
                    caption=row.get("caption", ""),
                    motion_score=motion_score,
                )
                yielded_clips += 1
                if limit > 0 and yielded_clips >= limit:
                    return


def count_matching_rows(
    csv_paths: list[Path],
    min_clarity: float | None,
    min_motion: float,
) -> int:
    total_count = 0
    for csv_path in csv_paths:
        shard_count = 0
        with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            required_columns = {"clarity_score", "motion_score"}
            missing_columns = required_columns.difference(reader.fieldnames or [])
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise ValueError(f"{csv_path} is missing columns: {missing}")

            for row in reader:
                try:
                    clarity_score = float(row["clarity_score"])
                    motion_score = float(row["motion_score"])
                except (TypeError, ValueError):
                    continue
                if min_clarity is not None and clarity_score <= min_clarity:
                    continue
                if motion_score > min_motion:
                    shard_count += 1

        total_count += shard_count
        print(f"{csv_path.name}: {shard_count:,} matching rows")

    print(
        "Total: "
        f"{total_count:,} rows with clarity_score > {min_clarity} "
        f"and motion_score > {min_motion}"
    )
    return total_count


def safe_stem(video_id: str) -> str:
    stem = SAFE_FILENAME_PATTERN.sub("_", video_id).strip("._")
    return stem or "clip"


def find_completed_stems(output_dir: Path) -> set[str]:
    completed_stems = set()
    for output_path in output_dir.glob("*.mp4"):
        try:
            if output_path.stat().st_size > 0:
                completed_stems.add(output_path.stem)
        except OSError:
            continue
    return completed_stems


def download_clip(
    clip: Clip,
    output_dir: Path,
    max_height: int,
    cookies_from_browser: str | None,
    cookies: Path | None,
    sleep_interval: float,
    max_sleep_interval: float,
    exact_cuts: bool,
    dry_run: bool,
) -> DownloadResult:
    output_path = output_dir / f"{safe_stem(clip.video_id)}.mp4"
    if output_path.exists() and output_path.stat().st_size > 0:
        return DownloadResult(clip.video_id, "skipped", str(output_path))

    output_template = output_dir / f"{safe_stem(clip.video_id)}.%(ext)s"
    command = [
        "yt-dlp",
        "--no-playlist",
        "--retries",
        "5",
        "--fragment-retries",
        "5",
        "--sleep-requests",
        "1",
        "--sleep-interval",
        str(sleep_interval),
        "--max-sleep-interval",
        str(max_sleep_interval),
        "--download-sections",
        f"*{clip.start}-{clip.end}",
        "--format",
        (
            f"bestvideo[height<={max_height}]+bestaudio/"
            f"best[height<={max_height}]/best"
        ),
        "--merge-output-format",
        "mp4",
        "--remux-video",
        "mp4",
        "--output",
        str(output_template),
    ]
    if exact_cuts:
        command.append("--force-keyframes-at-cuts")
    if cookies_from_browser:
        command.extend(["--cookies-from-browser", cookies_from_browser])
    if cookies:
        command.extend(["--cookies", str(cookies)])
    command.append(clip.url)

    if dry_run:
        print(" ".join(command))
        return DownloadResult(clip.video_id, "dry-run", str(output_path))

    process = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        error = process.stderr.strip().splitlines()
        return DownloadResult(
            clip.video_id,
            "failed",
            str(output_path),
            error[-1] if error else f"yt-dlp exited with {process.returncode}",
        )
    return DownloadResult(clip.video_id, "downloaded", str(output_path))


def fetch_hf_files(filenames: list[str], metadata_dir: Path) -> list[Path]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Install Hugging Face Hub first: pip install huggingface_hub"
        ) from exc

    metadata_dir = metadata_dir.expanduser().resolve()
    metadata_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename in filenames:
        print(f"Fetching {filename} from {DATASET_REPO} (~4.9 GB per shard)...")
        path = hf_hub_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            filename=filename,
            local_dir=metadata_dir,
        )
        paths.append(Path(path))
    return paths


def validate_environment(dry_run: bool, count_only: bool) -> None:
    if dry_run or count_only:
        return
    missing = [
        executable
        for executable in ("yt-dlp", "ffmpeg")
        if shutil.which(executable) is None
    ]
    if missing:
        raise RuntimeError(
            f"Missing executables: {', '.join(missing)}. "
            "Install yt-dlp and ffmpeg before running."
        )


def batched(clips: Iterator[Clip], batch_size: int) -> Iterator[list[Clip]]:
    while batch := list(islice(clips, batch_size)):
        yield batch


def main() -> int:
    args = build_parser().parse_args()
    if args.limit < 0 or args.start_row < 0:
        raise ValueError("--limit and --start-row must be non-negative")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.sleep_interval < 0:
        raise ValueError("--sleep-interval must be non-negative")
    if args.max_sleep_interval < args.sleep_interval:
        raise ValueError(
            "--max-sleep-interval must be greater than or equal to "
            "--sleep-interval"
        )
    if args.cookies:
        args.cookies = args.cookies.expanduser().resolve()
        if not args.cookies.is_file():
            raise FileNotFoundError(f"Cookie file not found: {args.cookies}")

    validate_environment(args.dry_run, args.count_only)
    if args.csv:
        csv_paths = [path.expanduser().resolve() for path in args.csv]
    else:
        filenames = ALL_CSV_FILES if args.hf_all else args.hf_file
        csv_paths = fetch_hf_files(filenames, args.metadata_dir)
    missing_paths = [str(path) for path in csv_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"CSV files not found: {', '.join(missing_paths)}")

    if args.count_only:
        count_matching_rows(
            csv_paths=csv_paths,
            min_clarity=args.min_clarity,
            min_motion=args.min_motion,
        )
        return 0

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    completed_stems = find_completed_stems(output_dir)
    print(
        f"Resume: found {len(completed_stems):,} completed MP4 files; "
        "they will be skipped."
    )
    clips = iter_clips(
        csv_paths=csv_paths,
        start_row=args.start_row,
        limit=args.limit,
        min_clarity=args.min_clarity,
        min_motion=args.min_motion,
        max_duration=args.max_duration,
        completed_stems=completed_stems,
    )

    manifest_path = output_dir / "manifest.jsonl"
    counts = {"downloaded": 0, "skipped": 0, "failed": 0, "dry-run": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        with manifest_path.open("a", encoding="utf-8") as manifest:
            for clip_batch in batched(clips, args.workers * 2):
                futures = {
                    executor.submit(
                        download_clip,
                        clip,
                        output_dir,
                        args.max_height,
                        args.cookies_from_browser,
                        args.cookies,
                        args.sleep_interval,
                        args.max_sleep_interval,
                        not args.no_exact_cuts,
                        args.dry_run,
                    ): clip
                    for clip in clip_batch
                }
                for future in as_completed(futures):
                    clip = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # Keep other downloads running.
                        result = DownloadResult(
                            clip.video_id,
                            "failed",
                            "",
                            str(exc),
                        )
                    counts[result.status] += 1
                    record = {**asdict(clip), **asdict(result)}
                    manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                    manifest.flush()
                    message = f"[{result.status}] {result.video_id}"
                    if result.error:
                        message += f": {result.error}"
                    print(message)

    print("Summary: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
