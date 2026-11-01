#!/usr/bin/env python3
"""Run a SeedVR2 API workflow in independently recoverable video sections."""

import argparse
import ast
import concurrent.futures
import contextlib
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
from fractions import Fraction

SECTION_SECONDS = 20
WARMUP_FRAMES = 16
FUTURE_FRAMES = 8
START_SECTION = 0
END_SECTION = None  # Exclusive; None means the end of the video.
WORKER_URLS = []  # ComfyUI proxy URLs for this machine.
POLL_SECONDS = 10
HTTP_TIMEOUT_SECONDS = 660


def run(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def probe(path):
    data = json.loads(run(
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,profile,codec_tag_string,nb_read_frames,avg_frame_rate,r_frame_rate,"
        "width,height,pix_fmt,sample_aspect_ratio,color_range,color_space,color_transfer,color_primaries",
        "-of", "json", str(path),
    ))
    streams = data.get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Cannot read a video stream from {path}")
    stream = streams[0]
    frames = int(stream["nb_read_frames"])
    fps = Fraction(stream["avg_frame_rate"])
    if frames <= 0 or fps <= 0 or fps != Fraction(stream["r_frame_rate"]):
        raise ValueError(f"Expected a nonempty constant-frame-rate video: {path}")
    return stream, frames, fps


def sections(total, fps):
    size = round(SECTION_SECONDS * fps)
    if size < 1:
        raise ValueError("Section duration produces no frames")
    return [
        {"index": i, "start": start, "end": min(total, start + size),
         "load_start": max(0, start - WARMUP_FRAMES),
         "load_end": min(total, start + size + FUTURE_FRAMES)}
        for i, start in enumerate(range(0, total, size))
    ]


def load_prompt(path):
    if path.suffix != ".py":
        data = json.loads(path.read_text())
        return data.get("prompt", data)
    # The existing submitter constructs its prompt from these four values.
    names = {"CHUNK_SECONDS": 1.0, "FRAME_CAP": 1, "TILE": 1024, "OVL": 64}
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "prompt" for t in node.targets
        ):
            class Substitute(ast.NodeTransformer):
                def visit_Name(self, name):
                    if name.id not in names:
                        raise ValueError(f"Unsupported prompt variable: {name.id}")
                    return ast.copy_location(ast.Constant(names[name.id]), name)
            return ast.literal_eval(Substitute().visit(node.value))
    raise ValueError("No prompt assignment found in the submitter")


def node_id(prompt, kind):
    matches = [key for key, value in prompt.items() if value.get("class_type") == kind]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {kind} node, found {len(matches)}")
    return matches[0]


def section_prompt(template, section, prefix, source):
    prompt = copy.deepcopy(template)
    input_node = prompt[node_id(prompt, "SeedVR2LongVideoInput")]["inputs"]
    input_node["video"] = source.name
    loader = prompt[node_id(prompt, "VHS_LoadVideoPath")]["inputs"]
    loader["skip_first_frames"] = section["load_start"]
    loader["frame_load_cap"] = section["load_end"] - section["load_start"]
    if loader.get("force_rate", 0) or loader.get("select_every_nth", 1) != 1:
        raise ValueError("Frame rate conversion and frame selection are not supported")
    combine = prompt[node_id(prompt, "VHS_VideoCombine")]["inputs"]
    if combine.get("format") != "video/ProRes":
        raise ValueError("The workflow must use ProRes output")
    combine["filename_prefix"] = prefix
    combine["save_output"] = True
    combine.pop("audio", None)
    for key in list(prompt):
        if prompt[key].get("class_type") == "SeedVR2RemuxOriginalAudio":
            del prompt[key]
    return prompt


def request(base, path, data=None):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(base.rstrip("/") + path, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as response:
        return json.load(response)


def has_prefix(prompt, prefix):
    return any(node.get("class_type") == "VHS_VideoCombine"
               and node.get("inputs", {}).get("filename_prefix") == prefix
               for node in prompt.values())


def active_prefix(queue, prefix):
    return any(has_prefix(item[2], prefix)
               for group in ("queue_running", "queue_pending")
               for item in queue.get(group, []))


def matching_history(history, prefix):
    return [item for item in history.values()
            if len(item.get("prompt", [])) > 2
            and has_prefix(item["prompt"][2], prefix)]


def await_section(base, prefix):
    missing_since = None
    while True:
        queue = request(base, "/queue")
        history = matching_history(request(base, "/history"), prefix)
        for item in history:
            status = item.get("status", {}).get("status_str")
            if status == "error":
                raise RuntimeError(f"ComfyUI section failed on {base}: {item['status']}")
            if status != "success":
                continue
            for output in item.get("outputs", {}).values():
                for video in output.get("gifs", []):
                    if video.get("format") == "video/ProRes" and video.get("fullpath"):
                        return Path(video["fullpath"])
        if active_prefix(queue, prefix):
            missing_since = None
        else:
            missing_since = missing_since or time.monotonic()
            if time.monotonic() - missing_since > 120:
                raise RuntimeError(f"Section disappeared from {base}; leaving it unfinished")
        time.sleep(POLL_SECONDS)


def validate_video(path, frames, fps, codec=None):
    stream, actual, actual_fps = probe(path)
    if actual != frames or actual_fps != fps or (codec and stream["codec_name"] != codec):
        raise ValueError(f"Unexpected frame count, rate or codec in {path}")
    run("ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-map", "0:v:0",
        "-f", "null", "-")
    return stream


def marker_path(folder, section):
    return folder / f"section-{section['index']:06d}.json"


def completed(folder, section, fps):
    marker = marker_path(folder, section)
    if not marker.exists():
        return False
    data = json.loads(marker.read_text())
    if data["section"] != section:
        raise ValueError(f"Section metadata mismatch: {marker}")
    path = (folder / data["video"]).resolve()
    if not path.is_relative_to(folder.resolve()):
        raise ValueError(f"Section video is outside the checkpoint directory: {path}")
    try:
        validate_video(path, section["load_end"] - section["load_start"], fps, "prores")
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Section {section['index']} needs to be regenerated: {exc}", file=sys.stderr)
        return False
    return True


def write_json(path, data):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        json.dump(data, f, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
        temp = f.name
    os.replace(temp, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def job_id(source, template, total, fps):
    digest = hashlib.sha256()
    with source.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(block)
    prompt = copy.deepcopy(template)
    prompt[node_id(prompt, "SeedVR2LongVideoInput")]["inputs"]["video"] = "<source>"
    identity = {"source_sha256": digest.hexdigest(), "frames": total, "fps": str(fps),
                "prompt": prompt, "section_seconds": SECTION_SECONDS,
                "warmup": WARMUP_FRAMES, "future": FUTURE_FRAMES}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]


def process_section(base, template, section, folder, prefix_root, source, fps):
    prefix = f"{prefix_root}/section-{section['index']:06d}-{uuid.uuid4().hex}"
    prompt = section_prompt(template, section, prefix, source)
    response = request(base, "/prompt", {
        "prompt": prompt, "client_id": str(uuid.uuid4()),
        "extra_data": {"extra_pnginfo": {"workflow": {"extra": {"VHS_MetadataImage": False}}}},
    })
    if not response.get("prompt_id") or response.get("node_errors"):
        raise RuntimeError(f"ComfyUI rejected section: {response}")
    video = await_section(base, prefix).resolve()
    if not video.is_relative_to(folder.resolve()):
        raise ValueError(f"ComfyUI output is outside the checkpoint directory: {video}")
    validate_video(video, section["load_end"] - section["load_start"], fps, "prores")
    with video.open("rb") as f:
        os.fsync(f.fileno())
    write_json(marker_path(folder, section),
               {"section": section, "video": str(video.relative_to(folder.resolve()))})
    print(f"Completed section {section['index']}", flush=True)


def claim_section(folder, section, fps):
    lock = (folder / f"section-{section['index']:06d}.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return None
    try:
        if completed(folder, section, fps):
            lock.close()
            return None
    except Exception:
        lock.close()
        raise
    return lock


def process_claimed_section(lock, *args):
    try:
        process_section(*args)
    finally:
        lock.close()


def assemble(folder, parts, source, fps, final):
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite {final}")
    with tempfile.TemporaryDirectory(prefix="assemble-", dir=final.parent) as temp:
        temp = Path(temp)
        files = []
        reference = None
        for section in parts:
            data = json.loads(marker_path(folder, section).read_text())
            warmup = section["start"] - section["load_start"]
            frames = section["end"] - section["start"]
            trimmed = temp / f"{section['index']:06d}.mov"
            start = float(Fraction(warmup, 1) / fps)
            video = (folder / data["video"]).resolve()
            run("ffmpeg", "-v", "error", "-xerror", "-n", "-ss", f"{start:.9f}",
                "-i", str(video), "-map", "0:v:0", "-c", "copy",
                "-frames:v", str(frames), str(trimmed))
            stream = validate_video(trimmed, frames, fps, "prores")
            shape = tuple(stream.get(k) for k in (
                "width", "height", "pix_fmt", "profile", "codec_tag_string",
                "sample_aspect_ratio", "color_range", "color_space", "color_transfer", "color_primaries",
            ))
            if reference is not None and shape != reference:
                raise ValueError("Section video formats differ")
            reference = shape
            files.append(trimmed)
        listing = temp / "concat.txt"
        listing.write_text("".join("file '" + str(p).replace("'", "'\\''") + "'\n" for p in files))
        result = temp / "result.mkv"
        run("ffmpeg", "-v", "error", "-xerror", "-n", "-f", "concat", "-safe", "0",
            "-i", str(listing), "-i", str(source), "-map", "0:v:0", "-map", "1:a?",
            "-c", "copy", str(result))
        validate_video(result, parts[-1]["end"], fps, "prores")
        with result.open("rb") as f:
            os.fsync(f.fileno())
        # A hard link publishes without replacing any existing final result.
        os.link(result, final)
        directory = os.open(final.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    print(f"Final output: {final}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=Path,
                        help="API prompt JSON or existing Python API submitter")
    parser.add_argument("--output-root", required=True, type=Path,
                        help="Shared ComfyUI output directory")
    parser.add_argument("--worker", action="append",
                        help="ComfyUI proxy URL; repeat for each worker")
    parser.add_argument("--start-section", type=int, default=START_SECTION, help="First section number, inclusive")
    parser.add_argument("--end-section", type=int, default=END_SECTION, help="Last section number, exclusive")
    parser.add_argument("--plan", action="store_true", help="Print section frame ranges without submitting jobs")
    parser.add_argument("--assemble-only", action="store_true")
    parser.add_argument("--final", type=Path, help="Final MKV path; must not exist")
    args = parser.parse_args()
    args.worker = [url.rstrip("/") for url in
                   (args.worker or ([] if args.plan or args.assemble_only else WORKER_URLS))]
    if args.plan and (args.assemble_only or args.worker):
        raise ValueError("--plan does not submit or assemble jobs")
    if args.assemble_only and (args.worker or not args.final):
        raise ValueError("--assemble-only requires --final and does not use workers")
    if not args.assemble_only and not args.plan and not args.worker:
        raise ValueError("Specify at least one worker proxy URL")
    if len(set(args.worker)) != len(args.worker):
        raise ValueError("Each worker proxy URL must be unique")
    if any(urllib.parse.urlsplit(url).scheme not in ("http", "https")
           or not urllib.parse.urlsplit(url).netloc for url in args.worker):
        raise ValueError("Worker proxy URLs must be HTTP or HTTPS URLs")
    source = args.source.resolve(strict=True)
    root = args.output_root.resolve(strict=True)
    final = args.final.absolute() if args.final else None
    if final and not final.parent.is_dir():
        raise ValueError("Final output parent directory does not exist")
    if final and final.suffix.lower() != ".mkv":
        raise ValueError("Final output must have an .mkv extension")
    if final and final.exists():
        raise FileExistsError(f"Refusing to overwrite {final}")
    template = load_prompt(args.prompt)
    _, total, fps = probe(source)
    parts = sections(total, fps)
    end = len(parts) if args.end_section is None else args.end_section
    if not 0 <= args.start_section < end <= len(parts):
        raise ValueError(f"Section range must satisfy 0 <= start < end <= {len(parts)}")
    if args.plan:
        for part in parts:
            print(f"{part['index']}: frames [{part['start']}, {part['end']}) "
                  f"load [{part['load_start']}, {part['load_end']})")
        return
    identity = job_id(source, template, total, fps)
    prefix_root = f"seedvr-sections/{identity}"
    folder = root / prefix_root
    folder.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as locks:
        # Never duplicate a section still running after a controller restart.
        for base in args.worker:
            worker_id = hashlib.sha256(f"{socket.gethostname()}:{base}".encode()).hexdigest()[:20]
            worker_lock = locks.enter_context((root / "seedvr-sections" / f".worker-{worker_id}.lock").open("a"))
            fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            queue = request(base, "/queue")
            for group in ("queue_running", "queue_pending"):
                for item in queue.get(group, []):
                    if any(node.get("inputs", {}).get("filename_prefix", "").startswith(prefix_root + "/")
                           for node in item[2].values()):
                        raise RuntimeError(f"A section is already active on {base}; wait for it to finish")
                    raise RuntimeError(f"ComfyUI is busy on {base}; wait for its queue to clear")
        selected = parts if args.assemble_only else parts[args.start_section:end]
        pending = [part for part in selected if not completed(folder, part, fps)]
        print(f"{len(selected) - len(pending)}/{len(selected)} selected sections verified "
              f"({len(parts)} total, job {identity})", flush=True)
        if args.assemble_only and pending:
            raise RuntimeError("Not all sections are available for assembly")
        if args.worker:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.worker)) as pool:
                active = {}
                remaining = iter(pending)

                def submit_next(base):
                    for part in remaining:
                        lock = claim_section(folder, part, fps)
                        if lock is not None:
                            future = pool.submit(process_claimed_section, lock, base, template, part,
                                                 folder, prefix_root, source, fps)
                            active[future] = base
                            return

                for base in args.worker:
                    submit_next(base)
                while active:
                    done, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        base = active.pop(future)
                        future.result()
                        submit_next(base)
        if not all(completed(folder, part, fps) for part in selected):
            raise RuntimeError("Some selected sections are unfinished or claimed by another controller")
        if final:
            if not all(completed(folder, part, fps) for part in parts):
                print("Selected sections complete; remaining sections are needed for final assembly", flush=True)
            else:
                assemble(folder, parts, source, fps, final)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
