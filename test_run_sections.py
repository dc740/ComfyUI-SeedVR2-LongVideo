from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import run_sections as sections


def prompt():
    return {
        "1": {"class_type": "SeedVR2LongVideoInput", "inputs": {"video": "input.mov"}},
        "2": {"class_type": "VHS_LoadVideoPath", "inputs": {
            "skip_first_frames": 0, "frame_load_cap": 0,
            "force_rate": 0, "select_every_nth": 1}},
        "3": {"class_type": "VHS_VideoCombine", "inputs": {
            "format": "video/ProRes", "filename_prefix": "video", "save_output": False}},
        "4": {"class_type": "SeedVR2RemuxOriginalAudio", "inputs": {}},
    }


class SectionTests(unittest.TestCase):
    def test_ranges_cover_frames_without_gaps_and_include_context(self):
        fps = Fraction(30000, 1001)
        parts = sections.sections(4000, fps)
        self.assertEqual([(p["start"], p["end"]) for p in parts],
                         [(0, 1798), (1798, 3596), (3596, 4000)])
        self.assertEqual((parts[1]["load_start"], parts[1]["load_end"]), (1782, 3604))
        self.assertEqual(parts[-1]["load_end"], 4000)

    def test_prompt_uses_range_and_preserves_original(self):
        template = prompt()
        part = sections.sections(4000, Fraction(30000, 1001))[1]
        result = sections.section_prompt(template, part, "job/part", Path("source.mov"))
        self.assertEqual(result["2"]["inputs"]["skip_first_frames"], 1782)
        self.assertEqual(result["2"]["inputs"]["frame_load_cap"], 1822)
        self.assertEqual(result["1"]["inputs"]["video"], "source.mov")
        self.assertEqual(result["3"]["inputs"]["filename_prefix"], "job/part")
        self.assertTrue(result["3"]["inputs"]["save_output"])
        self.assertNotIn("4", result)
        self.assertEqual(template["2"]["inputs"]["skip_first_frames"], 0)

    def test_out_of_order_checkpoints_do_not_skip_missing_sections(self):
        parts = sections.sections(10000, Fraction(30))
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            part = parts[4]
            (folder / "section.mov").touch()
            sections.write_json(sections.marker_path(folder, part),
                                {"section": part, "video": "section.mov"})
            with mock.patch.object(sections, "validate_video"):
                pending = [p["index"] for p in parts
                           if not sections.completed(folder, p, Fraction(30))]
            self.assertEqual(pending, [0, 1, 2, 3, 5])

    def test_incomplete_or_missing_video_is_not_a_checkpoint(self):
        part = sections.sections(100, Fraction(30))[0]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            sections.write_json(sections.marker_path(folder, part),
                                {"section": part, "video": "partial.mov"})
            with mock.patch.object(sections, "validate_video", side_effect=OSError("bad file")):
                self.assertFalse(sections.completed(folder, part, Fraction(30)))

    def test_checkpoint_path_cannot_escape_directory(self):
        part = sections.sections(100, Fraction(30))[0]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            sections.write_json(sections.marker_path(folder, part),
                                {"section": part, "video": "../outside.mov"})
            with self.assertRaises(ValueError):
                sections.completed(folder, part, Fraction(30))

    def test_separate_controllers_claim_sections_independently(self):
        parts = sections.sections(4000, Fraction(30))
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            first = sections.claim_section(folder, parts[0], Fraction(30))
            second = sections.claim_section(folder, parts[1], Fraction(30))
            try:
                self.assertIsNotNone(first)
                self.assertIsNotNone(second)
                self.assertIsNone(sections.claim_section(folder, parts[0], Fraction(30)))
            finally:
                first.close()
                second.close()
            resumed = sections.claim_section(folder, parts[0], Fraction(30))
            self.assertIsNotNone(resumed)
            resumed.close()

    def test_job_identity_is_portable_and_depends_on_source_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a"
            second = Path(tmp) / "b"
            first.write_bytes(b"source")
            second.write_bytes(b"source")
            a = prompt()
            b = prompt()
            b["1"]["inputs"]["video"] = "other-name.mov"
            self.assertEqual(sections.job_id(first, a, 100, Fraction(30)),
                             sections.job_id(second, b, 100, Fraction(30)))
            second.write_bytes(b"changed")
            self.assertNotEqual(sections.job_id(first, a, 100, Fraction(30)),
                                sections.job_id(second, b, 100, Fraction(30)))

    def test_submitter_prompt_is_read_without_executing_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "submit.py"
            path.write_text('raise RuntimeError("must not execute")\n'
                            'prompt = {"1": {"value": FRAME_CAP, "tile": TILE}}\n')
            self.assertEqual(sections.load_prompt(path), {"1": {"value": 1, "tile": 1024}})

    def test_await_tracks_requeued_prompt_until_final_output(self):
        base = "http://proxy"
        prefix = "job/part"
        graph = {"1": {"class_type": "VHS_VideoCombine",
                       "inputs": {"filename_prefix": prefix}}}
        first = {"prompt": [1, "first", graph], "outputs": {"1": {"unfinished_batch": [True]}},
                 "status": {"status_str": "success"}}
        last = {"prompt": [2, "second", graph], "outputs": {
            "1": {"gifs": [{"format": "video/ProRes", "fullpath": "/output/part.mov"}]}},
            "status": {"status_str": "success"}}
        replies = [
            {"queue_running": [[1, "second", graph]]}, {"first": first},
            {"queue_running": []}, {"first": first, "second": last},
        ]
        with mock.patch.object(sections, "request", side_effect=replies), \
             mock.patch.object(sections.time, "sleep"):
            self.assertEqual(sections.await_section(base, prefix), Path("/output/part.mov"))

    def test_bad_rate_conversion_is_rejected(self):
        template = prompt()
        template["2"]["inputs"]["force_rate"] = 24
        with self.assertRaises(ValueError):
            sections.section_prompt(template, sections.sections(100, Fraction(30))[0],
                                    "job/part", Path("source.mov"))

    def test_submission_disables_metadata_images(self):
        part = sections.sections(100, Fraction(30))[0]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            video = folder / "output.mov"
            video.touch()
            with mock.patch.object(sections, "request", return_value={"prompt_id": "id"}) as request, \
                 mock.patch.object(sections, "await_section", return_value=video), \
                 mock.patch.object(sections, "validate_video"):
                sections.process_section("http://proxy", prompt(), part, folder, "job", Path("source.mov"), Fraction(30))
            body = request.call_args.args[2]
            self.assertFalse(body["extra_data"]["extra_pnginfo"]["workflow"]["extra"]["VHS_MetadataImage"])
            data = json.loads(sections.marker_path(folder, part).read_text())
            self.assertEqual(data["video"], "output.mov")

    def test_assemble_only_does_not_require_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = folder / "source"
            source.write_bytes(b"source")
            template = folder / "prompt.json"
            template.write_text(json.dumps(prompt()))
            args = ["run_sections.py", "--source", str(source), "--prompt", str(template),
                    "--output-root", str(folder), "--assemble-only", "--final", str(folder / "final.mkv")]
            with mock.patch.object(sections.sys, "argv", args), \
                 mock.patch.object(sections, "probe", return_value=({}, 100, Fraction(30))), \
                 mock.patch.object(sections, "completed", return_value=True), \
                 mock.patch.object(sections, "request") as request, \
                 mock.patch.object(sections, "assemble") as assemble:
                sections.main()
            request.assert_not_called()
            assemble.assert_called_once()

    def test_assembly_trims_context_and_maps_original_audio(self):
        parts = sections.sections(2000, Fraction(30))
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for part in parts:
                sections.write_json(sections.marker_path(folder, part),
                                    {"section": part, "video": f"section-{part['index']}.mov"})
            commands = []

            def fake_run(*args):
                commands.append(args)
                Path(args[-1]).touch()
                return ""

            final = folder / "final.mkv"
            with mock.patch.object(sections, "run", side_effect=fake_run), \
                 mock.patch.object(sections, "validate_video", return_value={
                     "width": 32, "height": 32, "pix_fmt": "yuv422p10le"}):
                sections.assemble(folder, parts, folder / "source.mov", Fraction(30), final)
            self.assertTrue(final.exists())
            self.assertEqual(commands[1][commands[1].index("-frames:v") + 1], "200")
            self.assertEqual(commands[1][commands[1].index("-ss") + 1], f"{16 / 30:.9f}")
            self.assertIn("1:a?", commands[-1])

    @unittest.skipUnless(shutil.which("ffmpeg") and hasattr(os, "memfd_create"),
                         "requires ffmpeg and Linux memfd")
    def test_prores_stream_copy_drops_the_expected_warmup_frames(self):
        create = subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
             "testsrc=size=32x32:rate=30000/1001:duration=2", "-pix_fmt", "yuv422p10le",
             "-c:v", "prores_ks", "-threads", "1", "-f", "mov",
             "-movflags", "frag_keyframe+empty_moov", "pipe:1"],
            capture_output=True,
        )
        if create.returncode:
            self.skipTest("ffmpeg cannot create a synthetic ProRes stream")

        def frame_hashes(data):
            result = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", "pipe:0", "-map", "0:v:0",
                 "-f", "framemd5", "pipe:1"], input=data, capture_output=True, check=True)
            return [line.rsplit(",", 1)[-1].strip() for line in result.stdout.decode().splitlines()
                    if not line.startswith("#")]

        original = frame_hashes(create.stdout)
        fd = os.memfd_create("section-test")
        try:
            os.write(fd, create.stdout)
            for warmup in (0, 8, 16, 29):
                start = float(Fraction(warmup, 1) / Fraction(30000, 1001))
                result = subprocess.run(
                    ["ffmpeg", "-v", "error", "-ss", f"{start:.9f}",
                     "-i", f"/proc/self/fd/{fd}", "-map", "0:v:0", "-c", "copy",
                     "-frames:v", "5", "-f", "mov", "-movflags",
                     "frag_keyframe+empty_moov", "pipe:1"],
                    pass_fds=(fd,), capture_output=True, check=True)
                self.assertEqual(frame_hashes(result.stdout), original[warmup:warmup + 5])
        finally:
            os.close(fd)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe")
                         and Path("/dev/shm").is_dir(), "requires ffmpeg and shared memory")
    def test_real_join_preserves_each_frame_and_audio(self):
        def call(*args):
            return subprocess.run(args, capture_output=True, text=True, check=True).stdout

        def hashes(path):
            result = call("ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
                          "-f", "framemd5", "pipe:1")
            return [line.rsplit(",", 1)[-1].strip() for line in result.splitlines()
                    if not line.startswith("#")]

        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
            folder = Path(tmp)
            source = folder / "source.mov"
            call("ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                 "testsrc=size=32x32:rate=30:duration=3", "-f", "lavfi", "-i",
                 "sine=frequency=400:duration=3", "-pix_fmt", "yuv422p10le",
                 "-c:v", "prores_ks", "-threads", "1", "-c:a", "pcm_s16le",
                 "-shortest", str(source))
            with mock.patch.object(sections, "SECTION_SECONDS", 1):
                parts = sections.sections(90, Fraction(30))
            for part in parts:
                path = folder / f"raw-{part['index']}.mov"
                call("ffmpeg", "-v", "error", "-ss", f"{part['load_start'] / 30:.9f}",
                     "-i", str(source), "-map", "0:v:0", "-c", "copy", "-frames:v",
                     str(part["load_end"] - part["load_start"]), str(path))
                sections.write_json(sections.marker_path(folder, part),
                                    {"section": part, "video": path.name})
            final = folder / "final.mkv"
            sections.assemble(folder, parts, source, Fraction(30), final)
            self.assertEqual(hashes(source), hashes(final))
            streams = json.loads(call("ffprobe", "-v", "error", "-show_entries",
                                      "stream=codec_type", "-of", "json", str(final)))["streams"]
            self.assertTrue(any(stream["codec_type"] == "audio" for stream in streams))


if __name__ == "__main__":
    unittest.main()
