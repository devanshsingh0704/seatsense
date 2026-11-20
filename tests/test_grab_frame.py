import os
import subprocess
import tempfile
import unittest
from pathlib import Path

GRAB = Path(__file__).resolve().parent.parent / "scripts" / "grab_frame.sh"
# stands in for ffmpeg: writes its last argument (the output file) as FAKE_BYTES says, exits FAKE_EXIT
FAKE = '#!/bin/sh\nfor out; do :; done\nprintf "%s" "$FAKE_BYTES" > "$out"\nexit "$FAKE_EXIT"\n'


class GrabFrame(unittest.TestCase):
    """A frame is saved whole or not at all (16 Nov: a grab cut off by its timeout left a 0-byte frame)."""

    def grab(self, content, code):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir, out = Path(tmp) / "bin", Path(tmp) / "frames" / "d20_36480.jpg"
            bin_dir.mkdir()
            out.parent.mkdir()
            (bin_dir / "ffmpeg").write_text(FAKE)
            (bin_dir / "ffmpeg").chmod(0o755)
            env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_BYTES": content, "FAKE_EXIT": str(code)}
            rc = subprocess.run([str(GRAB), "rtsp://cam/stream", str(out)], env=env).returncode
            return rc, sorted(p.name for p in out.parent.iterdir()), out.read_bytes() if out.exists() else None

    def test_a_good_grab_is_saved(self):
        self.assertEqual(self.grab("jpeg bytes", 0), (0, ["d20_36480.jpg"], b"jpeg bytes"))

    def test_a_failed_grab_leaves_nothing(self):
        self.assertEqual(self.grab("half", 1), (1, [], None))

    def test_an_empty_grab_leaves_nothing(self):
        self.assertEqual(self.grab("", 0), (1, [], None))


if __name__ == "__main__":
    unittest.main()
