import hashlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import privacy_check as pc


def rules(path, text, private=()):
    return [r for _, r in pc.problems(path, text.encode() if isinstance(text, str) else text, private)]


class Files(unittest.TestCase):
    def test_data_and_secret_files_are_refused(self):
        for p in (".env", ".env.bak", "users.json", "certs/server.pem", "deploy/x.key", "models/seat_d20.pt",
                  "recordings/2026-11-02/frames/d20_1.jpg", "logs/access.log", "states/d20.json", "clip.mp4"):
            self.assertIn("file must not be public", rules(p, "x"), p)

    def test_only_approved_screenshots_pass_and_every_approved_version_stays_valid(self):
        old, new = b"\x89PNG first take", b"\x89PNG retaken"
        pc.APPROVED_IMAGES.update({hashlib.sha256(old).hexdigest(), hashlib.sha256(new).hexdigest()})
        try:
            self.assertEqual(rules("docs/screenshots/test.png", old), [])      # still in the history
            self.assertEqual(rules("docs/screenshots/test.png", new), [])
            self.assertIn("image not approved", rules("docs/screenshots/test.png", new + b"edited"))
            self.assertIn("file must not be public", rules("recordings/x.png", new))   # only as a screenshot
        finally:
            pc.APPROVED_IMAGES.difference_update({hashlib.sha256(old).hexdigest(), hashlib.sha256(new).hexdigest()})


class Content(unittest.TestCase):
    def test_private_addresses(self):
        for ip in ("192.168.4.20", "10.0.0.5", "172.20.1.1"):
            self.assertIn("private IP address", rules("a.py", f"HOST = '{ip}'"), ip)
        self.assertEqual(rules("a.py", "bind 127.0.0.1 or 0.0.0.0, torch 10.1, 172.32.0.1"), [])
        self.assertEqual(rules("tests/test_auth.py", "ip = '10.0.0.5'"), [])     # made-up test address

    def test_credentials_in_a_stream_address(self):
        self.assertIn("credentials in a URL", rules("a.md", "rtsp://user1:hunter22@cam/stream"))
        self.assertEqual(rules("a.py", "url = f\"rtsp://{env['NVR_USER']}:{env['NVR_PASS']}@{host}\""), [])
        self.assertEqual(rules("a.sh", 'URL="rtsp://$NVR_USER:$NVR_PASS@$NVR_HOST:554/x"'), [])

    def test_keys_tokens_hashes(self):
        self.assertIn("private key", rules("a.txt", "-----BEGIN RSA PRIVATE KEY-----"))
        self.assertIn("access token", rules("a.yml", "token: ghp_" + "a1" * 18))
        self.assertIn("password hash", rules("a.json", '{"h": "scrypt$' + "ab" * 16 + "$" + "cd" * 32 + '"}'))

    def test_hard_coded_password_outside_tests(self):
        self.assertIn("hard-coded password", rules("seatsense/a.py", 'password = "Office2026"'))
        self.assertEqual(rules("tests/test_auth.py", 'password = "correct horse"'), [])
        self.assertEqual(rules("seatsense/a.py", 'password = os.environ["PASS"]'), [])

    def test_emails_and_home_paths(self):
        self.assertIn("e-mail address", rules("a.md", "ask someone@company.com"))
        self.assertEqual(rules("a.md", "Co-Authored-By: x <noreply@github.com>"), [])
        self.assertIn("home folder path", rules("a.sh", "cd /home/someone/ws"))

    def test_private_words_are_found_but_never_printed(self):
        found = pc.problems("README.md", b"Installed at the Riverside Plant.", ["riverside plant"])
        self.assertEqual(len(found), 1)
        self.assertNotIn("riverside", str(found).lower())
        self.assertEqual(rules("a.html", "a plantation", ["plant"]), [])      # whole words only
        self.assertEqual(rules("a.md", "at 203.0.113.219:8095", ["203.0.113.2"]), [])   # 2 ≠ 219
        self.assertEqual(len(rules("a.md", "at 203.0.113.219:8095", ["203.0.113.219"])), 1)

    def test_clean_code_passes(self):
        self.assertEqual(rules("seatsense/x.py", "def f(a):\n    return a + 1\n"), [])


class Commits(unittest.TestCase):
    """Commit messages and author / committer identities go public with the repository too."""

    def repo(self, tmp, email):
        env = {**os.environ, "GIT_AUTHOR_NAME": "a", "GIT_AUTHOR_EMAIL": email,
               "GIT_COMMITTER_NAME": "a", "GIT_COMMITTER_EMAIL": email, "GIT_CONFIG_GLOBAL": os.devnull}
        run = lambda *a: subprocess.run(["git", "-C", tmp, *a], check=True, capture_output=True, env=env)
        run("init", "-q")
        (Path(tmp) / "a.txt").write_text("fine\n")
        run("add", "a.txt")
        run("commit", "-q", "-m", "fix camera at 192.168.7.7")
        return env

    def found(self, tmp, mode):
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            return [r for path, data in pc.files(Path(tmp), mode) for _, r in pc.problems(path, data)]
        finally:
            os.chdir(cwd)

    def test_history_checks_messages_and_identities(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.repo(tmp, "someone@company.com")
            found = self.found(tmp, "history")
            self.assertIn("private IP address", found)
            self.assertIn("e-mail address", found)

    def test_a_noreply_identity_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.repo(tmp, "1234+someone@users.noreply.github.com")
            self.assertNotIn("e-mail address", self.found(tmp, "history"))

    def test_a_commit_message_is_checked_before_the_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            msg = Path(tmp) / "MSG"
            msg.write_text("deploy to 10.1.2.3\n# a comment git drops: 10.9.9.9\n")
            self.assertEqual([r for _, r in pc.problems("commit message", pc.message(msg))], ["private IP address"])


if __name__ == "__main__":
    unittest.main()
