"""Refuse anything that must not be public: data files, secrets, private addresses, local words.

  python3 -m scripts.privacy_check             # what `git add -A` would publish (CI: every tracked file)
  python3 -m scripts.privacy_check --staged    # what is about to be committed, and by whom (pre-commit hook)
  python3 -m scripts.privacy_check --message F # a commit message (commit-msg hook)
  python3 -m scripts.privacy_check --history   # every file, message and author in every commit

Two local extras, read only where they exist (never in CI, never committed): every value in .env,
and one word or phrase per line in ~/.config/seatsense/private-words.txt (site name, real addresses,
user names, people's names). A hit names the rule and the line, never the private value itself.
Stdlib only, so it runs anywhere."""
import argparse
import fnmatch
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

# whole files that must never be committed (paths are repo-relative, matched per component and in full)
FORBIDDEN = (".env", ".env.*", "users.json", "*.pem", "*.key", "*.crt", "*.p12", "*.pt", "*.pth", "*.onnx",
             "*.engine", "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.mp4", "*.mkv", "*.avi", "*.h264", "*.log")
FORBIDDEN_DIRS = ("recordings", "snapshots", "states", "events", "models", "certs", "logs")
# sha256 of every screenshot a person has reviewed (no faces, camera blurred, no site name), allowed only
# as docs/screenshots/*.png. Add a new take's hash; never remove an old one -- it stays in the history.
APPROVED_IMAGES = {
    "49ddcdcfcf16361f8b2a09eb061088b3cce5e2d3297acf37da137f3f006d50d0",   # dashboard-live.png, 2026-10
    "6064f89082f1588ae49949a62a625935a042fe83f83ea302d4f8d0df2cdb7fd2",   # dashboard-reports.png, 2026-10
}
CONTENT = [
    ("private IP address", re.compile(r"\b(?:10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b")),
    ("credentials in a URL", re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s/:@'\"{}$]+:[^\s/@'\"{}$]+@", re.I)),
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("access token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_\w{30,}|AKIA[0-9A-Z]{16}"
                                r"|sk-[A-Za-z0-9_-]{20,}|gsk_[A-Za-z0-9]{20,}|xox[abprs]-[A-Za-z0-9-]{10,})")),
    ("password hash", re.compile(r"scrypt\$[0-9a-f]{16,}\$[0-9a-f]{32,}")),
    ("e-mail address", re.compile(r"\b[A-Za-z0-9._%+-]+@(?!example\.)[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")),
    ("home folder path", re.compile(r"(?:/home|/Users)/(?!runner/)[A-Za-z_][\w.-]*/")),
]
# test files may hold made-up passwords and addresses; real ones are still caught by the .env and private-word
# checks. The checker's own tests must contain examples of everything, so only private words apply to them.
SELF_TEST = "test_privacy_check.py"
PASSWORD = re.compile(r"""(?i)\b(?:pass(?:word)?|passwd|secret|api_?key|token)\s*[:=]\s*["'][^"'\s]{4,}["']""")
NOREPLY = re.compile(r"noreply@|no-reply@|@users\.noreply\.github\.com", re.I)


def problems(path, data, private=()):
    """[(line, rule)] for one file. `private` are lower-case words that must not appear anywhere."""
    parts = path.split("/")
    if any(p in FORBIDDEN_DIRS for p in parts[:-1]):
        return [(0, "file must not be public")]
    if fnmatch.fnmatch(path, "docs/screenshots/*.png"):
        return [] if hashlib.sha256(data).hexdigest() in APPROVED_IMAGES else [(0, "image not approved")]
    if any(fnmatch.fnmatch(parts[-1], pat) for pat in FORBIDDEN):
        return [(0, "file must not be public")]
    if b"\0" in data[:8000]:
        return [(0, "binary file")]
    test, found = parts[-1].startswith("test_"), []
    words = [re.compile(rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])") for w in private]   # "plant" ≠ "plantation"
    for n, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
        low = line.lower()
        found += [(n, f"private word #{i}") for i, w in enumerate(words, 1) if w.search(low)]
        if parts[-1] == SELF_TEST:
            continue
        for rule, rx in CONTENT:
            if rx.search(line) and not (rule == "e-mail address" and NOREPLY.search(line)) \
                    and not (test and rule == "private IP address"):
                found.append((n, rule))
        if PASSWORD.search(line) and not test:
            found.append((n, "hard-coded password"))
    return found


def private_words(root):
    """Values from .env (key names are public, values are not) plus the local word list, lower-case."""
    words = []
    env = root / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                words.append(line.split("=", 1)[1].strip().strip("\"'"))
    listed = Path(os.environ.get("SEATSENSE_PRIVATE_WORDS", Path.home() / ".config/seatsense/private-words.txt"))
    if listed.exists():
        words += [w.strip() for w in listed.read_text().splitlines() if w.strip() and not w.startswith("#")]
    return [w.lower() for w in words if len(w) >= 4]


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, check=True).stdout


def message(path):
    """A commit message as git will store it: the comment lines it drops are left out."""
    return b"".join(line for line in Path(path).read_bytes().splitlines(keepends=True) if not line.startswith(b"#"))


def files(root, mode):
    """(path, bytes) for everything the chosen mode would publish: files, and for commits their
    messages and author / committer identities (a non-file is named like "commit 1a2b3c4d5")."""
    if mode == "staged":
        for p in git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z").decode().split("\0"):
            if p:
                yield p, git("show", f":{p}")
        yield "author and committer", git("var", "GIT_AUTHOR_IDENT") + git("var", "GIT_COMMITTER_IDENT")
    elif mode == "history":
        for c in git("log", "--all", "-z", "--format=%H%n%an <%ae>%n%cn <%ce>%n%B").split(b"\0"):
            if c.strip():
                yield f"commit {c[:9].decode()}", c[41:]
        for line in git("rev-list", "--objects", "--all", "--filter=object:type=blob").decode().splitlines():
            sha, _, p = line.partition(" ")   # each file version once
            if p:
                yield p, git("cat-file", "blob", sha)
    else:
        for p in git("ls-files", "-z", "--cached", "--others", "--exclude-standard").decode().split("\0"):
            if p and (root / p).is_file():
                yield p, (root / p).read_bytes()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--staged", action="store_const", const="staged", dest="mode")
    ap.add_argument("--history", action="store_const", const="history", dest="mode")
    ap.add_argument("--message", metavar="FILE")
    args = ap.parse_args()
    mode = "message" if args.message else args.mode or "tree"
    msg = Path(args.message).resolve() if args.message else None
    root = Path(git("rev-parse", "--show-toplevel").decode().strip())
    os.chdir(root)
    private = private_words(root)
    hits, count = [], 0
    for path, data in [("commit message", message(msg))] if msg else files(root, mode):
        count += 1
        hits += [f"{path}:{n}: {rule}" for n, rule in problems(path, data, private)]
    hits = sorted(set(hits))
    for h in hits:
        print(h)
    print(f"privacy check ({mode}): {count} files, {len(private)} private values, "
          f"{f'{len(hits)} PROBLEMS' if hits else 'OK'}", file=sys.stderr)
    sys.exit(1 if hits else 0)


if __name__ == "__main__":
    main()
