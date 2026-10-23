#!/usr/bin/env python3
"""Dashboard accounts. Run on the GPU laptop; the dashboard picks changes up without a restart.

    python3 tools/users.py add <name> [--camera]     # asks for the password twice
    python3 tools/users.py passwd <name>             # new password
    python3 tools/users.py camera <name> on|off      # may this person see the camera?
    python3 tools/users.py remove <name>             # also signs them out
    python3 tools/users.py list

Only salted scrypt hashes are stored, in users.json (never in git).
"""
import sys
from getpass import getpass
from pathlib import Path

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS))
from auth import Auth   # noqa: E402


def ask():
    p = getpass("Password (10+ characters): ")
    if p != getpass("Again: "):
        sys.exit("The two passwords differ; nothing changed.")
    return p


def main():
    (WS / "logs").mkdir(exist_ok=True)
    a = Auth(WS / "users.json", WS / "logs" / "access.log", service_key="")
    cmd, *rest = sys.argv[1:] or ["list"]
    try:
        if cmd == "add" and rest:
            if rest[0] in a.users:
                sys.exit(f"{rest[0]} already exists (use passwd or camera).")
            a.set_user(rest[0], ask(), camera="--camera" in rest)
        elif cmd == "passwd" and rest and rest[0] in a.users:
            a.set_user(rest[0], ask(), camera=a.users[rest[0]]["camera"])
        elif cmd == "camera" and len(rest) == 2 and rest[0] in a.users and rest[1] in ("on", "off"):
            a.set_camera(rest[0], rest[1] == "on")
        elif cmd == "remove" and rest and rest[0] in a.users:
            a.remove_user(rest[0])
        elif cmd != "list":
            sys.exit(__doc__)
    except ValueError as e:
        sys.exit(str(e))
    for name, u in sorted(a.users.items()):
        print(f"{name:20s} camera: {'yes' if u['camera'] else 'no'}")
    if not a.users:
        print("no accounts yet")


if __name__ == "__main__":
    main()
