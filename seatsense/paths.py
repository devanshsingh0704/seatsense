"""Where SeatSense keeps its files. Code lives in the repository; runtime data (models/, recordings/,
states/, logs/, certs/, users.json, .env) lives under ROOT -- $SEATSENSE_HOME if set (the Docker image
mounts its data there), else the repository root of a checkout, else the working directory."""
import os
from pathlib import Path


def data_root(home, package_dir, cwd):
    if home:
        return Path(home)
    repo = package_dir.parent
    # an installed package sits in site-packages: data there would be lost on upgrade, or not writable
    return repo if (repo / "pyproject.toml").exists() else cwd


ROOT = data_root(os.environ.get("SEATSENSE_HOME"), Path(__file__).resolve().parent, Path.cwd())
ZONES = ROOT / "config" / "zones"   # the desk boxes per camera
WEB = Path(__file__).resolve().parent / "web"
