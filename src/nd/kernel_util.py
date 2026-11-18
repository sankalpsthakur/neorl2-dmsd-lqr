"""Helpers shared by the kernel mains (no heavy imports)."""
import json, os, pathlib, time, datetime


def arm_wall_cap(seconds):
    """In-kernel wall cap (PREREGISTRATION section 6): SIGALRM after `seconds` ends the script with SystemExit, so the kernel finishes in ERROR state and the slot can be released.
    Best effort (main thread, POSIX)."""
    import signal

    def _stop(signum, frame):
        raise SystemExit(f"WALL CAP of {seconds} s reached: stopped by the kernel's own timer")
    try:
        signal.signal(signal.SIGALRM, _stop); signal.alarm(int(seconds))
    except Exception:
        pass


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def find_dir(marker, bases=("/kaggle/input",), max_depth=6):
    """Locate a mounted kernel output holding the file `marker`."""
    for base in bases:
        for dp, dn, fn in os.walk(base):
            if dp[len(base):].count(os.sep) >= max_depth:
                dn[:] = []
            if marker in fn:
                return dp
    raise SystemExit(f"no mounted output holding {marker} under {bases}: run the earlier kernel first and list it as a kernel source")


def write_json(path, obj):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=float)
