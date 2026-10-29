"""Stopping and starting the comfy-gateway process from the interface.

The gateway is launched by its desktop shortcut: an alacritty window running
`scripts/launch.sh`, which starts the gateway, which starts ComfyUI. Stopping
it here sends SIGTERM to the gateway's own process, which stops its ComfyUI
on the way out, then closes the window the shortcut opened (it is held open
with `--hold`). Starting it runs the shortcut's command again.

The process is found by the port it listens on, read from `/proc`, and is
only signalled when its command line is comfy-gateway's: whatever else might
hold the port is never touched.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from . import gateway

GATEWAY_DIR = Path(
    os.environ.get("PS_GATEWAY_DIR", Path.home() / "Projets" / "comfy-gateway")
).expanduser()
LAUNCH_SCRIPT = GATEWAY_DIR / "scripts" / "launch.sh"
# The shortcut's window class, kept so the window looks the same either way.
WINDOW_CLASS = "comfy-gateway,comfy-gateway"
# What a gateway process's command line contains.
GATEWAY_MARKER = "comfy_gateway.main:app"

# The gateway gives its ComfyUI 20 s to stop before it exits itself.
STOP_WAIT_SECONDS = 30.0
KILL_WAIT_SECONDS = 5.0
START_WAIT_SECONDS = 30.0
POLL_SECONDS = 0.25

PROC = Path("/proc")
_LISTEN = "0A"


class GatewayProcessError(RuntimeError):
    """The gateway could not be stopped or started. The message says why."""


@dataclass(frozen=True)
class ProcessOutcome:
    """What a stop or a start did, in words the interface shows as they are."""

    running: bool
    message: str


def _port() -> int:
    return urlparse(gateway.GATEWAY_BASE_URL).port or 80


def listening_inodes(table: str, port: int) -> set[str]:
    """Socket inodes listening on `port` in one `/proc/net/tcp*` table."""
    inodes = set()
    for line in table.splitlines()[1:]:
        fields = line.split()
        if len(fields) < 10:
            continue
        local, state, inode = fields[1], fields[3], fields[9]
        if state == _LISTEN and int(local.rsplit(":", 1)[1], 16) == port:
            inodes.add(inode)
    return inodes


def _listening_pids(port: int) -> list[int]:
    inodes: set[str] = set()
    for name in ("tcp", "tcp6"):
        try:
            inodes |= listening_inodes((PROC / "net" / name).read_text(), port)
        except OSError:
            continue
    if not inodes:
        return []
    targets = {f"socket:[{inode}]" for inode in inodes}
    pids = []
    for entry in PROC.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            links = {os.readlink(fd) for fd in (entry / "fd").iterdir()}
        except OSError:
            continue  # another user's process, or gone meanwhile
        if links & targets:
            pids.append(int(entry.name))
    return pids


def _cmdline(pid: int) -> list[str]:
    try:
        raw = (PROC / str(pid) / "cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode(errors="replace") for part in raw.split(b"\0") if part]


def _parent(pid: int) -> int | None:
    try:
        stat = (PROC / str(pid) / "stat").read_text()
    except OSError:
        return None
    # The command name is in parentheses and may hold spaces: split after it.
    fields = stat.rsplit(")", 1)[1].split()
    return int(fields[1]) if len(fields) > 1 else None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _launch_window(pid: int) -> int | None:
    """The alacritty window the shortcut opened around this gateway, if any."""
    current: int | None = pid
    for _ in range(6):
        current = _parent(current) if current is not None else None
        if current is None or current <= 1:
            return None
        cmd = _cmdline(current)
        if cmd and Path(cmd[0]).name == "alacritty" and str(LAUNCH_SCRIPT) in cmd:
            return current
    return None


def _gateway_pids() -> list[int]:
    """The gateway processes holding the port. Raises if something else does."""
    pids = _listening_pids(_port())
    others = [pid for pid in pids if GATEWAY_MARKER not in " ".join(_cmdline(pid))]
    if others:
        raise GatewayProcessError(
            f"Port {_port()} is held by a process that is not comfy-gateway "
            f"(pid {', '.join(map(str, others))}): it is left alone."
        )
    return pids


def _wait(predicate, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(POLL_SECONDS)
    return predicate()


def is_running() -> bool:
    try:
        return bool(_gateway_pids())
    except GatewayProcessError:
        return False


def stop() -> ProcessOutcome:
    """Stop the gateway, and with it its ComfyUI, then close its window."""
    pids = _gateway_pids()
    if not pids:
        return ProcessOutcome(running=False, message="comfy-gateway was not running.")
    windows = {w for w in (_launch_window(pid) for pid in pids) if w is not None}
    for pid in pids:
        os.kill(pid, signal.SIGTERM)
    forced = False
    if not _wait(lambda: not any(_alive(pid) for pid in pids), STOP_WAIT_SECONDS):
        # Past its own grace period: the gateway is stuck. Its ComfyUI gets
        # SIGTERM from the kernel when it dies (PR_SET_PDEATHSIG).
        forced = True
        for pid in pids:
            if _alive(pid):
                os.kill(pid, signal.SIGKILL)
        _wait(lambda: not any(_alive(pid) for pid in pids), KILL_WAIT_SECONDS)
    for window in windows:
        if _alive(window):
            os.kill(window, signal.SIGTERM)
    if any(_alive(pid) for pid in pids):
        raise GatewayProcessError("comfy-gateway did not stop, even after SIGKILL.")
    if forced:
        return ProcessOutcome(
            running=False,
            message=(
                f"comfy-gateway did not stop within {STOP_WAIT_SECONDS:.0f} s and was killed. "
                "Its ComfyUI was sent SIGTERM."
            ),
        )
    return ProcessOutcome(running=False, message="comfy-gateway and its ComfyUI are stopped.")


def start() -> ProcessOutcome:
    """Run the shortcut's command again: the gateway in its alacritty window."""
    if _gateway_pids():
        return ProcessOutcome(running=True, message="comfy-gateway is already running.")
    if not LAUNCH_SCRIPT.is_file():
        raise GatewayProcessError(
            f"{LAUNCH_SCRIPT} does not exist. Set PS_GATEWAY_DIR to comfy-gateway's folder."
        )
    terminal = shutil.which("alacritty")
    if terminal is None:
        raise GatewayProcessError(
            "alacritty is not installed, so the gateway's window cannot open."
        )
    if not (os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY")):
        raise GatewayProcessError(
            "Persona Studio was not started from the desktop session, so it cannot open "
            "the gateway's window. Start comfy-gateway from its shortcut."
        )
    subprocess.Popen(  # noqa: S603 - fixed command, no shell, no user input
        [terminal, "--class", WINDOW_CLASS, "--hold", "-e", str(LAUNCH_SCRIPT)],
        cwd=GATEWAY_DIR,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        # Its own session: stopping or restarting Persona Studio leaves it running.
        start_new_session=True,
    )
    if _wait(is_running, START_WAIT_SECONDS):
        return ProcessOutcome(
            running=True,
            message="comfy-gateway is started. ComfyUI may need a minute more to answer.",
        )
    return ProcessOutcome(
        running=False,
        message=(
            f"The gateway's window was opened, but nothing listens on port {_port()} after "
            f"{START_WAIT_SECONDS:.0f} s. Its window says why."
        ),
    )


def restart() -> ProcessOutcome:
    stop()
    return start()
