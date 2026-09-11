"""
Provenance capture for experiment artefacts.

Without this, a JSON in experiments/ is an assertion rather than a measurement:
there is no way to tie a number back to the code and data that produced it.
"""

import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _git(*cmd) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", *cmd], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def file_digest(path, chunk=1 << 20) -> Optional[str]:
    """Short sha256 of a file, or None if it does not exist."""
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return f"sha256:{h.hexdigest()[:16]}"


def provenance(args: Any = None, data_dir: Optional[str] = None) -> dict:
    """
    Record commit, dirty state, interpreter, seed, and input data digests.

    `git_dirty` matters: a result produced from an uncommitted tree cannot be
    reproduced from the recorded commit, and the artefact should say so.
    """
    out = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    if args is not None:
        for field in ("seed", "encoder", "algo", "symbol", "timesteps", "pretrained"):
            if hasattr(args, field):
                out[field] = getattr(args, field)

    if data_dir:
        root = Path(data_dir)
        out["data_dir"] = str(root)
        digests = {}
        for npz in sorted(root.rglob("*.npz")):
            digests[str(npz.relative_to(root))] = file_digest(npz)
        manifest = root / "manifest.json"
        if manifest.is_file():
            digests["manifest.json"] = file_digest(manifest)
        out["data_digests"] = digests

    try:
        import torch
        out["torch"] = torch.__version__
        out["cuda"] = torch.cuda.is_available()
    except Exception:
        pass

    return out
