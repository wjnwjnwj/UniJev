"""Release-manifest verification shared with the CORA configuration checks."""
import json
from pathlib import Path
from campaign_train_v2.configuration import ROOT, sha256


def check_release(c):
    release = json.loads(Path(c["release_manifest"]).read_text())
    for path, expected in release["files"].items():
        if sha256(ROOT / path) != expected:
            raise ValueError("Frozen source changed: " + path)
