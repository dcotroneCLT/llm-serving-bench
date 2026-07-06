"""launch_cell image pinning: the local image tag must actually resolve to the
pinned digest (F2). A mutable tag re-pushed after pinning would otherwise be
recorded in the manifest under the pinned digest string while being different
bytes -- a manifest that lies about reproducibility.

Off-box: `docker image inspect` is stubbed. Run:
    python3 -m unittest tests.test_launch_cell_image_pin
"""
import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

import launch_cell as lc  # noqa: E402


class DieCalled(Exception):
    """Stand-in for die()'s sys.exit so tests can assert it fired."""


def _inspect_result(obj):
    """A fake subprocess.CompletedProcess for `docker image inspect`."""
    return types.SimpleNamespace(returncode=0, stdout=json.dumps([obj]), stderr="")


class VerifyImageDigest(unittest.TestCase):
    def setUp(self):
        # die() normally prints, cleans up, and sys.exits; make it raise instead.
        patcher = mock.patch.object(lc, "die", side_effect=DieCalled)
        self.die = patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self, inspect_obj, pinned_digest):
        with mock.patch.object(lc.subprocess, "run",
                               return_value=_inspect_result(inspect_obj)):
            lc.verify_image_digest("repo/img:tag", pinned_digest)

    def test_repodigest_match_passes(self):
        obj = {
            "Id": "sha256:aaaa",
            "RepoDigests": ["repo/img@sha256:deadbeef"],
        }
        self._run(obj, "sha256:deadbeef")  # must not raise
        self.die.assert_not_called()

    def test_repodigest_mismatch_dies(self):
        obj = {
            "Id": "sha256:aaaa",
            "RepoDigests": ["repo/img@sha256:0000live0000"],
        }
        with self.assertRaises(DieCalled):
            self._run(obj, "sha256:deadbeef")  # pinned != live tag

    def test_locally_built_image_id_fallback_passes(self):
        # No RepoDigests (locally built): pin_images.sh recorded the image Id.
        obj = {"Id": "sha256:localbuilt", "RepoDigests": []}
        self._run(obj, "sha256:localbuilt")
        self.die.assert_not_called()

    def test_locally_built_image_id_mismatch_dies(self):
        obj = {"Id": "sha256:localbuilt", "RepoDigests": []}
        with self.assertRaises(DieCalled):
            self._run(obj, "sha256:somethingelse")

    def test_absent_image_dies(self):
        with mock.patch.object(
            lc.subprocess, "run",
            return_value=types.SimpleNamespace(returncode=1, stdout="", stderr="No such image"),
        ):
            with self.assertRaises(DieCalled):
                lc.verify_image_digest("repo/img:tag", "sha256:deadbeef")


if __name__ == "__main__":
    unittest.main()
