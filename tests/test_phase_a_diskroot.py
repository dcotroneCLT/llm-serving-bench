"""PHASE A point 6: disk checks target the REAL docker data-root (DockerRootDir),
not /var/lib. Run: python3 -m unittest tests.test_phase_a_diskroot
"""
import subprocess
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = [REPO / "scripts" / "campaign_health.sh",
           REPO / "scripts" / "smoke_test.sh",
           REPO / "scripts" / "smoke_test_run.sh"]
# Old patterns that pointed the disk check at /var/lib must be gone.
STALE = ["disk_free_gb /var/lib", "df --output=avail -BG /var/lib ",
         "df -h /var/lib /home"]


class DiskRoot(unittest.TestCase):
    def test_scripts_parse(self):
        for s in SCRIPTS:
            r = subprocess.run(["bash", "-n", str(s)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f"{s.name} bash -n: {r.stderr}")

    def test_uses_docker_root_dir(self):
        for s in SCRIPTS:
            self.assertIn("docker info -f '{{.DockerRootDir}}'", s.read_text(),
                          f"{s.name} must resolve the docker data-root via docker info")

    def test_no_stale_var_lib_disk_check(self):
        for s in SCRIPTS:
            txt = s.read_text()
            for pat in STALE:
                self.assertNotIn(pat, txt, f"{s.name} still has a /var/lib disk check: {pat!r}")

    def test_server_setup_doc_updated(self):
        doc = (REPO / "docs" / "server_setup.md").read_text()
        self.assertIn("DockerRootDir", doc)
        self.assertNotIn("df -h /var/lib", doc)


if __name__ == "__main__":
    unittest.main()
