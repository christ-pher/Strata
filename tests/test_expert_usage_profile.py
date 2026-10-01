"""Usage profiles rank cumulative snapshots without preserving a stale base ranking."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_profile", ROOT / "tools/make_profile.py")
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)

class UsageProfileTest(unittest.TestCase):
    def test_counts_merge_and_duplicate_paths_are_not_counted_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b, out = (Path(tmp) / name for name in ("a.tsv", "b.tsv", "profile.bin"))
            a.write_text("# strata-expert-usage-v1 48 512\n2\t3\t8\n1\t4\t2\n")
            b.write_text("# strata-expert-usage-v1 48 512\n1\t4\t10\n")
            subprocess.run([sys.executable, str(ROOT / "tools/make_profile.py"), "--no-base",
                            "--usage", str(a), str(b), str(a), "--out", str(out)], check=True, capture_output=True)
            ranked = profile.read_profile(out)
            self.assertEqual(ranked[:2], [(1, 4), (2, 3)])
            self.assertEqual(len(set(ranked)), 48 * 512)

    def test_default_base_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, out = Path(tmp) / "a.tsv", Path(tmp) / "profile.bin"
            a.write_text("# strata-expert-usage-v1 48 512\n2\t3\t8\n")
            result = subprocess.run([sys.executable, str(ROOT / "tools/make_profile.py"),
                                     "--usage", str(a), "--out", str(out)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("requires --no-base", result.stderr)
            self.assertFalse(out.exists())

if __name__ == "__main__":
    unittest.main()
