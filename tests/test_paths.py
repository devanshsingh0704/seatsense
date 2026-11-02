import tempfile
import unittest
from pathlib import Path

from seatsense.paths import data_root


class DataRoot(unittest.TestCase):
    def test_seatsense_home_wins(self):
        self.assertEqual(data_root("/data", Path("/x/seatsense"), Path("/cwd")), Path("/data"))

    def test_a_checkout_keeps_its_data_beside_the_code(self):
        with tempfile.TemporaryDirectory() as repo:
            (Path(repo) / "pyproject.toml").write_text("")
            self.assertEqual(data_root("", Path(repo) / "seatsense", Path("/cwd")), Path(repo))

    def test_an_installed_package_uses_the_working_directory_not_site_packages(self):
        with tempfile.TemporaryDirectory() as site:
            self.assertEqual(data_root(None, Path(site) / "seatsense", Path("/cwd")), Path("/cwd"))


if __name__ == "__main__":
    unittest.main()
