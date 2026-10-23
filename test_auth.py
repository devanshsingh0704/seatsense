import tempfile
import unittest
from pathlib import Path

from auth import Auth, cookie_token, gate, hash_password

T0 = 1_000_000.0


def make(tmp, **users):
    a = Auth(Path(tmp) / "users.json", Path(tmp) / "access.log", service_key="svc-key")
    for name, camera in users.items():
        a.set_user(name, "correct horse", camera=camera)
    return a


class Passwords(unittest.TestCase):
    def test_hash_is_salted_and_never_the_password(self):
        h1, h2 = hash_password("secret"), hash_password("secret")
        self.assertNotEqual(h1, h2)
        self.assertNotIn("secret", h1)

    def test_login_with_the_right_password_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = make(tmp, hr=True)
            self.assertIsNotNone(a.login("hr", "correct horse", "10.0.0.5", now=T0))
            self.assertIsNone(a.login("hr", "wrong", "10.0.0.5", now=T0))
            self.assertIsNone(a.login("nobody", "correct horse", "10.0.0.5", now=T0))

    def test_users_file_survives_a_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            make(tmp, hr=True)
            again = Auth(Path(tmp) / "users.json", Path(tmp) / "access.log", service_key="k")
            self.assertIsNotNone(again.login("hr", "correct horse", "ip", now=T0))

    def test_an_account_added_by_the_tool_works_without_a_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            running = make(tmp)                                                        # the dashboard
            tool = Auth(Path(tmp) / "users.json", Path(tmp) / "access.log", service_key="")   # tools/users.py
            tool.set_user("newhr", "correct horse", camera=True)
            self.assertIsNotNone(running.login("newhr", "correct horse", "ip", now=T0))


class Lockout(unittest.TestCase):
    def test_five_wrong_passwords_lock_the_account_for_15_minutes(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = make(tmp, hr=True)
            for i in range(5):
                a.login("hr", "wrong", "ip", now=T0 + i)
            self.assertIsNone(a.login("hr", "correct horse", "ip", now=T0 + 10))          # locked, even with the right one
            self.assertTrue(a.locked("hr", now=T0 + 10))
            self.assertIsNotNone(a.login("hr", "correct horse", "ip", now=T0 + 15 * 60 + 10))

    def test_a_good_login_clears_earlier_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = make(tmp, hr=True)
            for i in range(4):
                a.login("hr", "wrong", "ip", now=T0 + i)
            a.login("hr", "correct horse", "ip", now=T0 + 5)
            a.login("hr", "wrong", "ip", now=T0 + 6)
            self.assertFalse(a.locked("hr", now=T0 + 7))


class Sessions(unittest.TestCase):
    def test_session_lasts_8_hours_and_ends_on_logout(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = make(tmp, hr=True)
            tok = a.login("hr", "correct horse", "ip", now=T0)
            self.assertEqual(a.user(tok, now=T0 + 3600)["name"], "hr")
            self.assertIsNone(a.user(tok, now=T0 + 8 * 3600 + 1))
            tok = a.login("hr", "correct horse", "ip", now=T0)
            a.logout(tok, "ip")
            self.assertIsNone(a.user(tok, now=T0 + 1))

    def test_removing_a_user_ends_their_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = make(tmp, hr=True)
            tok = a.login("hr", "correct horse", "ip", now=T0)
            a.remove_user("hr")
            self.assertIsNone(a.user(tok, now=T0 + 1))

    def test_camera_permission_comes_from_the_account(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = make(tmp, hr=True, staff=False)
            self.assertTrue(a.user(a.login("hr", "correct horse", "ip", now=T0), now=T0)["camera"])
            self.assertFalse(a.user(a.login("staff", "correct horse", "ip", now=T0), now=T0)["camera"])

    def test_access_log_records_logins_failures_and_logouts_without_passwords(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = make(tmp, hr=True)
            tok = a.login("hr", "correct horse", "10.0.0.5", now=T0)
            a.login("hr", "wrong", "10.0.0.9", now=T0)
            a.logout(tok, "10.0.0.5")
            log = (Path(tmp) / "access.log").read_text()
            self.assertIn("login ok", log); self.assertIn("login failed", log); self.assertIn("logout", log)
            self.assertIn("10.0.0.9", log)
            self.assertNotIn("correct horse", log); self.assertNotIn("wrong", log.replace("login failed", ""))


class Cookie(unittest.TestCase):
    def test_reads_our_cookie_among_others(self):
        self.assertEqual(cookie_token("a=1; dp_session=abc_123-x; b=2"), "abc_123-x")
        self.assertIsNone(cookie_token("a=1"))
        self.assertIsNone(cookie_token(None))


class Gate(unittest.TestCase):
    HR = {"name": "hr", "camera": True}
    STAFF = {"name": "staff", "camera": False}

    def test_signed_out_pages_go_to_login_and_data_is_refused(self):
        self.assertEqual(gate("/", None), "login")
        self.assertEqual(gate("/reports", None), "login")
        self.assertEqual(gate("/state", None), 401)
        self.assertEqual(gate("/frame/d20", None), 401)
        self.assertEqual(gate("/api/report", None), 401)

    def test_login_page_is_open(self):
        self.assertEqual(gate("/login", None), "ok")

    def test_camera_only_for_permitted_accounts(self):
        self.assertEqual(gate("/frame/d20", self.HR), "ok")
        self.assertEqual(gate("/frame/d20", self.STAFF), 403)
        self.assertEqual(gate("/state", self.STAFF), "ok")

    def test_service_key_reads_state_only(self):
        self.assertEqual(gate("/state", None, service=True), "ok")
        self.assertEqual(gate("/frame/d20", None, service=True), 401)
        self.assertEqual(gate("/api/report", None, service=True), 401)


if __name__ == "__main__":
    unittest.main()
