"""The dashboard's real HTTP handler with sign-in, over real HTTP on a free port. The model
modules are stood in for (no GPU or torch needed): the camera has no picture and no models."""
import json
import ssl
import subprocess
import sys
import tempfile
import threading
import types
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.modules.setdefault("ultralytics", types.SimpleNamespace(YOLO=None))
sys.modules.setdefault("seat", types.SimpleNamespace(available=lambda cam: False, SeatModel=None))
sys.modules.setdefault("activity", types.SimpleNamespace(available=lambda cam: False, ActivityModel=None, Activity=lambda: None))

import live   # noqa: E402
from auth import Auth   # noqa: E402

PW = "correct horse"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


OPEN = urllib.request.build_opener(NoRedirect)


class SignedIn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.auth = Auth(Path(cls.tmp.name) / "users.json", Path(cls.tmp.name) / "access.log", service_key="svc")
        cls.auth.set_user("hr", PW, camera=True)
        cls.auth.set_user("staff", PW, camera=False)
        cams = [live.Camera("d20", 20, "Camera D20")]
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), live.make_handler(cams, cls.auth))
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.tmp.cleanup()

    def req(self, path, data=None, cookie=None, headers=None):
        r = urllib.request.Request(self.base + path, data=json.dumps(data).encode() if data is not None else None,
                                   headers={**({"Cookie": cookie} if cookie else {}), **(headers or {})})
        try:
            with OPEN.open(r, timeout=5) as resp:
                return resp.status, resp.headers, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers, e.read()

    def login(self, name):
        code, h, _ = self.req("/login", {"name": name, "password": PW})
        self.assertEqual(code, 200)
        return h["Set-Cookie"].split(";")[0]

    def test_signed_out_pages_redirect_and_data_is_refused(self):
        code, h, _ = self.req("/")
        self.assertEqual((code, h["Location"]), (303, "/login"))
        for path in ("/state", "/frame/d20", "/api/report", "/api/mistakes"):
            self.assertEqual(self.req(path)[0], 401, path)
        self.assertEqual(self.req("/login")[0], 200)

    def test_wrong_password_is_refused_with_one_generic_message(self):
        code, _, body = self.req("/login", {"name": "hr", "password": "nope"})
        code2, _, body2 = self.req("/login", {"name": "nobody", "password": "nope"})
        self.assertEqual((code, code2), (401, 401))
        self.assertEqual(body, body2)   # does not tell which accounts exist

    def test_session_cookie_is_locked_down(self):
        _, h, _ = self.req("/login", {"name": "hr", "password": PW})
        c = h["Set-Cookie"]
        for flag in ("HttpOnly", "Secure", "SameSite=Strict", "Path=/"):
            self.assertIn(flag, c)

    def test_signed_in_state_says_who_and_whether_they_may_see_the_camera(self):
        code, _, body = self.req("/state", cookie=self.login("staff"))
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)["user"], {"name": "staff", "camera": False})

    def test_camera_refused_without_permission(self):
        self.assertEqual(self.req("/frame/d20", cookie=self.login("staff"))[0], 403)
        self.assertNotIn(self.req("/frame/d20", cookie=self.login("hr"))[0], (401, 403))   # 503 here: no picture yet

    def test_logout_ends_the_session(self):
        c = self.login("hr")
        self.assertEqual(self.req("/logout", {}, cookie=c)[0], 200)
        self.assertEqual(self.req("/state", cookie=c)[0], 401)

    def test_frame_logger_reads_state_with_its_service_key_only(self):
        self.assertEqual(self.req("/state", headers={"X-Service-Key": "svc"})[0], 200)
        self.assertEqual(self.req("/state", headers={"X-Service-Key": "wrong"})[0], 401)
        self.assertEqual(self.req("/frame/d20", headers={"X-Service-Key": "svc"})[0], 401)

    def test_reporting_a_mistake_needs_a_sign_in(self):
        self.assertEqual(self.req("/api/mistake", {"desks": {"1": "seated_shown_vacant"}})[0], 401)

    def test_pages_are_not_framed_or_sniffed(self):
        _, h, _ = self.req("/login")
        self.assertEqual(h["X-Frame-Options"], "DENY")
        self.assertEqual(h["X-Content-Type-Options"], "nosniff")


class Https(unittest.TestCase):
    def test_serves_https_with_a_self_made_certificate(self):
        with tempfile.TemporaryDirectory() as tmp:
            cert, key = Path(tmp) / "cert.pem", Path(tmp) / "key.pem"
            subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1", "-subj", "/CN=test",
                            "-keyout", str(key), "-out", str(cert)], check=True, capture_output=True)
            auth = Auth(Path(tmp) / "users.json", Path(tmp) / "access.log", service_key="")
            srv = ThreadingHTTPServer(("127.0.0.1", 0), live.make_handler([live.Camera("d20", 20, "Camera D20")], auth))
            live.use_tls(srv, cert, key)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            try:
                ctx = ssl.create_default_context(cafile=str(cert))
                ctx.check_hostname = False
                with urllib.request.urlopen(f"https://127.0.0.1:{srv.server_address[1]}/login", context=ctx, timeout=5) as r:
                    self.assertEqual(r.status, 200)
                with self.assertRaises(Exception):   # plain HTTP is not answered on the HTTPS port
                    urllib.request.urlopen(f"http://127.0.0.1:{srv.server_address[1]}/login", timeout=3).read()
                import socket   # a client that connects and stays silent must not block everyone else
                with socket.create_connection(("127.0.0.1", srv.server_address[1])):
                    with urllib.request.urlopen(f"https://127.0.0.1:{srv.server_address[1]}/login", context=ctx, timeout=5) as r:
                        self.assertEqual(r.status, 200)
            finally:
                srv.shutdown()


if __name__ == "__main__":
    unittest.main()
