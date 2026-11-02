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
sys.modules.setdefault("seatsense.seat", types.SimpleNamespace(available=lambda cam: False, SeatModel=None))
sys.modules.setdefault("seatsense.activity", types.SimpleNamespace(available=lambda cam: False, ActivityModel=None, Activity=lambda: None))

from seatsense import app as live   # noqa: E402
from seatsense.auth import Auth   # noqa: E402

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
    """HTTPS on the port, and plain HTTP on the same port answered only with a redirect to HTTPS."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cert, key = Path(cls.tmp.name) / "cert.pem", Path(cls.tmp.name) / "key.pem"
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1", "-subj", "/CN=test",
                        "-addext", "subjectAltName=IP:203.0.113.7,DNS:desk.example", "-keyout", str(key), "-out", str(cert)], check=True, capture_output=True)
        cls.auth = Auth(Path(cls.tmp.name) / "users.json", Path(cls.tmp.name) / "access.log", service_key="")
        cls.auth.set_user("hr", PW, camera=True)
        cls.srv = live.Server(("127.0.0.1", 0), live.make_handler([live.Camera("d20", 20, "Camera D20")], cls.auth))
        live.use_tls(cls.srv, cert, key)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.port = cls.srv.server_address[1]
        cls.ctx = ssl.create_default_context(cafile=str(cert))
        cls.ctx.check_hostname = False

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.tmp.cleanup()

    def plain(self, raw):
        """Send raw bytes as plain HTTP, return the whole reply."""
        import socket
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as s:
            s.sendall(raw)
            out = b""
            while chunk := s.recv(4096):
                out += chunk
        return out.decode("latin-1")

    def test_https_works(self):
        with urllib.request.urlopen(f"https://127.0.0.1:{self.port}/login", context=self.ctx, timeout=5) as r:
            self.assertEqual(r.status, 200)

    def test_plain_http_is_redirected_to_the_same_page_over_https(self):
        out = self.plain(f"GET /reports?date=2026-10-24 HTTP/1.1\r\nHost: 127.0.0.1:{self.port}\r\n\r\n".encode())
        self.assertTrue(out.startswith("HTTP/1.1 301"), out[:40])
        self.assertIn(f"Location: https://127.0.0.1:{self.port}/reports?date=2026-10-24\r\n", out)

    def test_redirect_never_points_to_a_foreign_host(self):
        out = self.plain(b"GET /x HTTP/1.1\r\nHost: evil.example\r\n\r\n")
        self.assertIn(f"Location: https://127.0.0.1:{self.port}/x\r\n", out)
        self.assertNotIn("evil", out)

    def test_redirect_keeps_a_name_the_certificate_holds(self):
        # in Docker the socket address is the container's; the browser asked for the host by a name
        # the certificate holds, so that name is where it goes back to
        for host in ("203.0.113.7:8095", "desk.example:8095"):
            out = self.plain(f"GET /x HTTP/1.1\r\nHost: {host}\r\n\r\n".encode())
            self.assertIn(f"Location: https://{host}/x\r\n", out)

    def test_redirect_carries_nothing_else(self):
        out = self.plain(f"GET /a%0d%0aSet-Cookie:x HTTP/1.1\r\nHost: 127.0.0.1:{self.port}\r\n\r\n".encode())
        head = out.split("\r\n\r\n")[0]
        self.assertNotIn("Set-Cookie", head.replace("Set-Cookie:x", ""))   # only inside the still-encoded path
        self.assertNotIn("\r\nSet-Cookie", head)

    def test_nothing_is_served_or_processed_over_plain_http(self):
        body = json.dumps({"name": "hr", "password": PW})
        out = self.plain(f"POST /login HTTP/1.1\r\nHost: 127.0.0.1:{self.port}\r\nContent-Type: application/json\r\n"
                         f"Content-Length: {len(body)}\r\n\r\n{body}".encode())
        self.assertTrue(out.startswith("HTTP/1.1 301"))
        log = Path(self.tmp.name) / "access.log"
        self.assertNotIn("login ok", log.read_text() if log.exists() else "")   # the password was never checked
        out = self.plain(f"GET /state HTTP/1.1\r\nHost: 127.0.0.1:{self.port}\r\nCookie: dp_session=x\r\n\r\n".encode())
        self.assertTrue(out.startswith("HTTP/1.1 301"))
        self.assertNotIn("cameras", out)

    def test_silent_clients_do_not_block_anyone_and_nothing_fills_the_log(self):
        import contextlib, io, socket, time
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            with socket.create_connection(("127.0.0.1", self.port)):          # connects, says nothing
                with urllib.request.urlopen(f"https://127.0.0.1:{self.port}/login", context=self.ctx, timeout=5) as r:
                    self.assertEqual(r.status, 200)
            self.plain(b"garbage\r\n\r\n")
            time.sleep(2.5)
        self.assertNotIn("Traceback", err.getvalue())

    def test_browser_security_headers(self):
        with urllib.request.urlopen(f"https://127.0.0.1:{self.port}/login", context=self.ctx, timeout=5) as r:
            csp = r.headers["Content-Security-Policy"]
        for rule in ("default-src 'self'", "frame-ancestors 'none'", "form-action 'self'", "base-uri 'none'"):
            self.assertIn(rule, csp)


class Listen(unittest.TestCase):
    """Where the server listens: never plain HTTP on the network unless asked for explicitly."""

    def test_with_a_certificate_it_serves_https_to_the_network(self):
        self.assertEqual(live.listen(certs=True, bind=""), ("0.0.0.0", True))

    def test_without_one_it_stays_on_this_machine(self):
        self.assertEqual(live.listen(certs=False, bind=""), ("127.0.0.1", False))

    def test_a_container_sets_where_to_listen_and_https_still_follows_the_certificate(self):
        self.assertEqual(live.listen(certs=False, bind="0.0.0.0"), ("0.0.0.0", False))
        self.assertEqual(live.listen(certs=True, bind="0.0.0.0"), ("0.0.0.0", True))

    def test_running_code_version_never_crashes_without_git(self):
        import unittest.mock
        with unittest.mock.patch("subprocess.run", side_effect=FileNotFoundError):
            self.assertEqual(live.running_commit(), "unknown")


if __name__ == "__main__":
    unittest.main()
