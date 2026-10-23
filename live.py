"""Live desk presence: D20 -> person detection -> per-box occupancy -> dashboard.

Runs on the GPU laptop (219). The dashboard binds to 127.0.0.1 only; view it through
    ssh -N -L 8095:127.0.0.1:8095 <user>@<gpu-host>   ->   http://localhost:8095
No video is stored -- only occupancy changes, in events/<date>.csv.

Frames are decoded by ffmpeg at VIEW_FPS straight to W x H, so the view is smooth while
detection runs slower (both cameras batched every DET_PERIOD s). A camera that stops
delivering frames freezes its desks rather than marking them vacant.
"""
import csv
import json
import os
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from presence import Desk, _inside_fraction, assign, load_boxes, seat_present
import report
import seat
import activity
from vest import is_visitor
from watchdog import watch

HERE = Path(__file__).parent
CAMERAS = {"d20": (20, "Camera D20")}   # D3 dropped by the user on 2026-10-16; CAMERA_NAME in .env renames it
SRC_W = 2560                # boxes are marked on the 2560x1440 main stream
W, H = 1280, 720            # decode size; detection + view work at this size
VIEW_FPS = 10
DET_PERIOD = 1.0            # seconds between detection passes; desks don't need more
IMGSZ, CONF = 960, 0.35
STALE_S = 5                 # no frame for this long = camera offline
PORT = 8095
# "classifier": the trained seat model decides each box (cameras that have one); "box": the 85%-inside rule
DESK_MODE = os.environ.get("DESK_MODE", "classifier")
LIME, YELLOW, CYAN, ORANGE = (0, 255, 0), (0, 255, 255), (255, 200, 0), (0, 140, 255)  # BGR

EVENTS = deque(maxlen=50)
STATS = {"det_ms": None, "det_at": None, "gpu": {}, "started": time.time()}


def load_env():
    for line in (HERE / ".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


class Camera:
    def __init__(self, cid, channel, name):
        self.id, self.channel, self.name = cid, channel, name
        self.boxes = load_boxes(HERE / "zones" / f"{cid}_focus.json")
        self.desks = {n: Desk() for n in self.boxes}
        s = W / SRC_W
        self.view_boxes = {n: tuple(round(v * s) for v in b) for n, b in self.boxes.items()}
        self.frame, self.frame_t, self.status = None, 0.0, "connecting"
        self.persons = []   # [(x1, y1, x2, y2, conf, box)] in view pixels, inside a box only
        self.visitors = []  # [(x1, y1, x2, y2)] people in the visitor vest, never counted
        self.seat = seat.SeatModel(cid) if DESK_MODE == "classifier" and seat.available(cid) else None
        self.p_sitting = {}  # box -> latest classifier probability
        # beta: working / not working for occupied desks, when the activity model is there
        self.act = activity.ActivityModel(cid) if self.seat and activity.available(cid) else None
        self.activity = {n: activity.Activity() for n in self.boxes}
        self.p_not, self.act_state = {}, {}
        self._cache = {}    # people flag -> (frame_t, jpeg bytes)

    def url(self):
        return (f"rtsp://{os.environ['NVR_USER']}:{os.environ['NVR_PASS']}@{os.environ['NVR_HOST']}:554"
                f"/cam/realmonitor?channel={self.channel}&subtype=0")

    def read_forever(self):
        backoff, size = 2, W * H * 3
        cmd = ["ffmpeg", "-loglevel", "error", "-rtsp_transport", "tcp", "-i", self.url(), "-an",
               "-vf", f"fps={VIEW_FPS},scale={W}:{H}", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
        while True:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            # a stream that dies without closing would block the read below forever
            threading.Thread(target=watch, args=(proc, lambda: self.frame_t), kwargs={"stall_s": 15}, daemon=True).start()
            while len(buf := proc.stdout.read(size)) == size:
                self.frame = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
                self.frame_t, self.status, backoff = time.time(), "live", 2
            proc.kill()
            proc.wait()
            self.status = f"reconnecting in {backoff} s"
            print(f"[{self.id}] stream lost, {self.status}", flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)

    def online(self, now):
        return self.frame is not None and now - self.frame_t < STALE_S

    def apply(self, xyxy, confs, now, frame):
        # visitors always wear the hi-vis vest: they never occupy a desk
        vis = [is_visitor(frame, b) for b in xyxy]
        self.visitors = [tuple(map(int, b)) for b, v in zip(xyxy, vis) if v]
        xyxy = [b for b, v in zip(xyxy, vis) if not v]
        confs = [c for c, v in zip(confs, vis) if not v]
        src = [tuple(v * SRC_W / W for v in b) for b in xyxy]
        if self.seat:
            # the seat classifier decides; people boxes are only for the overlay and the visitor check
            owners = assign(src, self.boxes, min_inside=0.5)
            self.p_sitting = self.seat.predict(frame, self.view_boxes)
            vis_src = [tuple(v * SRC_W / W for v in b) for b in self.visitors]
            present = {n: seat_present(self.p_sitting[n],
                                       any(_inside_fraction(v, b) >= 0.5 for v in vis_src), bool(owners[n]))
                       for n, b in self.boxes.items()}
        else:
            owners = assign(src, self.boxes)
            present = {n: bool(owners[n]) for n in self.boxes}
        box_of = {i: n for n, idx in owners.items() for i in idx}
        self.persons = [(*map(int, xyxy[i]), confs[i], box_of[i]) for i in box_of]
        for n, desk in self.desks.items():
            ev = desk.update(present[n], now)
            if ev:
                log_event(self, n, ev)
        if self.act:
            occ = {n: self.view_boxes[n] for n, d in self.desks.items() if d.state == "occupied"}
            self.p_not = self.act.predict(frame, occ)
            for n in self.boxes:
                if n in occ:
                    self.act_state[n] = self.activity[n].update(self.p_not[n], now)
                else:
                    self.activity[n].reset()
                    self.act_state.pop(n, None)

    def desk_state(self, n):
        """vacant / occupied, or working / not for an occupied desk once the activity model runs."""
        return self.act_state.get(n, self.desks[n].state)

    def render(self, people=False, boxes=True):
        """Full picture, lime focus boxes, yellow box numbers; detected people only on request.
        boxes=False gives the clean picture -- the new dashboard draws coloured boxes itself."""
        out = self.frame.copy()
        if people:
            for x1, y1, x2, y2, conf, _ in self.persons:
                cv2.rectangle(out, (x1, y1), (x2, y2), CYAN, 1)
                cv2.putText(out, f"{conf:.2f}", (x1 + 3, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CYAN, 1, cv2.LINE_AA)
            for x1, y1, x2, y2 in self.visitors:
                cv2.rectangle(out, (x1, y1), (x2, y2), ORANGE, 2)
                cv2.putText(out, "visitor", (x1 + 3, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 1, cv2.LINE_AA)
        for n, (x1, y1, x2, y2) in (self.view_boxes.items() if boxes else ()):
            cv2.rectangle(out, (x1, y1), (x2, y2), LIME, 2)
            (tw, th), _ = cv2.getTextSize(str(n), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(out, (x1 + 4, y1 + 4), (x1 + 12 + tw, y1 + 12 + th), (0, 0, 0), -1)
            cv2.putText(out, str(n), (x1 + 8, y1 + 8 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.9, YELLOW, 2, cv2.LINE_AA)
        return cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()

    def jpeg(self, people, boxes=True):
        """Encode each frame at most once per view mode, however many viewers ask for it."""
        t, key = self.frame_t, (people, boxes)
        if self._cache.get(key, (None,))[0] != t:
            self._cache[key] = (t, self.render(people, boxes))
        return self._cache[key]


def log_event(cam, box, ev):
    state, since, stretch = ev
    EVENTS.append({"t": since, "cam": cam.name, "box": box, "state": state, "stretch": stretch})
    f = HERE / "events" / f"{datetime.now():%Y-%m-%d}.csv"
    f.parent.mkdir(exist_ok=True)
    new = not f.exists()
    with f.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["time", "camera", "box", "state", "occupied_stretch_s"])
        w.writerow([datetime.fromtimestamp(since).isoformat(timespec="seconds"), cam.id, box, state,
                    "" if stretch is None else round(stretch)])


def detect_forever(model, cams):
    while True:
        t0 = time.time()
        live = [c for c in cams if c.online(t0)]
        try:
            if live:
                frames = [c.frame for c in live]   # the exact frames detected on, for the vest check too
                results = model.predict(frames, classes=[0], conf=CONF, imgsz=IMGSZ, half=True, verbose=False)
                now = time.time()
                for c, f, r in zip(live, frames, results):
                    c.apply(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), now, f)
                STATS["det_ms"], STATS["det_at"] = round((now - t0) * 1000), now
        except Exception as e:   # keep monitoring; the dashboard shows "stalled" if this repeats
            print(f"detection failed: {e!r}", flush=True)
            time.sleep(2)
        time.sleep(max(0.0, DET_PERIOD - (time.time() - t0)))


def gpu_forever():
    q = "temperature.gpu,clocks.sm,clocks.max.sm,utilization.gpu,clocks_throttle_reasons.active"
    while True:
        try:
            out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout.split(",")
            temp, clk, clk_max, util, reasons = (v.strip() for v in out)
            STATS["gpu"] = {"temp": int(temp), "clock": int(clk), "clock_max": int(clk_max),
                            "util": int(util), "hot": bool(int(reasons, 16) & 0x68)}  # SW/HW thermal, HW slowdown
        except Exception as e:   # the dashboard just shows no GPU numbers
            STATS["gpu"] = {"error": str(e)}
        time.sleep(5)


def minute_log_forever(cams):
    """Each full minute, append every desk's state to states/<date>.jsonl -- the source of the
    reports. Skipped while the camera or detector is down, so that time reads as "no data"."""
    while True:
        time.sleep(60 - time.time() % 60 + 1)
        now = time.time()
        if STATS["det_at"] is None or now - STATS["det_at"] > 10:
            continue
        dt = datetime.now()
        for c in cams:
            if not c.online(now):
                continue
            f = HERE / "states" / f"{dt:%Y-%m-%d}.jsonl"
            f.parent.mkdir(exist_ok=True)
            line = {"t": dt.hour * 3600 + dt.minute * 60 + dt.second, "cam": c.id,
                    "desks": {str(n): c.desk_state(n) for n in c.desks}}
            with f.open("a") as fh:
                fh.write(json.dumps(line) + "\n")


def report_for(cams, date, period):
    desks = [str(n) for n in cams[0].desks]
    dates = report.week_dates(date) if period == "week" else [date]
    days = {d: report.day_report(report.load(HERE / "states" / f"{d}.jsonl"), desks) for d in dates}
    total = {k: {f: sum(days[d][k][f] for d in dates) for f in ("at_desk", "working", "not", "vacant", "no_data", "times_left")}
             for k in desks}
    for k in desks:
        total[k]["longest"] = max(days[d][k]["longest"] for d in dates)
    three_state = any(days[d][k]["working"] or days[d][k]["not"] for d in dates for k in desks)
    hhmm = lambda s: f"{s // 3600:02d}:{s % 3600 // 60:02d}"
    return {"date": date, "period": period, "dates": dates, "open": hhmm(report.OPEN_S), "close": hhmm(report.CLOSE_S), "open_s": report.OPEN_S,
            "minutes_per_day": report.MINUTES, "three_state": three_state, "desks": total, "days": days}


def state(cams):
    now = time.time()
    return {"now": now, "started": STATS["started"], "det_ms": STATS["det_ms"], "det_age": now - STATS["det_at"] if STATS["det_at"] else None,
            "gpu": STATS["gpu"], "events": list(EVENTS)[::-1],
            "cameras": [{"id": c.id, "name": c.name, "ch": c.channel, "online": c.online(now), "status": c.status,
                         "boxes": {n: [round(b[0] / SRC_W * 100, 2), round(b[1] / 14.4, 2), round((b[2] - b[0]) / SRC_W * 100, 2),
                                       round((b[3] - b[1]) / 14.4, 2)] for n, b in c.boxes.items()},
                         "mode": ("seat classifier" if c.seat else "85% box rule") + (" + activity beta" if c.act else ""),
                         "desks": [{"no": n, "state": c.desk_state(n), "since": d.since, "people": sum(p[5] == n for p in c.persons),
                                    "p_sitting": round(c.p_sitting[n], 3) if n in c.p_sitting else None,
                                    "p_not": round(c.p_not[n], 3) if n in c.p_not else None,
                                    "occupied_s": d.occupied_total(now), "departures": d.departures}
                                   for n, d in c.desks.items()]} for c in cams]}


def make_handler(cams):
    by_id = {c.id: c for c in cams}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"   # keep-alive: one connection carries every frame request

        def log_message(self, *a):
            pass

        def send(self, code, body, ctype, extra=()):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in extra:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path, _, query = self.path.partition("?")
            q = dict(p.split("=", 1) for p in query.split("&") if "=" in p)
            if path == "/":
                return self.send(200, (HERE / "web" / "live.html").read_bytes(), "text/html")
            if path == "/reports":
                return self.send(200, (HERE / "web" / "reports.html").read_bytes(), "text/html")
            if path in ("/api/report", "/api/report.csv"):
                date = q.get("date") or datetime.now().strftime("%Y-%m-%d")
                period = "week" if q.get("period") == "week" else "day"
                try:
                    datetime.strptime(date, "%Y-%m-%d")
                except ValueError:
                    return self.send(400, b"date must be YYYY-MM-DD", "text/plain")
                r = report_for(cams, date, period)
                if path.endswith(".csv"):
                    body = "".join(report.to_csv(r["days"][d], d).split("\n", 1)[1] if i else report.to_csv(r["days"][d], d)
                                   for i, d in enumerate(r["dates"]))
                    return self.send(200, body.encode(), "text/csv",
                                     [("Content-Disposition", f'attachment; filename="desk-report-{period}-{date}.csv"')])
                return self.send(200, json.dumps(r).encode(), "application/json")
            if path == "/state":
                return self.send(200, json.dumps(state(cams)).encode(), "application/json")
            if path.startswith("/frame/") and path[7:] in by_id:
                return self.frame(by_id[path[7:]], float(q.get("after", 0)), q.get("people") == "1", q.get("boxes") != "0")
            self.send(404, b"not found", "text/plain")

        def frame(self, cam, after, people, boxes=True):
            """Pull, not push: the browser asks for the next frame only after showing the
            previous one, so a slow link lowers the frame rate instead of building a backlog."""
            deadline = time.time() + 2
            while (cam.frame is None or cam.frame_t <= after) and time.time() < deadline:
                time.sleep(0.02)   # wait for a newer frame than the one the browser has
            if cam.frame is None:
                return self.send(503, b"no frame yet", "text/plain")
            t, jpg = cam.jpeg(people, boxes)
            self.send(200, jpg, "image/jpeg", [("X-Frame-T", repr(t))])

    return Handler


def main():
    load_env()
    cams = [Camera(cid, ch, os.environ.get("CAMERA_NAME", name)) for cid, (ch, name) in CAMERAS.items()]
    model = YOLO(str(HERE / "models" / "yolo26m.pt"))
    for c in cams:
        threading.Thread(target=c.read_forever, daemon=True).start()
    threading.Thread(target=detect_forever, args=(model, cams), daemon=True).start()
    threading.Thread(target=gpu_forever, daemon=True).start()
    threading.Thread(target=minute_log_forever, args=(cams,), daemon=True).start()
    print(f"dashboard on 127.0.0.1:{PORT} -- " + ", ".join(f"{c.id}: boxes {list(c.boxes)}" for c in cams), flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), make_handler(cams)).serve_forever()


if __name__ == "__main__":
    main()
