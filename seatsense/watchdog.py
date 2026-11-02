"""Kill a stream reader that is alive but has stopped delivering frames.

An RTSP stream that dies without closing leaves ffmpeg waiting forever: no frames, no exit,
so the read loop never reconnects (D20 froze this way from 18 to 22 Oct 2026). Run watch()
in a thread beside the reader: once no frame has arrived for stall_s, the process is killed,
the read returns, and the normal reconnect path takes over.
"""
import time


def watch(proc, last_frame, stall_s=15, poll_s=2):
    started = time.time()
    while proc.poll() is None:
        if time.time() - max(last_frame(), started) > stall_s:
            proc.kill()
            proc.wait()
            return True
        time.sleep(poll_s)
    return False
