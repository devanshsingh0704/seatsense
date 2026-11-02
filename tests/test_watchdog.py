import subprocess
import threading
import time
import unittest

from seatsense.watchdog import watch


class Watch(unittest.TestCase):
    def test_kills_a_reader_that_stops_delivering_frames(self):
        proc = subprocess.Popen(["sleep", "30"])               # a hung stream: alive, but no frames
        started = time.time()
        watch(proc, last_frame=lambda: started, stall_s=1, poll_s=0.2)
        self.assertIsNotNone(proc.poll())
        self.assertLess(time.time() - started, 5)

    def test_leaves_a_healthy_reader_alone(self):
        proc = subprocess.Popen(["sleep", "2"])
        t = threading.Thread(target=watch, args=(proc,), kwargs=dict(last_frame=time.time, stall_s=1, poll_s=0.2))
        t.start()
        time.sleep(1.2)
        self.assertIsNone(proc.poll())                          # still running: frames keep "arriving"
        proc.wait()
        t.join(3)


if __name__ == "__main__":
    unittest.main()
