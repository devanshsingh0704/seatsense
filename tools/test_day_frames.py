import unittest

from day_frames import is_sample, segment_start


class SegmentStart(unittest.TestCase):
    def test_file_name_gives_the_second_of_the_day_it_started(self):
        self.assertEqual(segment_start("d20_090004.mkv"), 9 * 3600 + 4)
        self.assertEqual(segment_start("d3_161715.mkv"), 16 * 3600 + 17 * 60 + 15)


class IsSample(unittest.TestCase):
    def test_keeps_the_frame_on_each_full_minute_of_the_clock(self):
        start = 12 * 3600 + 4 * 60 + 2                      # file starts 12:04:02, frames 1 per second
        kept = [k for k in range(200) if is_sample(start, k, every=60)]
        self.assertEqual(kept, [58, 118, 178])               # 12:05:00, 12:06:00, 12:07:00


if __name__ == "__main__":
    unittest.main()
