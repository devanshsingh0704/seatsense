import unittest

from train_posture import letterbox_size, split_by_time


class SplitByTime(unittest.TestCase):
    def test_whole_time_blocks_go_to_validation_never_single_crops(self):
        items = [{"cam": "d20", "t": t, "label": "sit"} for t in range(0, 600, 20)]
        train, val = split_by_time(items, block_s=120, every=3)
        val_blocks = {it["t"] // 120 for it in val}
        train_blocks = {it["t"] // 120 for it in train}
        self.assertFalse(val_blocks & train_blocks)            # no block is split across both
        self.assertEqual(len(train) + len(val), len(items))
        self.assertTrue(val)

    def test_blocks_are_per_camera(self):
        items = [{"cam": c, "t": 0, "label": "sit"} for c in ("d20", "d3")]
        train, val = split_by_time(items, block_s=120, every=1)
        self.assertEqual(len(val), 2)


class Letterbox(unittest.TestCase):
    def test_keeps_the_aspect_ratio_inside_the_square(self):
        self.assertEqual(letterbox_size(100, 200, 224), (112, 224))   # tall person stays tall
        self.assertEqual(letterbox_size(300, 150, 224), (224, 112))


if __name__ == "__main__":
    unittest.main()
