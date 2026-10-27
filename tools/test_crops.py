import unittest

from crops import crop_box, letterbox_size


class CropBox(unittest.TestCase):
    def test_adds_a_margin_so_the_chair_and_desk_show(self):
        self.assertEqual(crop_box((100, 100, 200, 300), 2560, 1440, margin=0.2), (80, 60, 220, 340))

    def test_margin_is_clamped_to_the_frame(self):
        self.assertEqual(crop_box((0, 0, 100, 100), 2560, 1440, margin=0.2), (0, 0, 120, 120))


class LetterboxSize(unittest.TestCase):
    def test_keeps_the_shape(self):
        self.assertEqual(letterbox_size(100, 200, 224), (112, 224))   # tall person stays tall
        self.assertEqual(letterbox_size(300, 150, 224), (224, 112))


if __name__ == "__main__":
    unittest.main()
