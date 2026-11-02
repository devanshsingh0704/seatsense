"""Visitor check: visitors always wear a fluorescent yellow-green "VISITOR" vest (site policy).

Only the centre of the torso is measured, so yellow cabinets or folders beside the person
don't count. Measured on real crops: people without a vest score 0.000-0.005, people in
the vest 0.047-0.30 (fully hidden vests score 0 -- the 15 s rule covers those).
"""
import cv2

VEST_LO, VEST_HI = (25, 110, 140), (50, 255, 255)   # OpenCV HSV: fluorescent yellow-green
MIN_VEST = 0.03


def vest_fraction(img, box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    torso = img[int(y1 + 0.15 * h):int(y1 + 0.6 * h), int(x1 + 0.2 * w):int(x2 - 0.2 * w)]
    if torso.size == 0:
        return 0.0
    return float((cv2.inRange(cv2.cvtColor(torso, cv2.COLOR_BGR2HSV), VEST_LO, VEST_HI) > 0).mean())


def is_visitor(img, box):
    return vest_fraction(img, box) >= MIN_VEST
