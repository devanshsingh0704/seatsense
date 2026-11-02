"""Desk crops, shared by the live models (seat.py, activity.py) and their training scripts."""


def crop_box(b, w, h, margin=0.2):
    """The box (x1, y1, x2, y2) plus a margin on every side, so the chair and desk show; clamped to w x h."""
    x1, y1, x2, y2 = b
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    return (max(0, round(x1 - mx)), max(0, round(y1 - my)), min(w, round(x2 + mx)), min(h, round(y2 + my)))


def letterbox_size(w, h, size):
    """w x h scaled to fit a size x size square without stretching."""
    s = size / max(w, h)
    return round(w * s), round(h * s)
