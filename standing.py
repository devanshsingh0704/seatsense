"""Standing veto from body pose: a desk is not occupied if everyone in its box is standing.

leg_ratio = (knee height below hip) / (hip height below shoulder), in image pixels.
Seated: thighs roughly level, so knees sit near hip height (small ratio). Standing: knees
far below the hips (ratio near 1). Unknown (None) when hips, knees or shoulders aren't seen
clearly -- then the veto never fires, so a hidden sitter is never thrown away.
"""
from presence import _inside_fraction

KP_CONF = 0.5
MIN_INSIDE = 0.3


def leg_ratio(p):
    kp = p["kp"]
    best = None
    for s, h, k in ((5, 11, 13), (6, 12, 14)):         # left side, right side
        if min(kp[s][2], kp[h][2], kp[k][2]) < KP_CONF:
            continue
        torso = kp[h][1] - kp[s][1]
        if torso <= 0:
            continue
        r = (kp[k][1] - kp[h][1]) / torso
        best = r if best is None else min(best, r)       # the more seated-looking side wins
    return best


def veto(people, box, threshold):
    inside = [p for p in people if _inside_fraction(p["box"], box) >= MIN_INSIDE]
    if not inside:
        return False
    ratios = [leg_ratio(p) for p in inside]
    return all(r is not None and r >= threshold for r in ratios)
