"""Desk-presence logic, no dependencies: focus boxes, person->box assignment, occupancy."""
import json

# A person counts for a box only if this much of them is inside it. Seated people sit
# almost fully inside a box drawn round the seat (measured 0.89-1.00 on D20); people
# standing in front of a desk stick out of it (at most 0.77 on D3). A chair check was
# tried and rejected: a seated person hides their own chair from this camera angle.
MIN_INSIDE = 0.85


def load_boxes(path):
    """Read a <cam>_focus.json; return {box number: (x1, y1, x2, y2)} in source pixels.

    Numbers follow the file order (1-based) so they match what the user marked;
    an exact repeat of an earlier box (ENTER pressed twice) is dropped.
    """
    out = {}
    for n, r in enumerate(json.loads(path.read_text())["regions"], 1):
        if tuple(r) not in out.values():
            out[n] = tuple(r)
    return out


def _inside_fraction(p, b):
    w = min(p[2], b[2]) - max(p[0], b[0])
    h = min(p[3], b[3]) - max(p[1], b[1])
    area = (p[2] - p[0]) * (p[3] - p[1])
    return max(w, 0) * max(h, 0) / area if area > 0 else 0.0


def assign(persons, boxes, min_inside=MIN_INSIDE):
    """Give each person to the one box that holds most of them; {box: [person index]}."""
    out = {n: [] for n in boxes}
    for i, p in enumerate(persons):
        frac, best = max(((_inside_fraction(p, b), n) for n, b in boxes.items()), default=(0, None))
        if frac >= min_inside:
            out[best].append(i)
    return out


def seat_present(p_sitting, visitor_inside, staff_inside, threshold=0.5):
    """Seat-classifier decision for one box. A visitor (hi-vis vest) alone in the box never
    occupies it; with a staff member also there, the classifier decides as usual."""
    if visitor_inside and not staff_inside:
        return False
    return p_sitting >= threshold


class Desk:
    """Occupied/vacant with hysteresis, so one missed detection never flips the state.

    Occupied after `enter_s` of continuous presence, counted from the first sighting --
    long enough that someone walking past or standing by the desk does not count (a 3 s
    rule flagged D3 box 8 from 7-9 s visits). A gap of up to `gap_s` without a detection
    does not break that streak. Vacant after `leave_s` without a sighting (10 s: the user wanted
    departures shown faster than the original 20 s; on screen it is about 11-13 s).
    """

    def __init__(self, enter_s=15, leave_s=10, gap_s=3):
        self.enter_s, self.leave_s, self.gap_s = enter_s, leave_s, gap_s
        self.state, self.since = "vacant", None
        self.first_seen = self.last_seen = None
        self.occupied_s, self.departures = 0, 0

    def update(self, present, now):
        """Feed one detection result; return (new_state, since, finished_stretch_s) on a change."""
        gap = self.last_seen is None or now - self.last_seen > self.gap_s
        if present:
            if self.first_seen is None or gap:
                self.first_seen = now
            self.last_seen = now
        elif gap:
            self.first_seen = None
        if self.state == "vacant" and present and now - self.first_seen >= self.enter_s:
            self.state, self.since = "occupied", self.first_seen
            return ("occupied", self.since, None)
        if self.state == "occupied" and not present and now - self.last_seen >= self.leave_s:
            stretch = self.last_seen - self.since
            self.occupied_s += stretch
            self.departures += 1
            self.state, self.since = "vacant", self.last_seen
            return ("vacant", self.since, stretch)
        return None

    def occupied_total(self, now):
        return self.occupied_s + (now - self.since if self.state == "occupied" else 0)
